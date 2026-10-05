from __future__ import annotations

import asyncio
import unittest

from asr_lab.asr_service_lifecycle import QwenServiceLifecycleRegistry
from asr_lab.config import Settings
from asr_lab.qwen_streaming import QwenStreamingRuntime, StreamingError


class LifecycleWorker:
    def __init__(self) -> None:
        self.ready = False
        self.streams: dict[str, dict[str, object]] = {}
        self.requests: list[tuple[str, dict[str, object]]] = []

    async def start(self, **_options):
        self.ready = True
        return {"model_load_ms": 1.0, "vram_after_warmup": {"available": False}}

    async def request(self, operation: str, **payload):
        self.requests.append((operation, dict(payload)))
        stream_id = payload.get("stream_id")
        if operation == "init":
            assert isinstance(stream_id, str)
            self.streams[stream_id] = {"text": "hola mundo"}
            return {"stream_id": stream_id, "stream_state_init_wall_ms": 1.0}
        if operation in {"push", "finish"}:
            assert isinstance(stream_id, str)
            response = {
                "scheduler_key": payload.get("scheduler_key"),
                "decoded": True,
                "decode_wall_ms": 2.0,
                "decode_steps_delta": 1,
                "text": self.streams[stream_id]["text"],
                "language": "Spanish",
            }
            return response
        if operation == "close":
            self.streams.pop(stream_id, None)
            return {"closed": True}
        raise AssertionError(operation)

    async def close(self) -> None:
        self.ready = False
        self.streams.clear()


class QwenServiceLifecycleRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.worker = LifecycleWorker()
        self.runtime = QwenStreamingRuntime(
            worker=self.worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="fixed",
            gpu_memory_utilization=0.65,
            max_active_sessions=6,
            max_pending_jobs=24,
            max_backlog_chunks=4,
            max_stream_seconds=0.25,
            session_idle_ttl_seconds=120.0,
            max_context_chars=512,
            default_chunk_ms=250,
            default_language="auto",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        await self.runtime.start()
        self.events: list[dict[str, object]] = []
        self.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet",
            "ASR_API_TOKEN": "t" * 32,
            "ASR_MAX_CHUNK_SECONDS": "1",
        })

        async def collect_event(event: dict[str, object]) -> None:
            self.events.append(dict(event))

        self.registry = QwenServiceLifecycleRegistry(
            runtime=self.runtime,
            settings=self.settings,
            connection_id="connection-test",
            event_sink=collect_event,
        )

    async def asyncTearDown(self) -> None:
        await self.registry.close_connection()
        await self.runtime.close()

    async def wait_until(self, predicate, message: str) -> None:
        for _ in range(1_000):
            if predicate():
                return
            await asyncio.sleep(0.001)
        self.fail(message)

    async def test_default_open_rolls_two_local_epochs_under_one_public_stream(self):
        started = await self.registry.open_default_source("mic")
        self.assertIsNotNone(started)
        assert started is not None
        public_stream_id = started["stream_id"]
        self.assertEqual(started["qwen_local_stream_id"], public_stream_id)
        self.assertIsNone(await self.registry.open_default_source("mic"))
        with self.assertRaises(StreamingError) as duplicate:
            await self.registry.open_source(source="mic")
        self.assertEqual(duplicate.exception.code, "stream_already_active")

        pcm = b"\x11\x00" * 4_000
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) == 1, "initial epoch candidate missing")
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 2, "first successor candidate missing")
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 3, "second successor candidate missing")

        final = await self.registry.finish(source="mic", request_id="flush-test")
        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(final["event"], "final_candidate")
        self.assertEqual(final["stream_id"], public_stream_id)
        self.assertEqual(final["request_id"], "flush-test")
        self.assertEqual(final["client_request_id"], "flush-test")
        self.assertEqual(final["truth_status"], "candidate_only")

        local_ids = {event["qwen_local_stream_id"] for event in self.events}
        local_ids.add(final["qwen_local_stream_id"])
        self.assertEqual(len(local_ids), 3)
        self.assertEqual({event["stream_id"] for event in self.events}, {public_stream_id})
        self.assertEqual(final["qwen_epoch_observability"]["current_epoch"]["epoch_seq"], 2)
        self.assertEqual(
            final["qwen_epoch_observability"]["logical_cumulative"]["EPOCH_ROLLOVER_COUNT"],
            2,
        )
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 1)
        self.assertFalse(self.registry.has_active_source("mic"))
        self.assertEqual(await self.registry.finish(source="mic", request_id="flush-again"), None)
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 1)

    async def test_registry_manual_rollovers_keep_one_source_until_natural_eos(self):
        # Keep the runtime hard bound above each fixture chunk so these two
        # transitions exercise the registry-owned manual/test seam directly.
        self.runtime.max_stream_seconds = 2.0
        with self.assertRaises(StreamingError) as inactive:
            await self.registry.rollover_source("mic")
        self.assertEqual(inactive.exception.code, "stream_not_started")

        started = await self.registry.open_default_source("mic")
        self.assertIsNotNone(started)
        assert started is not None
        public_stream_id = started["stream_id"]
        epoch_a = started["qwen_local_stream_id"]
        entry = self.registry._sources["mic"]
        logical_lifecycle = entry.lifecycle
        self.assertIsNotNone(logical_lifecycle)

        pcm = b"\x11\x00" * 4_000
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 1, "epoch A candidate missing")
        self.assertEqual(self.events[0]["qwen_local_stream_id"], epoch_a)

        first_rollover = await self.registry.rollover_source("mic")
        self.assertTrue(first_rollover.completed)
        self.assertTrue(self.registry.has_active_source("mic"))
        self.assertIs(self.registry._sources["mic"].lifecycle, logical_lifecycle)
        epoch_b = logical_lifecycle.local_stream_id
        self.assertNotEqual(epoch_a, epoch_b)
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 0)

        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 2, "epoch B candidate missing")
        self.assertEqual(self.events[1]["qwen_local_stream_id"], epoch_b)

        second_rollover = await self.registry.rollover_source("mic")
        self.assertTrue(second_rollover.completed)
        self.assertTrue(self.registry.has_active_source("mic"))
        self.assertIs(self.registry._sources["mic"].lifecycle, logical_lifecycle)
        epoch_c = logical_lifecycle.local_stream_id
        self.assertNotEqual(epoch_b, epoch_c)
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 0)

        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 3, "epoch C candidate missing")
        self.assertEqual(self.events[2]["qwen_local_stream_id"], epoch_c)
        self.assertEqual({epoch_a, epoch_b, epoch_c}, {event["qwen_local_stream_id"] for event in self.events[:3]})
        self.assertEqual({event["stream_id"] for event in self.events[:3]}, {public_stream_id})

        before_eos = logical_lifecycle.observability_snapshot()
        self.assertEqual(before_eos["current_epoch"]["epoch_seq"], 2)
        self.assertEqual(before_eos["logical_cumulative"]["EPOCH_ROLLOVER_COUNT"], 2)
        self.assertTrue(self.registry.has_active_source("mic"))
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 0)

        final = await self.registry.finish(source="mic", request_id="natural-eos")
        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(final["event"], "final_candidate")
        self.assertEqual(final["stream_id"], public_stream_id)
        self.assertEqual(final["qwen_local_stream_id"], epoch_c)
        self.assertEqual(final["qwen_epoch_observability"]["current_epoch"]["epoch_seq"], 2)
        self.assertEqual(final["qwen_epoch_observability"]["logical_cumulative"]["EPOCH_ROLLOVER_COUNT"], 2)
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 1)
        self.assertFalse(self.registry.has_active_source("mic"))

    async def test_disconnect_disposes_without_eos_and_suppresses_late_events(self):
        started = await self.registry.open_default_source("mic")
        self.assertIsNotNone(started)
        assert started is not None
        before_close = len(self.events)
        session = self.runtime.sessions[("connection-test", "mic")]
        late_event_sink = session.event_sink
        self.assertIsNotNone(late_event_sink)

        await self.registry.close_connection()
        self.assertEqual(self.runtime.sessions, {})
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 0)

        # Model a worker callback racing after teardown; the detached source
        # cannot enqueue a stale candidate into the closed connection.
        await late_event_sink({
            "event": "partial_candidate",
            "stream_id": started["qwen_local_stream_id"],
            "text": "late candidate",
            "final": False,
            "provisional": True,
            "replace": True,
            "truth_status": "candidate_only",
            "revision": 1,
        })
        self.assertEqual(len(self.events), before_close)
        self.assertFalse(self.registry.has_active_source("mic"))


if __name__ == "__main__":
    unittest.main()
