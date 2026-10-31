from __future__ import annotations

import asyncio
import json
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
            self.streams[stream_id] = {"text": f"candidate-{stream_id}"}
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
        self.partial_diagnostics: list[dict[str, object]] = []
        self.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet",
            "ASR_API_TOKEN": "t" * 32,
            "ASR_MAX_CHUNK_SECONDS": "1",
        })

        async def collect_event(event: dict[str, object]) -> None:
            self.events.append(dict(event))

        def collect_partial_diagnostic(boundary, public_stream_id, local_stream_id, candidate):
            self.partial_diagnostics.append({
                "boundary": boundary,
                "stream_id": public_stream_id,
                "qwen_local_stream_id": local_stream_id,
                "epoch_seq": candidate.get("epoch_seq"),
                "revision": candidate.get("revision", candidate.get("revision_new")),
                "partial_id": candidate.get("partial_id"),
                "decode_outcome": candidate.get("outcome"),
                "filter_outcome": candidate.get("diagnostic_outcome"),
                "cursor_end_samples": candidate.get("cursor_end_samples"),
                "decode_steps_delta": candidate.get("decode_steps_delta"),
                "revision_previous": candidate.get("revision_previous"),
                "revision_new": candidate.get("revision_new"),
                "diagnostic_payload": dict(candidate),
                "has_text": "text" in candidate,
            })

        self.registry = QwenServiceLifecycleRegistry(
            runtime=self.runtime,
            settings=self.settings,
            connection_id="connection-test",
            event_sink=collect_event,
            diagnostic_sink=collect_partial_diagnostic,
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

    async def test_partial_diagnostic_identity_stays_public_across_epoch_rollover(self):
        self.runtime.max_stream_seconds = 2.0
        started = await self.registry.open_default_source("mic")
        assert started is not None
        public_stream_id = started["stream_id"]
        first_local_id = started["qwen_local_stream_id"]
        stitched_text_sentinel = "STITCHED_TRANSCRIPT_SENTINEL_DO_NOT_LOG"
        self.worker.streams[first_local_id]["text"] = stitched_text_sentinel

        pcm = b"\x11\x00" * 4_000
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 1, "epoch 0 partial candidate missing")
        self.assertTrue((await self.registry.rollover_source("mic")).completed)
        second_local_id = self.registry._sources["mic"].lifecycle.local_stream_id
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) >= 2, "epoch 1 partial candidate missing")

        decode_records = [row for row in self.partial_diagnostics if row["boundary"] == "decode_push_result"]
        runtime_records = [row for row in self.partial_diagnostics if row["boundary"] == "runtime_candidate"]
        filter_records = [row for row in self.partial_diagnostics if row["boundary"] == "canonical_filter"]
        server_records = [row for row in self.partial_diagnostics if row["boundary"] == "server_event"]
        self.assertEqual(len(decode_records), 2)
        self.assertEqual(len(runtime_records), 2)
        self.assertEqual(len(filter_records), 2)
        self.assertEqual(len(server_records), 2)
        self.assertEqual(
            {row["stream_id"] for row in decode_records + runtime_records + filter_records + server_records},
            {public_stream_id},
        )
        self.assertEqual({row["qwen_local_stream_id"] for row in runtime_records}, {first_local_id, second_local_id})
        self.assertEqual(
            [(row["epoch_seq"], row["revision"]) for row in runtime_records],
            [(0, 1), (1, 1)],
        )
        self.assertEqual(
            [(row["epoch_seq"], row["revision"]) for row in server_records],
            [(0, 1), (1, 1)],
        )
        self.assertEqual(
            [(row["stream_id"], row["epoch_seq"], row["revision"]) for row in runtime_records],
            [(row["stream_id"], row["epoch_seq"], row["revision"]) for row in server_records],
        )
        self.assertEqual(
            [(row["epoch_seq"], row["revision"], row["decode_outcome"]) for row in decode_records],
            [(0, 1, "changed_text_candidate"), (1, 1, "changed_text_candidate")],
        )
        self.assertEqual([row["filter_outcome"] for row in filter_records], ["accepted", "accepted"])
        self.assertEqual(
            [row["partial_id"] for row in decode_records],
            [row["partial_id"] for row in runtime_records],
        )
        self.assertFalse(any(row["has_text"] for row in decode_records + runtime_records + filter_records + server_records))
        self.assertEqual(self.events[0]["stitch_raw_current_text"], stitched_text_sentinel)
        canonical_diagnostics = [row for row in filter_records if row["filter_outcome"] == "accepted"]
        self.assertEqual(len(canonical_diagnostics), 2)
        diagnostic_json = json.dumps(self.partial_diagnostics)
        self.assertNotIn("stitch_raw_current_text", diagnostic_json)
        self.assertNotIn(stitched_text_sentinel, diagnostic_json)

    async def test_unchanged_decode_has_no_candidate_identity_after_lifecycle_binding(self):
        self.runtime.max_stream_seconds = 2.0
        started = await self.registry.open_default_source("mic")
        assert started is not None
        public_stream_id = started["stream_id"]
        pcm = b"\x12\x00" * 4_000

        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(lambda: len(self.events) == 1, "first candidate missing")
        await self.registry.submit_pcm(source="mic", pcm16le=pcm)
        await self.wait_until(
            lambda: len([row for row in self.partial_diagnostics if row["boundary"] == "decode_push_result"]) == 2,
            "unchanged decode opportunity missing",
        )

        decode_records = [row for row in self.partial_diagnostics if row["boundary"] == "decode_push_result"]
        self.assertEqual(
            [(row["decode_outcome"], row["revision_previous"], row["revision_new"]) for row in decode_records],
            [("changed_text_candidate", 0, 1), ("unchanged_text", 1, 1)],
        )
        self.assertEqual(decode_records[0]["partial_id"], f"{public_stream_id}/0/1")
        self.assertIsNone(decode_records[1]["partial_id"])

    async def test_100ms_ingress_crosses_three_rollovers_with_pcm_and_identity_continuity(self):
        self.runtime.max_stream_seconds = 0.25
        self.runtime.default_chunk_ms = 100
        started = await self.registry.open_source(source="mic", model_chunk_ms=100)
        public_stream_id = started["stream_id"]
        lifecycle = self.registry._sources["mic"].lifecycle
        self.assertIsNotNone(lifecycle)
        # A small replay overlap is synthetic test data only. Production
        # registry wiring remains unchanged at zero overlap.
        lifecycle.pcm_handoff.replay_overlap_samples = 160

        frame = b"\x21\x00" * 1_600  # exactly 100 ms of 16 kHz PCM16LE
        next_frame_at = asyncio.get_running_loop().time()
        ingress_tasks = []
        for index in range(8):
            await asyncio.sleep(max(0.0, next_frame_at - asyncio.get_running_loop().time()))
            ingress_tasks.append(asyncio.create_task(
                self.registry.submit_pcm(source="mic", pcm16le=frame),
            ))
            next_frame_at += 0.1
        await asyncio.gather(*ingress_tasks)
        await self.wait_until(
            lambda: lifecycle.observability_snapshot()["current_epoch"]["scheduler_pending_jobs"] == 0,
            "decode jobs did not drain after the fixed-cadence synthetic ingress ended",
        )

        before_eos = lifecycle.observability_snapshot()
        transitions = before_eos["transition_history"]
        epochs = before_eos["epoch_history"]
        self.assertGreaterEqual(len(transitions), 3)
        self.assertEqual(before_eos["logical_cumulative"]["rollover_success_count"], len(transitions))
        self.assertEqual([epoch["epoch_seq"] for epoch in epochs], list(range(len(epochs))))
        self.assertGreaterEqual(before_eos["logical_cumulative"]["replay_admitted_samples"], 3 * 160)
        self.assertTrue(all(item["state"] == "ACTIVE" for item in transitions))
        self.assertTrue(all(item["stage"] == "ACTIVE" for item in transitions))
        self.assertTrue(all(item["PCM_LOST"] == 0 for item in transitions))
        self.assertTrue(all(item["PRIMARY_DUP"] == 0 for item in transitions))
        self.assertEqual(len({epoch["local_stream_id"] for epoch in epochs}), len(epochs))
        self.assertEqual([epoch["epoch_seq"] for epoch in epochs], [0, 1, 2, 3])
        self.assertEqual({event["stream_id"] for event in self.events}, {public_stream_id})
        for event in self.events:
            snapshot = event["qwen_epoch_observability"]
            current = snapshot["current_epoch"]
            self.assertEqual(event["qwen_local_stream_id"], current["local_stream_id"])
            self.assertEqual(event["stream_id"], public_stream_id)
            self.assertIn(current["epoch_seq"], range(4))
        pcm_before_eos = lifecycle.pcm_handoff.snapshot()
        self.assertEqual(pcm_before_eos.received_samples, 8 * 1_600)
        self.assertEqual(pcm_before_eos.unique_primary_admitted_samples, 8 * 1_600)
        self.assertEqual(pcm_before_eos.transition_queued_samples, 0)

        final = await self.registry.finish(source="mic", request_id="100ms-eos")
        self.assertIsNotNone(final)
        assert final is not None
        final_epochs = final["qwen_epoch_observability"]["epoch_history"]
        self.assertEqual(final_epochs[-1]["lifecycle_state"], "NATURAL_EOS")
        self.assertEqual(sum(operation == "finish" for operation, _ in self.worker.requests), 1)
        self.assertEqual(sum(operation == "close" for operation, _ in self.worker.requests), len(epochs))
        self.assertEqual(final["stream_id"], public_stream_id)
        self.assertEqual(final["qwen_local_stream_id"], epochs[-1]["local_stream_id"])

    async def test_error_snapshot_is_available_before_source_cleanup(self):
        started = await self.registry.open_default_source("mic")
        self.assertIsNotNone(started)
        before_abort = self.registry.source_observability_snapshot("mic")
        self.assertIsNotNone(before_abort)
        assert before_abort is not None
        await self.registry.abort_source("mic")
        self.assertIsNone(self.registry.source_observability_snapshot("mic"))
        self.assertEqual(before_abort["qwen_public_stream_id"], started["stream_id"])
        self.assertEqual(
            before_abort["qwen_epoch_observability"]["current_epoch"]["epoch_seq"], 0,
        )

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
