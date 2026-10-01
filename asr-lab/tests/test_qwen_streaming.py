from __future__ import annotations

import asyncio
import base64
import unittest

from asr_lab.qwen_streaming import (
    QWEN_LANGUAGE_NAMES, QwenStreamingRuntime, StreamingError,
    token_revision_metrics, validate_candidate_event,
)


class FakeWorker:
    def __init__(self, text_sequences=None, finals=None):
        self.ready = False
        self.text_sequences = text_sequences or {}
        self.finals = finals or {}
        self.streams = {}
        self.requests = []

    async def start(self, **_options):
        self.ready = True
        return {"model_load_ms": 1.0, "vram_after_warmup": {"available": False}}

    async def request(self, operation, **payload):
        self.requests.append((operation, payload.copy()))
        stream_id = payload.get("stream_id")
        if operation == "init":
            self.streams[stream_id] = {
                "context": payload["context"],
                "index": 0,
                "text": "",
            }
            return {"stream_id": stream_id}
        if operation == "push":
            stream = self.streams[stream_id]
            sequence = self.text_sequences.get(stream["context"], [""])
            index = min(stream["index"], len(sequence) - 1)
            stream["text"] = sequence[index]
            stream["index"] += 1
            return {
                "decoded": True,
                "decode_wall_ms": 4.5 + index,
                "text": stream["text"],
                "language": "es",
            }
        if operation == "finish":
            stream = self.streams[stream_id]
            stream["text"] = self.finals.get(stream["context"], stream["text"])
            return {"decode_wall_ms": 1.25, "text": stream["text"], "language": "es"}
        if operation == "close":
            self.streams.pop(stream_id, None)
            return {"closed": True}
        raise AssertionError(operation)

    async def close(self):
        self.ready = False
        self.streams.clear()


class QwenStreamingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = FakeWorker(
            text_sequences={
                "client A": ["quiero reservar para cuatro", "quiero reservar para seis"],
                "client B": ["cancelar mi cita"],
                "zero partial": [""],
            },
            finals={
                "client A": "quiero reservar para seis mañana",
                "client B": "cancelar mi cita",
                "zero partial": "final hypothesis only",
            },
        )
        self.runtime = QwenStreamingRuntime(
            worker=self.worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="fixed",
            gpu_memory_utilization=0.65,
            max_active_sessions=2,
            max_stream_seconds=2.0,
            session_idle_ttl_seconds=120.0,
            max_context_chars=512,
            default_chunk_ms=1000,
            default_language="auto",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        await self.runtime.start()

    async def asyncTearDown(self):
        await self.runtime.close()

    async def test_connection_and_source_own_distinct_stream_state_and_results(self):
        first = await self.runtime.open_session(
            connection_id="conn-a", source="mic", context="client A", chunk_size_ms=250,
        )
        second = await self.runtime.open_session(
            connection_id="conn-b", source="mic", context="client B", chunk_size_ms=500,
        )
        self.assertNotEqual(first["stream_id"], second["stream_id"])

        partial_a = await self.runtime.push_audio(
            connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 1600,
        )
        partial_b = await self.runtime.push_audio(
            connection_id="conn-b", source="mic", pcm16le=b"\0\0" * 1600,
        )
        self.assertEqual(partial_a["text"], "quiero reservar para cuatro")
        self.assertEqual(partial_b["text"], "cancelar mi cita")
        self.assertNotEqual(partial_a["stream_id"], partial_b["stream_id"])
        with self.assertRaisesRegex(StreamingError, "stream_not_started"):
            await self.runtime.push_audio(
                connection_id="conn-a", source="system", pcm16le=b"\0\0" * 1600,
            )

        final_a = await self.runtime.finish(connection_id="conn-a", source="mic", request_id="eos-a")
        final_b = await self.runtime.finish(connection_id="conn-b", source="mic", request_id="eos-b")
        self.assertEqual(final_a["text"], "quiero reservar para seis mañana")
        self.assertEqual(final_b["text"], "cancelar mi cita")
        self.assertTrue(final_a["final"])
        self.assertEqual(final_a["event"], "final_candidate")
        self.assertNotEqual(final_a.get("event"), "transcript")
        self.assertNotEqual(final_a.get("type"), "transcript")
        self.assertNotIn("type", final_a)
        self.assertEqual(final_a["truth_status"], "candidate_only")
        self.assertTrue(final_a["candidate_only"])
        self.assertEqual(final_a["request_id"], "eos-a")

    async def test_partial_is_a_versioned_replacement_and_final_revision_advances(self):
        await self.runtime.open_session(connection_id="conn-a", source="mic", context="client A")
        first = await self.runtime.push_audio(
            connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 800,
        )
        second = await self.runtime.push_audio(
            connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 800,
        )
        self.assertFalse(first["final"])
        self.assertTrue(first["provisional"])
        self.assertTrue(first["replace"])
        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)
        self.assertEqual(first["event"], "partial_candidate")
        self.assertNotEqual(first.get("event"), "transcript")
        self.assertNotEqual(first.get("type"), "transcript")
        self.assertAlmostEqual(second["PARTIAL_REVISION_RATE"], 0.25)
        self.assertAlmostEqual(second["PARTIAL_STABILITY"], 0.75)
        final = await self.runtime.finish(connection_id="conn-a", source="mic")
        self.assertTrue(final["final"])
        self.assertEqual(final["revision"], 3)
        self.assertGreaterEqual(final["FINALIZATION_AFTER_SERVER_EOS_MS"], 0)
        self.assertEqual(final["SERVER_EOS_TO_FINAL_CANDIDATE_MS"], final["FINALIZATION_AFTER_SERVER_EOS_MS"])
        self.assertNotIn("start", final)
        self.assertNotIn("end", final)
        self.assertAlmostEqual(final["QWEN_STREAM_RTF"], final["QWEN_CUMULATIVE_DECODE_WALL_MS"] / 100.0, places=3)

    async def test_final_only_stream_does_not_invent_a_first_partial(self):
        await self.runtime.open_session(connection_id="conn-zero", source="mic", context="zero partial")
        no_partial = await self.runtime.push_audio(
            connection_id="conn-zero", source="mic", pcm16le=b"\0\0" * 800,
        )
        self.assertIsNone(no_partial)
        final = await self.runtime.finish(connection_id="conn-zero", source="mic")
        self.assertEqual(final["event"], "final_candidate")
        self.assertEqual(final["text"], "final hypothesis only")
        self.assertEqual(final["PARTIAL_COUNT"], 0)
        self.assertIsNone(final["FIRST_PARTIAL_MS"])

    async def test_candidate_event_validator_rejects_transcript_truth_events(self):
        partial = {
            "event": "partial_candidate", "truth_status": "candidate_only",
            "provisional": True, "final": False,
        }
        self.assertIs(validate_candidate_event(partial, "partial_candidate"), partial)
        valid = {
            "event": "final_candidate", "truth_status": "candidate_only",
            "provisional": True, "final": True, "candidate_only": True,
        }
        self.assertIs(validate_candidate_event(valid, "final_candidate"), valid)
        with self.assertRaisesRegex(StreamingError, "invalid_candidate_event"):
            validate_candidate_event({**valid, "event": "transcript"}, "final_candidate")
        with self.assertRaisesRegex(StreamingError, "invalid_candidate_event"):
            validate_candidate_event({**valid, "type": "transcript"}, "final_candidate")

    async def test_iso_language_is_mapped_to_qwen_official_language_name(self):
        self.assertEqual(QWEN_LANGUAGE_NAMES["es"], "Spanish")
        await self.runtime.open_session(connection_id="conn-a", source="mic", language="es")
        init_payload = next(payload for operation, payload in self.worker.requests if operation == "init")
        self.assertEqual(init_payload["language"], "Spanish")

    async def test_worker_timeout_releases_owned_session_capacity(self):
        class FailingWorker(FakeWorker):
            async def request(self, operation, **payload):
                if operation == "push":
                    raise StreamingError("stream_worker_timeout")
                return await super().request(operation, **payload)

        worker = FailingWorker()
        worker.ready = True
        self.runtime.worker = worker
        await self.runtime.open_session(connection_id="conn-a", source="mic")
        with self.assertRaisesRegex(StreamingError, "stream_worker_failed"):
            await self.runtime.push_audio(
                connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 1600,
            )
        self.assertEqual(self.runtime.active_sessions, 0)

    async def test_late_partial_from_disconnected_connection_is_discarded(self):
        class DelayedWorker(FakeWorker):
            def __init__(self):
                super().__init__()
                self.push_started = asyncio.Event()
                self.release_push = asyncio.Event()

            async def request(self, operation, **payload):
                if operation == "push":
                    self.push_started.set()
                    await self.release_push.wait()
                    return {"decoded": True, "decode_wall_ms": 3, "text": "stale result", "language": "Spanish"}
                return await super().request(operation, **payload)

        worker = DelayedWorker()
        runtime = QwenStreamingRuntime(
            worker=worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="fixed",
            gpu_memory_utilization=0.65,
            max_active_sessions=2,
            max_stream_seconds=2.0,
            session_idle_ttl_seconds=120.0,
            max_context_chars=512,
            default_chunk_ms=1000,
            default_language="auto",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        await runtime.start()
        try:
            await runtime.open_session(connection_id="dead-connection", source="mic")
            pending = asyncio.create_task(runtime.push_audio(
                connection_id="dead-connection", source="mic", pcm16le=b"\0\0" * 1600,
            ))
            await worker.push_started.wait()
            await runtime.close_connection("dead-connection")
            worker.release_push.set()
            self.assertIsNone(await pending)
            self.assertEqual(runtime.active_sessions, 0)
        finally:
            worker.release_push.set()
            await runtime.close()

    async def test_active_session_limit_and_disconnect_cleanup_are_enforced(self):
        await self.runtime.open_session(connection_id="conn-a", source="mic")
        await self.runtime.open_session(connection_id="conn-b", source="mic")
        with self.assertRaisesRegex(StreamingError, "stream_capacity_exceeded"):
            await self.runtime.open_session(connection_id="conn-c", source="mic")
        await self.runtime.close_connection("conn-a")
        self.assertEqual(self.runtime.active_sessions, 1)
        self.assertFalse(any(payload.get("stream_id") in self.worker.streams for op, payload in self.worker.requests if op == "close"))
        await self.runtime.open_session(connection_id="conn-c", source="mic")
        self.assertEqual(self.runtime.active_sessions, 2)

    async def test_duration_limit_closes_owned_session_and_backpressure_slot(self):
        self.runtime.max_stream_seconds = 0.1
        await self.runtime.open_session(connection_id="conn-a", source="mic")
        with self.assertRaisesRegex(StreamingError, "stream_duration_limit"):
            await self.runtime.push_audio(
                connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 1601,
            )
        self.assertEqual(self.runtime.active_sessions, 0)

    async def test_idle_ttl_expires_orphaned_session(self):
        self.runtime.session_idle_ttl_seconds = 0.3
        self.runtime._sweeper.cancel()
        await asyncio.gather(self.runtime._sweeper, return_exceptions=True)
        self.runtime._sweeper = asyncio.create_task(self.runtime._expire_loop())
        await self.runtime.open_session(connection_id="conn-a", source="mic")
        # The periodic reaper may sleep for up to 250 ms after the session opens.
        await asyncio.sleep(0.7)
        self.assertEqual(self.runtime.active_sessions, 0)

    def test_revision_metric_ignores_append_and_measures_rollback_of_previous_tokens(self):
        self.assertEqual(token_revision_metrics("", "quiero"), (None, None))
        self.assertEqual(token_revision_metrics("quiero mesa", "quiero mesa cuatro"), (0.0, 1.0))
        self.assertEqual(token_revision_metrics("quiero cuatro", "quiero seis"), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
