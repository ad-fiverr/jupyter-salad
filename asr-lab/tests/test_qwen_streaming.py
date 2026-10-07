from __future__ import annotations

import asyncio
import base64
import unittest
from unittest.mock import patch

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
            return {"stream_id": stream_id, "stream_state_init_wall_ms": 2.0}
        if operation == "push":
            stream = self.streams[stream_id]
            sequence = self.text_sequences.get(stream["context"], [""])
            index = min(stream["index"], len(sequence) - 1)
            stream["text"] = sequence[index]
            stream["index"] += 1
            response = {
                "decoded": True,
                "decode_wall_ms": 4.5 + index,
                "decode_steps_delta": 1,
                "text": stream["text"],
                "language": "es",
            }
        if operation == "finish":
            stream = self.streams[stream_id]
            stream["text"] = self.finals.get(stream["context"], stream["text"])
            response = {"decode_wall_ms": 1.25, "decode_steps_delta": 1, "text": stream["text"], "language": "es"}
        if operation in {"push", "finish"}:
            if payload.get("scheduler_key") is not None:
                response["scheduler_key"] = payload["scheduler_key"]
            return response
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
                "short stream": ["short candidate"],
                "empty stream": [""],
            },
            finals={
                "client A": "quiero reservar para seis mañana",
                "client B": "cancelar mi cita",
                "zero partial": "final hypothesis only",
                "empty stream": "",
            },
        )
        self.runtime = QwenStreamingRuntime(
            worker=self.worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="fixed",
            gpu_memory_utilization=0.65,
            max_active_sessions=2,
            max_pending_jobs=12,
            max_backlog_chunks=4,
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

        await self.runtime.push_audio(connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 4000)
        await self.runtime.push_audio(connection_id="conn-b", source="mic", pcm16le=b"\0\0" * 8000)
        partial_a = await self.runtime.receive_event(connection_id="conn-a", source="mic")
        partial_b = await self.runtime.receive_event(connection_id="conn-b", source="mic")
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
        await self.runtime.open_session(connection_id="conn-a", source="mic", context="client A", chunk_size_ms=250)
        await self.runtime.push_audio(connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 4000)
        first = await self.runtime.receive_event(connection_id="conn-a", source="mic")
        await self.runtime.push_audio(connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 4000)
        second = await self.runtime.receive_event(connection_id="conn-a", source="mic")
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
        self.assertEqual(first["SERVER_FIRST_PARTIAL_MS"], first["FIRST_PARTIAL_MS"])
        self.assertEqual(second["SERVER_FIRST_PARTIAL_MS"], first["SERVER_FIRST_PARTIAL_MS"])
        self.assertEqual(second["FIRST_PARTIAL_MS"], first["FIRST_PARTIAL_MS"])
        final = await self.runtime.finish(connection_id="conn-a", source="mic")
        self.assertTrue(final["final"])
        self.assertEqual(final["revision"], 3)
        self.assertGreaterEqual(final["FINALIZATION_AFTER_SERVER_EOS_MS"], 0)
        self.assertEqual(final["SERVER_EOS_TO_FINAL_CANDIDATE_MS"], final["FINALIZATION_AFTER_SERVER_EOS_MS"])
        self.assertNotIn("start", final)
        self.assertNotIn("end", final)
        self.assertAlmostEqual(final["QWEN_STREAM_RTF"], final["QWEN_CUMULATIVE_DECODE_WALL_MS"] / 500.0, places=3)

    async def test_final_only_stream_does_not_invent_a_first_partial(self):
        await self.runtime.open_session(connection_id="conn-zero", source="mic", context="zero partial", chunk_size_ms=250)
        await self.runtime.push_audio(connection_id="conn-zero", source="mic", pcm16le=b"\0\0" * 4000)
        final = await self.runtime.finish(connection_id="conn-zero", source="mic")
        self.assertEqual(final["event"], "final_candidate")
        self.assertEqual(final["text"], "final hypothesis only")
        self.assertEqual(final["PARTIAL_COUNT"], 0)
        self.assertIsNone(final["FIRST_PARTIAL_MS"])

    async def test_subwindow_speech_starts_worker_push_before_eos_without_turn_id(self):
        await self.runtime.open_session(
            connection_id="short", source="mic", context="short stream", model_chunk_ms=250,
        )
        pcm = b"\x17\x00" * 800
        await self.runtime.push_audio(connection_id="short", source="mic", pcm16le=pcm)
        partial = await self.runtime.receive_event(connection_id="short", source="mic")

        self.assertEqual(partial["event"], "partial_candidate")
        self.assertEqual(partial["text"], "short candidate")
        self.assertEqual(partial["truth_status"], "candidate_only")
        self.assertNotIn("turnId", partial)
        self.assertNotIn("turn_id", partial)
        operations_before_eos = [operation for operation, _ in self.worker.requests]
        self.assertIn("push", operations_before_eos)
        self.assertNotIn("finish", operations_before_eos)

        await self.runtime.finish(connection_id="short", source="mic")
        operations_after_eos = [operation for operation, _ in self.worker.requests]
        self.assertLess(operations_after_eos.index("push"), operations_after_eos.index("finish"))

    async def test_decode_audio_increment_and_epoch_accumulation_are_separate(self):
        await self.runtime.open_session(
            connection_id="cadence", source="mic", context="short stream", model_chunk_ms=100,
        )
        pcm = b"\x18\x00" * 1_600
        await self.runtime.push_audio(connection_id="cadence", source="mic", pcm16le=pcm)
        partial = await asyncio.wait_for(
            self.runtime.receive_event(connection_id="cadence", source="mic"), 2,
        )
        self.assertEqual(partial["QWEN_NEW_AUDIO_MS"], 100.0)
        self.assertEqual(partial["EPOCH_AUDIO_ACCUMULATED_MS"], 100.0)
        self.assertIsNone(partial["QWEN_AUDIO_ACCUM_MS"])

        final = await asyncio.wait_for(
            self.runtime.finish(connection_id="cadence", source="mic"), 2,
        )
        self.assertIsNone(final["QWEN_NEW_AUDIO_MS"])
        self.assertEqual(final["EPOCH_AUDIO_ACCUMULATED_MS"], 100.0)
        self.assertIsNone(final["QWEN_AUDIO_ACCUM_MS"])

    async def test_explicit_capacity_wait_preserves_only_the_opted_in_session(self):
        class BlockingWorker(FakeWorker):
            def __init__(self):
                super().__init__()
                self.push_started = asyncio.Event()
                self.release_push = asyncio.Event()

            async def request(self, operation, **payload):
                if operation == "push":
                    self.push_started.set()
                    await self.release_push.wait()
                return await super().request(operation, **payload)

        worker = BlockingWorker()
        worker.ready = True
        self.runtime.worker = worker
        started = await self.runtime.open_session(
            connection_id="handoff", source="mic", model_chunk_ms=100,
        )
        stream_id = started["stream_id"]
        frame = b"\x41\x00" * 1600
        try:
            await self.runtime.push_audio(connection_id="handoff", source="mic", pcm16le=frame)
            await asyncio.wait_for(worker.push_started.wait(), 1)
            for _ in range(4):
                await self.runtime.push_audio(connection_id="handoff", source="mic", pcm16le=frame)

            with self.assertRaisesRegex(StreamingError, "stream_scheduler_capacity_wait"):
                await self.runtime.push_audio(
                    connection_id="handoff", source="mic", pcm16le=frame,
                    allow_capacity_wait=True,
                )

            self.assertIn(("handoff", "mic"), self.runtime.sessions)
            self.assertIn(("handoff", stream_id), self.runtime.scheduler.streams)
            self.assertFalse(self.runtime.scheduler.streams[("handoff", stream_id)].fenced)
            self.assertFalse(any(
                operation == "close" and payload.get("stream_id") == stream_id
                for operation, payload in worker.requests
            ))

            # ACTIVE/default admission remains terminal for the same full queue.
            with self.assertRaisesRegex(StreamingError, "stream_scheduler_overrun"):
                await self.runtime.push_audio(
                    connection_id="handoff", source="mic", pcm16le=frame,
                )
            self.assertNotIn(("handoff", "mic"), self.runtime.sessions)
            self.assertNotIn(("handoff", stream_id), self.runtime.scheduler.streams)
            self.assertTrue(any(
                operation == "close" and payload.get("stream_id") == stream_id
                for operation, payload in worker.requests
            ))
        finally:
            worker.release_push.set()

    async def test_empty_stream_eos_finalizes_without_pcm_push_or_partial(self):
        await self.runtime.open_session(
            connection_id="empty", source="mic", context="empty stream", model_chunk_ms=250,
        )

        final = await self.runtime.finish(connection_id="empty", source="mic")

        self.assertEqual(final["event"], "final_candidate")
        self.assertEqual(final["PARTIAL_COUNT"], 0)
        self.assertNotIn("turnId", final)
        self.assertNotIn("turn_id", final)
        self.assertEqual(
            [operation for operation, _ in self.worker.requests if operation in {"push", "finish"}],
            ["finish"],
        )

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
        failure = asyncio.get_running_loop().create_future()
        async def sink(payload):
            if payload.get("event") == "error" and not failure.done():
                failure.set_result(payload)
        await self.runtime.open_session(connection_id="conn-a", source="mic", chunk_size_ms=250, event_sink=sink)
        await self.runtime.push_audio(connection_id="conn-a", source="mic", pcm16le=b"\0\0" * 4000)
        self.assertEqual((await asyncio.wait_for(failure, 1))['code'], "stream_worker_failed")
        await asyncio.sleep(0)
        self.assertEqual(self.runtime.active_sessions, 0)

    async def test_runtime_boundary_rejects_multi_step_worker_reply(self):
        from asr_lab.qwen_scheduler import DecodeJob, RequestKey
        started = await self.runtime.open_session(
            connection_id="invalid-delta", source="mic", model_chunk_ms=50,
        )
        async def invalid_reply(_operation, **payload):
            return {
                "scheduler_key": payload["scheduler_key"],
                "decode_steps_delta": 2,
                "decode_wall_ms": 10,
                "text": "must not become a candidate",
            }
        with patch.object(self.worker, "request", side_effect=invalid_reply):
            with self.assertRaisesRegex(StreamingError, "invalid_worker_response"):
                await self.runtime._execute_scheduled(
                    RequestKey("invalid-delta", started["stream_id"], 1),
                    DecodeJob("push", b"\x01\x00" * 800, 800, 0.0),
                )

    async def test_open_session_validates_each_present_chunk_alias_before_comparing(self):
        for canonical, legacy in ((50, 50.0), (None, 50.0)):
            with self.subTest(canonical=canonical, legacy=legacy), self.assertRaisesRegex(StreamingError, "invalid_stream_chunk"):
                await self.runtime.open_session(
                    connection_id=f"alias-{canonical}-{legacy}", source="mic",
                    model_chunk_ms=canonical, chunk_size_ms=legacy,
                )
        with self.assertRaisesRegex(StreamingError, "conflicting_model_chunk"):
            await self.runtime.open_session(
                connection_id="alias-conflict", source="mic", model_chunk_ms=50, chunk_size_ms=100,
            )

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
                    return {
                        "scheduler_key": payload.get("scheduler_key"),
                        "decoded": True,
                        "decode_wall_ms": 3,
                        "decode_steps_delta": 1,
                        "text": "stale result",
                        "language": "Spanish",
                    }
                return await super().request(operation, **payload)

        worker = DelayedWorker()
        results = []
        async def sink(payload):
            results.append(payload)
        runtime = QwenStreamingRuntime(
            worker=worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="fixed",
            gpu_memory_utilization=0.65,
            max_active_sessions=2,
            max_pending_jobs=12,
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
            processed = []
            stale_rejects = []
            await runtime.open_session(
                connection_id="dead-connection", source="mic", chunk_size_ms=1000,
                event_sink=sink,
                processed_audio_observer=lambda stream_id, cursor: processed.append((stream_id, cursor)),
                local_stale_reject_observer=lambda stream_id, gate: stale_rejects.append((stream_id, gate)),
            )
            pending = asyncio.create_task(runtime.push_audio(
                connection_id="dead-connection", source="mic", pcm16le=b"\0\0" * 16000,
            ))
            await worker.push_started.wait()
            await runtime.close_connection("dead-connection")
            worker.release_push.set()
            self.assertIsNone(await pending)
            for _ in range(100):
                if stale_rejects:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(runtime.active_sessions, 0)
            self.assertEqual(results, [])
            self.assertEqual(processed, [])
            self.assertEqual(len(stale_rejects), 1)
            self.assertEqual(stale_rejects[0][1], "scheduler_request_key_stale")
        finally:
            worker.release_push.set()
            await runtime.close()

    async def test_runtime_secondary_freshness_gate_reports_once_without_delivering(self):
        from asr_lab.qwen_scheduler import DecodeJob, RequestKey

        stale_rejects = []
        events = []
        async def sink(payload):
            events.append(payload)

        started = await self.runtime.open_session(
            connection_id="freshness", source="mic", event_sink=sink,
            local_stale_reject_observer=lambda stream_id, gate: stale_rejects.append((stream_id, gate)),
        )
        stream_id = started["stream_id"]
        session = self.runtime.sessions[("freshness", "mic")]
        key = RequestKey("freshness", stream_id, 1)
        session.latest_scheduler_key = RequestKey("freshness", stream_id, 2)
        job = DecodeJob("push", b"\0\0", 1, 0.0, local_stale_reject_observer=lambda sid, gate: stale_rejects.append((sid, gate)))
        with patch.object(self.runtime.scheduler, "is_current", return_value=True):
            await self.runtime._scheduled_result(
                key, job,
                {"text": "must not deliver", "decode_wall_ms": 1.0, "decode_steps_delta": 1},
                {},
            )
        self.assertEqual(stale_rejects, [(stream_id, "runtime_session_key_stale")])
        self.assertEqual(events, [])
        self.assertEqual(session.partial_count, 0)

    async def test_processed_audio_observer_tracks_successful_pushes_including_blank_text_and_eos_residual(self):
        worker = FakeWorker(
            text_sequences={"blank observer": [""]},
            finals={"blank observer": "final only"},
        )
        runtime = QwenStreamingRuntime(
            worker=worker, model_id="Qwen/Qwen3-ASR-1.7B", model_revision="fixed",
            gpu_memory_utilization=0.65, max_active_sessions=1, max_pending_jobs=12,
            max_backlog_chunks=4, max_stream_seconds=2.0, session_idle_ttl_seconds=120.0,
            max_context_chars=512, default_chunk_ms=250, default_language="auto",
            unfixed_chunk_num=2, unfixed_token_num=5,
        )
        await runtime.start()
        try:
            observed = []
            first_push_processed = asyncio.Event()

            def observe(stream_id, cursor):
                observed.append((stream_id, cursor))
                first_push_processed.set()

            opened = await runtime.open_session(
                connection_id="observer", source="mic", context="blank observer",
                chunk_size_ms=250, processed_audio_observer=observe,
            )
            await runtime.push_audio(
                connection_id="observer", source="mic", pcm16le=b"\x01\x00" * 4000,
            )
            await asyncio.wait_for(first_push_processed.wait(), 1)
            self.assertEqual(observed, [(opened["stream_id"], 4000)])

            # Leave a sub-chunk tail so EOS must successfully push it before
            # the finish RPC. The finish operation itself has no PCM progress.
            await runtime.push_audio(
                connection_id="observer", source="mic", pcm16le=b"\x02\x00" * 17,
            )
            final = await runtime.finish(connection_id="observer", source="mic")
            self.assertEqual(final["text"], "final only")
            self.assertEqual(observed, [(opened["stream_id"], 4000), (opened["stream_id"], 4017)])
            self.assertEqual(
                [operation for operation, _ in worker.requests if operation in {"push", "finish"}],
                ["push", "push", "finish"],
            )
        finally:
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

    async def test_six_concurrent_clients_keep_candidate_events_owned(self):
        worker = FakeWorker(text_sequences={f"client-{i}": [f"candidate-{i}"] for i in range(6)})
        runtime = QwenStreamingRuntime(
            worker=worker, model_id="Qwen/Qwen3-ASR-1.7B", model_revision="fixed",
            gpu_memory_utilization=0.65, max_active_sessions=6, max_pending_jobs=24,
            max_backlog_chunks=4, max_stream_seconds=2.0, session_idle_ttl_seconds=120.0,
            max_context_chars=512, default_chunk_ms=250, default_language="auto",
            unfixed_chunk_num=2, unfixed_token_num=5,
        )
        await runtime.start()
        try:
            events = {f"client-{i}": [] for i in range(6)}
            stream_ids = {}
            async def make_sink(client):
                async def sink(payload):
                    events[client].append(payload)
                return sink
            for i in range(6):
                client = f"client-{i}"
                started = await runtime.open_session(
                    connection_id=client, source="mic", context=client,
                    chunk_size_ms=250, event_sink=await make_sink(client),
                )
                stream_ids[client] = started["stream_id"]
            await asyncio.gather(*(
                runtime.push_audio(connection_id=f"client-{i}", source="mic", pcm16le=b"\0\0" * 4000)
                for i in range(6)
            ))
            async def wait_for_event(client):
                for _ in range(100):
                    if events[client]:
                        return
                    await asyncio.sleep(0.01)
                self.fail(f"candidate missing for {client}")
            await asyncio.gather(*(wait_for_event(f"client-{i}") for i in range(6)))
            for i in range(6):
                client = f"client-{i}"
                self.assertEqual([event["text"] for event in events[client]], [f"candidate-{i}"])
                self.assertEqual(events[client][0]["stream_id"], stream_ids[client])
                self.assertNotIn("connection_id", events[client][0])
            self.assertEqual(len(set(stream_ids.values())), 6)
            self.assertEqual(runtime.scheduler.snapshot()["active_stream_count"], 6)
        finally:
            await runtime.close()

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


    async def test_init_rpc_latency_excludes_scheduler_registration(self):
        from types import SimpleNamespace
        import asr_lab.qwen_streaming as streaming_module
        clock_values = iter((0.0, 0.0, 0.0, 0.007))
        with patch.object(streaming_module, "time", SimpleNamespace(perf_counter=lambda: next(clock_values))):
            started = await self.runtime.open_session(connection_id="init-clock", source="mic", model_chunk_ms=50)
        self.assertEqual(started["FIRST_STREAM_INIT_MS"], 7.0)
        self.assertEqual(started["FIRST_STREAM_STATE_INIT_WALL_MS"], 2.0)
        self.assertEqual(started["FIRST_STREAM_INIT_RPC_OVERHEAD_MS"], 5.0)

    async def test_first_decode_clocks_use_server_admission_ready_and_dispatch(self):
        from types import SimpleNamespace
        import asr_lab.qwen_scheduler as scheduler_module
        import asr_lab.qwen_streaming as streaming_module
        await self.runtime.open_session(connection_id="clocked", source="mic", model_chunk_ms=50, context="client A")
        stream_clock = [0.010, 0.012]
        def server_clock():
            return stream_clock.pop(0) if stream_clock else 0.016
        with (
            patch.object(streaming_module, "time", SimpleNamespace(perf_counter=server_clock)),
            patch.object(scheduler_module, "time", SimpleNamespace(perf_counter=lambda: 0.015)),
        ):
            await self.runtime.push_audio(connection_id="clocked", source="mic", pcm16le=b"\x01\x00" * 400)
            await self.runtime.push_audio(connection_id="clocked", source="mic", pcm16le=b"\x02\x00" * 400)
            partial = await self.runtime.receive_event(connection_id="clocked", source="mic")
        self.assertEqual(partial["FIRST_AUDIO_TO_FIRST_DECODE_READY_MS"], 0.0)
        self.assertEqual(partial["FIRST_SCHEDULER_WAIT_MS"], 5.0)
        self.assertEqual(partial["FIRST_AUDIO_TO_FIRST_DECODE_START_MS"], 5.0)
        self.assertEqual(partial["FIRST_AUDIO_TO_FIRST_DECODE_READY_MS"] + partial["FIRST_SCHEDULER_WAIT_MS"], partial["FIRST_AUDIO_TO_FIRST_DECODE_START_MS"])
        self.assertEqual(partial["SERVER_FIRST_PARTIAL_MS"], partial["FIRST_PARTIAL_MS"])

    async def test_real_decode_ordinals_exclude_zero_step_and_include_eos_finish(self):
        class SequenceWorker(FakeWorker):
            def __init__(self):
                super().__init__()
                self.push_index = 0

            async def request(self, operation, **payload):
                self.requests.append((operation, payload.copy()))
                stream_id = payload.get("stream_id")
                if operation == "init":
                    self.streams[stream_id] = {"context": "ordinal", "index": 0, "text": ""}
                    return {"stream_id": stream_id, "stream_state_init_wall_ms": 2.0}
                if operation == "close":
                    return {"closed": True}
                if operation == "push":
                    cases = [(1, 10.0, "one"), (0, 999.0, "one"), (1, 20.0, "two"), (0, 999.0, "two")]
                    steps, wall, text = cases[self.push_index]
                    self.push_index += 1
                elif operation == "finish":
                    steps, wall, text = (1, 30.0, "two")
                else:
                    raise AssertionError(operation)
                response = {"decode_steps_delta": steps, "decode_wall_ms": wall, "text": text, "language": "es"}
                if payload.get("scheduler_key") is not None:
                    response["scheduler_key"] = payload["scheduler_key"]
                return response

        worker = SequenceWorker()
        runtime = QwenStreamingRuntime(
            worker=worker, model_id="qwen", model_revision="fixed", gpu_memory_utilization=0.65,
            max_active_sessions=2, max_stream_seconds=2, session_idle_ttl_seconds=120,
            max_context_chars=512, default_chunk_ms=50, default_language="auto",
            unfixed_chunk_num=2, unfixed_token_num=5,
        )
        await runtime.start()
        try:
            await runtime.open_session(connection_id="ordinal", source="mic", model_chunk_ms=50)
            frame = b"\x03\x00" * 800
            await runtime.push_audio(connection_id="ordinal", source="mic", pcm16le=frame)
            first = await runtime.receive_event(connection_id="ordinal", source="mic")
            await runtime.push_audio(connection_id="ordinal", source="mic", pcm16le=frame)
            await runtime.push_audio(connection_id="ordinal", source="mic", pcm16le=frame)
            second = await runtime.receive_event(connection_id="ordinal", source="mic")
            await runtime.push_audio(connection_id="ordinal", source="mic", pcm16le=b"\x04\x00" * 400)
            final = await runtime.finish(connection_id="ordinal", source="mic")
            self.assertEqual(first["EPOCH_FIRST_DECODE_WALL_MS"], 10.0)
            self.assertEqual(second["EPOCH_SECOND_DECODE_WALL_MS"], 20.0)
            self.assertEqual(final["EPOCH_FIRST_DECODE_WALL_MS"], 10.0)
            self.assertEqual(final["EPOCH_SECOND_DECODE_WALL_MS"], 20.0)
            self.assertEqual(final["EPOCH_STEADY_DECODE_WALL_P50_MS"], 30.0)
            self.assertEqual(final["EPOCH_STEADY_DECODE_WALL_P95_MS"], 30.0)
            self.assertEqual(final["QWEN_CUMULATIVE_DECODE_WALL_MS"], 60.0)
            self.assertEqual(final["QWEN_DECODE_WALL_SAMPLE_COUNT"], 3)
            self.assertEqual(final["QWEN_DECODE_STEPS_DELTA_TOTAL"], 3)
            self.assertIsNotNone(final["FIRST_PARTIAL_MS"])
            self.assertEqual(final["SERVER_FIRST_PARTIAL_MS"], final["FIRST_PARTIAL_MS"])
        finally:
            await runtime.close()


    def test_decode_ordinal_metric_nullability_for_zero_one_and_two_real_steps(self):
        from asr_lab.qwen_streaming import StreamSession, _decode_ordinal_metrics
        session = StreamSession("c", "mic", "s", None, "auto", 100, "", 0, 0, 60)
        expected = (None, None, None, None)
        self.assertEqual(tuple(_decode_ordinal_metrics(session).values()), expected * 2)
        session.decode_wall_samples[:] = [10.0]
        one = _decode_ordinal_metrics(session)
        self.assertEqual(one["FIRST_DECODE_WALL_MS"], 10.0)
        self.assertIsNone(one["SECOND_DECODE_WALL_MS"])
        self.assertIsNone(one["STEADY_DECODE_WALL_P50_MS"])
        session.decode_wall_samples[:] = [10.0, 20.0]
        two = _decode_ordinal_metrics(session)
        self.assertEqual(two["FIRST_DECODE_WALL_MS"], 10.0)
        self.assertEqual(two["SECOND_DECODE_WALL_MS"], 20.0)
        self.assertIsNone(two["STEADY_DECODE_WALL_P95_MS"])

    async def test_optional_candidate_filter_enriches_only_after_local_partial_gate(self):
        filtered = asyncio.Event()
        delivered = []
        filter_calls = []

        def candidate_filter(event):
            filter_calls.append(event["event"])
            result = dict(event)
            result.update({
                "asr_job_id": "trusted-job",
                "speech_segment_id": "trusted-segment",
                "epoch_id": "trusted-epoch",
                "epoch_seq": 9,
            })
            filtered.set()
            return result

        async def sink(event):
            delivered.append(dict(event))

        await self.runtime.open_session(
            connection_id="filtered", source="mic", context="client A",
            chunk_size_ms=250, event_sink=sink,
            candidate_result_filter=candidate_filter,
        )
        await self.runtime.push_audio(
            connection_id="filtered", source="mic", pcm16le=b"\x01\x00" * 4000,
        )
        await asyncio.wait_for(filtered.wait(), 1)
        self.assertEqual(filter_calls, ["partial_candidate"])
        self.assertEqual(len(delivered), 1)
        candidate = delivered[0]
        self.assertEqual(candidate["truth_status"], "candidate_only")
        self.assertEqual(candidate["asr_job_id"], "trusted-job")
        self.assertEqual(candidate["speech_segment_id"], "trusted-segment")
        self.assertEqual(candidate["epoch_id"], "trusted-epoch")
        self.assertEqual(candidate["epoch_seq"], 9)
        self.assertIn("stream_id", candidate)
        self.assertIn("QWEN_SCHEDULER_REVISION", candidate)

    async def test_candidate_filter_rejection_suppresses_partial_and_final_and_closes(self):
        filter_calls = []
        filtered_partial = asyncio.Event()
        delivered = []

        def reject_candidate(event):
            filter_calls.append(event["event"])
            if event["event"] == "partial_candidate":
                filtered_partial.set()
            return None

        async def sink(event):
            delivered.append(dict(event))

        await self.runtime.open_session(
            connection_id="rejected", source="mic", context="client A",
            chunk_size_ms=250, event_sink=sink,
            candidate_result_filter=reject_candidate,
        )
        await self.runtime.push_audio(
            connection_id="rejected", source="mic", pcm16le=b"\x02\x00" * 4000,
        )
        await asyncio.wait_for(filtered_partial.wait(), 1)
        self.assertEqual(delivered, [])

        final = await self.runtime.finish(connection_id="rejected", source="mic")
        self.assertIsNone(final)
        self.assertEqual(filter_calls, ["partial_candidate", "final_candidate"])
        self.assertEqual(delivered, [])
        self.assertEqual(self.runtime.active_sessions, 0)


if __name__ == "__main__":
    unittest.main()
