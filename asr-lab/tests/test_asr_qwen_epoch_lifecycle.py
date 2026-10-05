from __future__ import annotations

import asyncio
import base64
import unittest

from asr_lab.asr_epoch_controller import LocalAcousticEpochController
from asr_lab.asr_epoch_controller import EpochControllerState
from asr_lab.asr_fencing import ExecutionFence, FenceOutcome
from asr_lab.asr_pcm_handoff import LocalPCMHandoffCoordinator
from asr_lab.asr_epoch_rollover_policy import (
    SignalClass,
    SoftRolloverPolicy,
    SoftSignalDefinition,
)
from asr_lab.asr_qwen_epoch_lifecycle import (
    QwenEpochLifecycle,
    QwenEpochLifecycleState,
)
from asr_lab.qwen_streaming import QwenStreamingRuntime, StreamingError


class SequenceAllocator:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class LifecycleWorker:
    """RPC fake preserving operation order and allowing old decode fencing."""

    def __init__(self) -> None:
        self.ready = False
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.stream_ids: list[str] = []
        self.second_init = asyncio.Event()
        self.release_second_init = asyncio.Event()
        self.old_push_started = asyncio.Event()
        self.release_old_push = asyncio.Event()
        self.successor_push_started = asyncio.Event()
        self.release_successor_push = asyncio.Event()
        self.fail_second_init = False

    async def start(self, **_options):
        self.ready = True
        return {"model_load_ms": 1.0}

    async def request(self, operation: str, **payload):
        self.requests.append((operation, dict(payload)))
        stream_id = payload.get("stream_id")
        if operation == "init":
            if self.fail_second_init and self.stream_ids:
                raise RuntimeError("successor init failed")
            assert isinstance(stream_id, str)
            self.stream_ids.append(stream_id)
            if len(self.stream_ids) >= 2:
                self.second_init.set()
                await self.release_second_init.wait()
            return {"stream_id": stream_id, "stream_state_init_wall_ms": 1.0}
        if operation == "push":
            assert isinstance(stream_id, str)
            if stream_id == self.stream_ids[0]:
                self.old_push_started.set()
                await self.release_old_push.wait()
            else:
                self.successor_push_started.set()
                await self.release_successor_push.wait()
            return {
                "scheduler_key": payload.get("scheduler_key"),
                "decoded": True,
                "decode_wall_ms": 5.0,
                "decode_steps_delta": 1,
                "text": "",
                "language": "English",
            }
        if operation == "finish":
            return {
                "scheduler_key": payload.get("scheduler_key"),
                "decode_wall_ms": 1.0,
                "decode_steps_delta": 1,
                "text": "final candidate",
                "language": "English",
            }
        if operation == "close":
            return {"closed": True}
        raise AssertionError(operation)

    async def close(self):
        self.ready = False
        self.release_old_push.set()
        self.release_successor_push.set()


class CandidateLifecycleWorker(LifecycleWorker):
    def __init__(self) -> None:
        super().__init__()
        self.block_finish = False
        self.finish_started = asyncio.Event()
        self.release_finish = asyncio.Event()
        self.completed_push_streams: list[str] = []
        self.push_counts: dict[str, int] = {}
        self.push_texts_by_stream_ordinal: list[list[str]] = []
        self.blocked_push: tuple[int, int] | None = None
        self.blocked_push_started = asyncio.Event()
        self.release_blocked_push = asyncio.Event()

    async def request(self, operation: str, **payload):
        if operation == "finish" and self.block_finish:
            self.finish_started.set()
            await self.release_finish.wait()
        stream_id = payload.get("stream_id")
        push_ordinal = None
        push_number = None
        if operation == "push" and isinstance(stream_id, str):
            push_ordinal = self.stream_ids.index(stream_id)
            push_number = self.push_counts.get(stream_id, 0) + 1
            self.push_counts[stream_id] = push_number
            if self.blocked_push == (push_ordinal, push_number):
                self.blocked_push_started.set()
                await self.release_blocked_push.wait()
        response = await super().request(operation, **payload)
        if operation == "push":
            response["text"] = "partial candidate"
            if push_ordinal is not None and push_number is not None:
                if push_ordinal < len(self.push_texts_by_stream_ordinal):
                    texts = self.push_texts_by_stream_ordinal[push_ordinal]
                    if push_number <= len(texts):
                        response["text"] = texts[push_number - 1]
            if isinstance(stream_id, str):
                self.completed_push_streams.append(stream_id)
        return response


class QwenEpochLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def make_lifecycle(
        self,
        worker: LifecycleWorker | None = None,
        *,
        replay_overlap_samples: int = 0,
        event_sink=None,
        max_stream_seconds: float = 60.0,
        soft_rollover_policy: SoftRolloverPolicy | None = None,
        monotonic=None,
    ):
        worker = worker or LifecycleWorker()
        controller = LocalAcousticEpochController(
            asr_job_id="job-stable",
            speech_segment_id="segment-stable",
            initial_epoch_seq=17,
            epoch_id_allocator=SequenceAllocator(
                "canonical-epoch-17",
                "canonical-epoch-18",
                "canonical-epoch-19",
                "canonical-epoch-20",
            ),
        )
        handoff = LocalPCMHandoffCoordinator(
            epoch_controller=controller,
            replay_overlap_samples=replay_overlap_samples,
            max_retained_samples=100_000,
            max_transition_samples=20_000,
        )
        runtime = QwenStreamingRuntime(
            worker=worker, model_id="Qwen/Qwen3-ASR-1.7B", model_revision="fixed",
            gpu_memory_utilization=0.65, max_active_sessions=1, max_pending_jobs=12,
            max_backlog_chunks=4, max_stream_seconds=max_stream_seconds, session_idle_ttl_seconds=120.0,
            max_context_chars=512, default_chunk_ms=250, default_language="auto",
            unfixed_chunk_num=2, unfixed_token_num=5,
        )
        await runtime.start()
        lifecycle = QwenEpochLifecycle(
            epoch_controller=controller,
            pcm_handoff=handoff,
            qwen_runtime=runtime,
            connection_id="connection-stable",
            source="mic",
            language="en",
            context="stable session context",
            chunk_size_ms=250,
            request_id="request-stable",
            event_sink=event_sink,
            soft_rollover_policy=soft_rollover_policy,
            **({"monotonic": monotonic} if monotonic is not None else {}),
        )
        return worker, controller, handoff, runtime, lifecycle

    async def wait_until(self, predicate, message: str) -> None:
        for _ in range(1000):
            if predicate():
                return
            await asyncio.sleep(0.001)
        self.fail(message)

    async def test_two_fresh_epochs_preserve_pcm_order_and_finish_only_successor(self):
        worker, controller, handoff, runtime, lifecycle = await self.make_lifecycle()
        first_pcm = b"\x11\x00" * 4000
        transition_pcm = b"\x22\x00" * 4000
        live_pcm = b"\x33\x00" * 4000
        try:
            initial = await lifecycle.initialize()
            old_fence = lifecycle.current_fence
            old_stream = lifecycle.local_stream_id
            self.assertEqual(old_fence.epoch_id, "canonical-epoch-17")
            self.assertNotEqual(old_stream, old_fence.epoch_id)

            ack = await lifecycle.submit_pcm(first_pcm)
            self.assertEqual(ack.accepted_samples, 4000)
            await asyncio.wait_for(worker.old_push_started.wait(), 1)
            before_rollover = handoff.snapshot()
            self.assertEqual(before_rollover.unique_primary_admitted_samples, 4000)
            self.assertEqual(before_rollover.processed_cursor, 0)

            rollover_task = asyncio.create_task(lifecycle.rollover())
            await asyncio.wait_for(worker.second_init.wait(), 1)
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.HANDOFF_PENDING)
            self.assertEqual(controller.state, EpochControllerState.SUCCESSOR_PREPARED)
            self.assertEqual(controller.current, old_fence)
            self.assertIsNone(await lifecycle.submit_pcm(transition_pcm))
            self.assertEqual(handoff.snapshot().transition_queued_samples, 4000)
            worker.release_second_init.set()

            # Let the old, now-fenced RPC complete. Its result must not advance
            # the successor's processing cursor.
            worker.release_old_push.set()
            await asyncio.wait_for(worker.successor_push_started.wait(), 1)
            self.assertEqual(lifecycle.snapshot().observed_local_processed_cursor, 0)
            self.assertEqual(lifecycle.current_fence.epoch_id, "canonical-epoch-18")
            self.assertNotEqual(lifecycle.local_stream_id, old_stream)
            self.assertNotEqual(lifecycle.local_stream_id, lifecycle.current_fence.epoch_id)

            drained = await asyncio.wait_for(rollover_task, 1)
            self.assertTrue(drained.completed)
            worker.release_successor_push.set()
            await self.wait_until(
                lambda: lifecycle.snapshot().observed_local_processed_cursor == 8000,
                "successor replay and transition were not processed",
            )
            lifecycle._sync_processed()
            after_handoff = handoff.snapshot()
            self.assertEqual(after_handoff.processed_cursor, 8000)
            self.assertEqual(after_handoff.current_epoch_admitted_cursor, 8000)

            live_ack = await lifecycle.submit_pcm(live_pcm)
            self.assertEqual(live_ack.accepted_samples, 4000)
            await self.wait_until(
                lambda: lifecycle.snapshot().observed_local_processed_cursor == 12000,
                "successor live PCM was not processed",
            )
            lifecycle._sync_processed()

            final = await lifecycle.finish()
            self.assertEqual(final["event"], "final_candidate")
            self.assertEqual(final["text"], "final candidate")
            self.assertEqual(final["stream_id"], worker.stream_ids[1])
            self.assertEqual(final["asr_job_id"], controller.current.asr_job_id)
            self.assertEqual(final["speech_segment_id"], controller.current.speech_segment_id)
            self.assertEqual(final["epoch_id"], controller.current.epoch_id)
            self.assertEqual(final["epoch_seq"], controller.current.epoch_seq)
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.CLOSED)

            successor = controller.current
            self.assertEqual(successor.asr_job_id, "job-stable")
            self.assertEqual(successor.speech_segment_id, "segment-stable")
            self.assertEqual(successor.epoch_seq, old_fence.epoch_seq + 1)
            snapshot = handoff.snapshot()
            self.assertEqual(snapshot.received_samples, 12000)
            self.assertEqual(snapshot.unique_primary_admitted_samples, 12000)
            self.assertEqual(snapshot.replay_admitted_samples, 4000)
            self.assertEqual(snapshot.transition_queued_samples, 0)
            self.assertEqual(snapshot.processed_cursor, 12000)

            pushes = [payload for operation, payload in worker.requests if operation == "push"]
            self.assertEqual([payload["stream_id"] for payload in pushes], [
                worker.stream_ids[0], worker.stream_ids[1], worker.stream_ids[1], worker.stream_ids[1],
            ])
            self.assertEqual(
                [base64.b64decode(payload["pcm16le_base64"]) for payload in pushes],
                [first_pcm, first_pcm, transition_pcm, live_pcm],
            )
            operations = [operation for operation, _ in worker.requests]
            self.assertEqual(operations.count("init"), 2)
            self.assertEqual(operations.count("finish"), 1)
            finish_payload = next(payload for operation, payload in worker.requests if operation == "finish")
            self.assertEqual(finish_payload["stream_id"], worker.stream_ids[1])
            old_close_index = next(
                index for index, (operation, payload) in enumerate(worker.requests)
                if operation == "close" and payload.get("stream_id") == worker.stream_ids[0]
            )
            new_init_index = next(
                index for index, (operation, payload) in enumerate(worker.requests)
                if operation == "init" and payload.get("stream_id") == worker.stream_ids[1]
            )
            self.assertLess(old_close_index, new_init_index)
            self.assertEqual(initial["stream_id"], worker.stream_ids[0])
        finally:
            worker.release_old_push.set()
            worker.release_successor_push.set()
            await runtime.close()

    async def test_successor_init_failure_stays_fail_closed_without_resuming_old_state(self):
        worker = LifecycleWorker()
        worker.fail_second_init = True
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker, controller, handoff, runtime, lifecycle = await self.make_lifecycle(worker)
        try:
            await lifecycle.initialize()
            old_fence = controller.current
            await lifecycle.submit_pcm(b"\x01\x00" * 4000)
            await self.wait_until(
                lambda: lifecycle.snapshot().observed_local_processed_cursor == 4000,
                "initial fake decode did not finish",
            )
            with self.assertRaisesRegex(RuntimeError, "successor init failed"):
                await lifecycle.rollover()
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.HANDOFF_PENDING)
            self.assertEqual(controller.current, old_fence)
            self.assertEqual(runtime.active_sessions, 0)
            self.assertTrue(handoff.snapshot().handoff_in_progress)
            self.assertNotIn("finish", [operation for operation, _ in worker.requests])
            # Handoff-pending source audio remains A6.2-owned as TRANSITION;
            # no old Qwen state is reopened implicitly.
            self.assertIsNone(await lifecycle.submit_pcm(b"\x02\x00"))
            self.assertEqual(handoff.snapshot().transition_queued_samples, 1)
        finally:
            await runtime.close()

    async def test_processed_overlap_is_replayed_without_becoming_unique_primary_again(self):
        worker = LifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        worker, controller, handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            replay_overlap_samples=800,
        )
        original_pcm = b"\x42\x00" * 4000
        try:
            await lifecycle.initialize()
            await lifecycle.submit_pcm(original_pcm)
            await self.wait_until(
                lambda: lifecycle.snapshot().observed_local_processed_cursor == 4000,
                "initial epoch PCM was not processed before overlap rollover",
            )
            lifecycle._sync_processed()
            self.assertEqual(handoff.snapshot().processed_cursor, 4000)

            drained = await lifecycle.rollover()
            self.assertTrue(drained.completed)
            replay_snapshot = handoff.snapshot()
            self.assertEqual(replay_snapshot.replay_start_cursor, 3200)
            self.assertEqual(replay_snapshot.unique_primary_admitted_samples, 4000)
            self.assertEqual(replay_snapshot.replay_admitted_samples, 800)

            final = await lifecycle.finish()
            self.assertEqual(final["stream_id"], worker.stream_ids[1])
            snapshot = handoff.snapshot()
            self.assertEqual(snapshot.received_samples, 4000)
            self.assertEqual(snapshot.unique_primary_admitted_samples, 4000)
            self.assertEqual(snapshot.replay_admitted_samples, 800)
            self.assertEqual(snapshot.processed_cursor, 4000)
            replay_pushes = [
                base64.b64decode(payload["pcm16le_base64"])
                for operation, payload in worker.requests
                if operation == "push" and payload.get("stream_id") == worker.stream_ids[1]
            ]
            self.assertEqual(replay_pushes, [original_pcm[3200 * 2:]])
            self.assertEqual(controller.current.epoch_seq, 18)
        finally:
            await runtime.close()

    async def test_current_partial_and_final_candidates_keep_exact_epoch_provenance(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            fence = controller.current
            await lifecycle.submit_pcm(b"\x10\x00" * 4000)
            await self.wait_until(
                lambda: len(delivered) == 1,
                "current partial candidate was not delivered",
            )
            partial = delivered[0]
            self.assertEqual(partial["event"], "partial_candidate")
            self.assertEqual(partial["truth_status"], "candidate_only")
            self.assertEqual(partial["asr_job_id"], fence.asr_job_id)
            self.assertEqual(partial["speech_segment_id"], fence.speech_segment_id)
            self.assertEqual(partial["epoch_id"], fence.epoch_id)
            self.assertEqual(partial["epoch_seq"], fence.epoch_seq)
            self.assertNotIn("turnId", partial)

            final = await lifecycle.finish()
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final["event"], "final_candidate")
            self.assertEqual(final["truth_status"], "candidate_only")
            self.assertEqual(final["asr_job_id"], fence.asr_job_id)
            self.assertEqual(final["speech_segment_id"], fence.speech_segment_id)
            self.assertEqual(final["epoch_id"], fence.epoch_id)
            self.assertEqual(final["epoch_seq"], fence.epoch_seq)
            self.assertNotIn("turnId", final)
            self.assertEqual(len(delivered), 1)
        finally:
            await runtime.close()

    async def test_lifecycle_candidate_binding_suppresses_lower_and_colliding_revisions(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        worker, _controller, _handoff, runtime, lifecycle = await self.make_lifecycle(worker)
        try:
            await lifecycle.initialize()
            session = runtime.sessions[("connection-stable", "mic")]
            result_filter = session.candidate_result_filter
            assert result_filter is not None

            def event(text: str, revision: int) -> dict[str, object]:
                return {
                    "event": "partial_candidate",
                    "text": text,
                    "revision": revision,
                    "final": False,
                    "provisional": True,
                    "candidate_only": True,
                    "truth_status": "candidate_only",
                    "replace": True,
                }

            invalid = event("must not become the base", 9)
            invalid["truth_status"] = "truth"
            self.assertIsNone(result_filter(invalid))
            self.assertIsNone(lifecycle.candidate_stitcher_snapshot.highest_revision)

            fresh = result_filter(event("fresh words", 4))
            self.assertIsNotNone(fresh)
            assert fresh is not None
            self.assertEqual(fresh["text"], "fresh words")

            self.assertIsNone(result_filter(event("older words", 3)))
            self.assertEqual(
                lifecycle.candidate_stitcher_snapshot.last_rejection_reason,
                "obsolete_revision",
            )
            self.assertIsNone(result_filter(event("collision", 4)))
            snapshot = lifecycle.candidate_stitcher_snapshot
            self.assertEqual(snapshot.last_rejection_reason, "revision_text_collision")
            self.assertEqual(snapshot.latest_text, "fresh words")
            self.assertEqual(snapshot.highest_revision, 4)
        finally:
            await runtime.close()

    async def test_canonically_stale_partial_is_rejected_while_local_request_is_current(self):
        worker = CandidateLifecycleWorker()
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            old_fence = controller.current
            session = runtime.sessions[("connection-stable", "mic")]
            filter_calls = []
            original_filter = session.candidate_result_filter
            assert original_filter is not None

            def observe_filter(event):
                filter_calls.append(event["event"])
                return original_filter(event)

            session.candidate_result_filter = observe_filter
            await lifecycle.submit_pcm(b"\x11\x00" * 4000)
            await asyncio.wait_for(worker.old_push_started.wait(), 1)
            controller.request_rollover()
            controller.prepare_successor()
            successor = controller.activate_successor()
            self.assertEqual(controller.classify(old_fence), FenceOutcome.STALE)
            worker.release_old_push.set()
            await self.wait_until(
                lambda: bool(session.scheduler_metrics),
                "old partial result did not reach its local result gate",
            )
            self.assertEqual(session.partial_count, 1)
            self.assertEqual(filter_calls, ["partial_candidate"])
            self.assertEqual(successor.epoch_seq, old_fence.epoch_seq + 1)
            self.assertEqual(delivered, [])
            self.assertIsNone(lifecycle.candidate_stitcher_snapshot.latest_text)
            self.assertIsNone(lifecycle.candidate_stitcher_snapshot.highest_revision)
        finally:
            worker.release_old_push.set()
            await runtime.close()

    async def test_canonically_stale_final_is_not_relabelled_as_successor(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        worker.block_finish = True
        worker.release_finish = asyncio.Event()
        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(worker)
        try:
            await lifecycle.initialize()
            old_fence = controller.current
            session = runtime.sessions[("connection-stable", "mic")]
            filter_calls = []
            original_filter = session.candidate_result_filter
            assert original_filter is not None

            def observe_filter(event):
                filter_calls.append(event["event"])
                return original_filter(event)

            session.candidate_result_filter = observe_filter
            finish_task = asyncio.create_task(lifecycle.finish())
            await asyncio.wait_for(worker.finish_started.wait(), 1)
            controller.request_rollover()
            controller.prepare_successor()
            successor = controller.activate_successor()
            self.assertEqual(controller.classify(old_fence), FenceOutcome.STALE)
            worker.release_finish.set()
            final = await asyncio.wait_for(finish_task, 1)
            self.assertIsNone(final)
            self.assertEqual(filter_calls, ["final_candidate"])
            self.assertEqual(successor.epoch_seq, old_fence.epoch_seq + 1)
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.CLOSED)
        finally:
            worker.release_finish.set()
            await runtime.close()

    async def test_local_stream_fence_cannot_be_rescued_by_current_canonical_fence(self):
        worker = CandidateLifecycleWorker()
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            current_fence = controller.current
            session = runtime.sessions[("connection-stable", "mic")]
            filter_calls = []
            original_filter = session.candidate_result_filter
            assert original_filter is not None

            def observe_filter(event):
                filter_calls.append(event["event"])
                return original_filter(event)

            session.candidate_result_filter = observe_filter
            await lifecycle.submit_pcm(b"\x12\x00" * 4000)
            await asyncio.wait_for(worker.old_push_started.wait(), 1)
            self.assertEqual(controller.classify(current_fence), FenceOutcome.ACCEPT)
            await runtime.close_session(connection_id="connection-stable", source="mic")
            worker.release_old_push.set()
            await asyncio.sleep(0.02)
            self.assertEqual(filter_calls, [])
            self.assertEqual(delivered, [])
        finally:
            worker.release_old_push.set()
            await runtime.close()

    def test_candidate_binding_fails_closed_for_invalid_scopes_and_stale_sequence(self):
        from asr_lab.asr_qwen_epoch_lifecycle import _QwenCandidateFenceBinding

        accepted_payload = {
            "event": "partial_candidate",
            "candidate_only": True,
            "truth_status": "candidate_only",
            "provisional": True,
            "final": False,
            "text": "untrusted candidate text",
            "revision": 1,
            "asr_job_id": "payload-job",
            "speech_segment_id": "payload-segment",
            "epoch_id": "payload-epoch",
            "epoch_seq": -1,
        }
        current = ExecutionFence("job-stable", "epoch-17", 17, "segment-stable")
        accepted_controller = LocalAcousticEpochController(
            asr_job_id="job-stable",
            speech_segment_id="segment-stable",
            initial_epoch_seq=17,
            epoch_id_allocator=SequenceAllocator("fixture-epoch"),
        )
        accepted_controller._current = current
        accepted_binding = _QwenCandidateFenceBinding(accepted_controller, current)
        accepted = accepted_binding.filter_candidate(accepted_payload)
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertEqual(accepted["asr_job_id"], current.asr_job_id)
        self.assertEqual(accepted["speech_segment_id"], current.speech_segment_id)
        self.assertEqual(accepted["epoch_id"], current.epoch_id)
        self.assertEqual(accepted["epoch_seq"], current.epoch_seq)
        self.assertIsNone(_QwenCandidateFenceBinding(accepted_controller).filter_candidate(accepted_payload))

        cases = (
            (
                "wrong-job",
                ExecutionFence("other-job", "epoch-17", 17, "segment-stable"),
                current,
                FenceOutcome.INVALID_SCOPE,
            ),
            (
                "missing-segment",
                ExecutionFence("job-stable", "epoch-17", 17, None),
                current,
                FenceOutcome.INVALID_SCOPE,
            ),
            (
                "different-segment",
                ExecutionFence("job-stable", "epoch-17", 17, "other-segment"),
                current,
                FenceOutcome.INVALID_SCOPE,
            ),
            (
                "higher-unadmitted-sequence",
                ExecutionFence("job-stable", "epoch-18", 18, "segment-stable"),
                current,
                FenceOutcome.INVALID_SCOPE,
            ),
            (
                "same-sequence-different-epoch",
                ExecutionFence("job-stable", "other-epoch", 17, "segment-stable"),
                current,
                FenceOutcome.INVALID_SCOPE,
            ),
            (
                "lower-sequence",
                ExecutionFence("job-stable", "epoch-16", 16, "segment-stable"),
                current,
                FenceOutcome.STALE,
            ),
        )

        for label, producer, authoritative_current, expected in cases:
            with self.subTest(scope=label):
                controller = LocalAcousticEpochController(
                    asr_job_id="job-stable",
                    speech_segment_id="segment-stable",
                    initial_epoch_seq=17,
                    epoch_id_allocator=SequenceAllocator("fixture-epoch"),
                )
                # This test fixture models an authoritative current fence that
                # the public same-lineage successor API cannot create (for
                # example, a wrong job or speech segment).
                controller._current = authoritative_current
                binding = _QwenCandidateFenceBinding(controller, producer)
                self.assertIsNone(binding.filter_candidate(accepted_payload))
                self.assertIs(expected, controller.classify(producer))

    async def test_locally_stale_final_is_not_rescued_by_current_canonical_fence(self):
        worker = CandidateLifecycleWorker()
        worker.block_finish = True
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            current_fence = controller.current
            session = runtime.sessions[("connection-stable", "mic")]
            filter_calls = []
            original_filter = session.candidate_result_filter
            assert original_filter is not None

            def observe_filter(event):
                filter_calls.append(event["event"])
                return original_filter(event)

            session.candidate_result_filter = observe_filter
            finish_task = asyncio.create_task(
                runtime.finish(connection_id="connection-stable", source="mic")
            )
            await asyncio.wait_for(worker.finish_started.wait(), 1)
            self.assertIs(controller.classify(current_fence), FenceOutcome.ACCEPT)
            await runtime.close_session(connection_id="connection-stable", source="mic")
            worker.release_finish.set()
            with self.assertRaises(StreamingError):
                await asyncio.wait_for(finish_task, 1)
            self.assertEqual(filter_calls, [])
            self.assertIs(controller.classify(current_fence), FenceOutcome.ACCEPT)
            self.assertEqual(delivered, [])
        finally:
            worker.release_finish.set()
            await runtime.close()

    async def test_rollover_rejects_late_predecessor_and_accepts_successor_provenance(self):
        worker = CandidateLifecycleWorker()
        worker.release_successor_push = asyncio.Event()
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            old_fence = controller.current
            old_stream_id = lifecycle.local_stream_id
            await lifecycle.submit_pcm(b"\x13\x00" * 4000)
            await asyncio.wait_for(worker.old_push_started.wait(), 1)

            rollover_task = asyncio.create_task(lifecycle.rollover())
            await asyncio.wait_for(worker.second_init.wait(), 1)
            worker.release_second_init.set()
            await self.wait_until(
                lambda: controller.current.epoch_seq == old_fence.epoch_seq + 1,
                "successor was not canonically activated",
            )
            successor_fence = controller.current
            successor_stream_id = lifecycle.local_stream_id
            self.assertEqual(successor_fence.epoch_seq, old_fence.epoch_seq + 1)
            self.assertNotEqual(successor_stream_id, old_stream_id)

            # The predecessor result completes after successor admission. Its
            # closed local stream cannot send or inherit successor provenance.
            worker.release_old_push.set()
            await self.wait_until(
                lambda: old_stream_id in worker.completed_push_streams,
                "late predecessor RPC did not complete",
            )
            await asyncio.wait_for(worker.successor_push_started.wait(), 1)
            self.assertEqual(delivered, [])

            worker.release_successor_push.set()
            drained = await asyncio.wait_for(rollover_task, 1)
            self.assertTrue(drained.completed)
            await self.wait_until(
                lambda: len(delivered) == 1,
                "successor candidate was not delivered after replay",
            )
            candidate = delivered[0]
            self.assertEqual(candidate["stream_id"], successor_stream_id)
            self.assertEqual(candidate["asr_job_id"], successor_fence.asr_job_id)
            self.assertEqual(candidate["speech_segment_id"], successor_fence.speech_segment_id)
            self.assertEqual(candidate["epoch_id"], successor_fence.epoch_id)
            self.assertEqual(candidate["epoch_seq"], successor_fence.epoch_seq)

            final = await lifecycle.finish()
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final["stream_id"], successor_stream_id)
            self.assertEqual(final["epoch_id"], successor_fence.epoch_id)
            self.assertEqual(final["epoch_seq"], successor_fence.epoch_seq)
            self.assertEqual(len(delivered), 1)
        finally:
            worker.release_old_push.set()
            worker.release_successor_push.set()
            worker.release_second_init.set()
            await runtime.close()

    async def test_candidate_text_stitches_and_base_advances_across_three_epochs(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        worker.push_texts_by_stream_ordinal = [
            ["we want to book"],
            ["want to book a room"],
            ["a room tomorrow"],
        ]
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            first_fence = controller.current
            await lifecycle.submit_pcm(b"\x21\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 1, "epoch 17 candidate missing")
            self.assertEqual(delivered[0]["text"], "we want to book")
            self.assertEqual(delivered[0]["stitch_mode"], "current_only")

            first_handoff = await lifecycle.rollover()
            self.assertTrue(first_handoff.completed)
            second_fence = controller.current
            second_snapshot = lifecycle.candidate_stitcher_snapshot
            self.assertEqual(second_snapshot.base.text, "we want to book")
            self.assertEqual(second_snapshot.base.fence, first_fence)

            await lifecycle.submit_pcm(b"\x22\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 2, "epoch 18 candidate missing")
            self.assertEqual(delivered[1]["text"], "we want to book a room")
            self.assertEqual(delivered[1]["stitch_raw_current_text"], "want to book a room")
            self.assertEqual(delivered[1]["stitch_overlap_token_count"], 3)
            self.assertEqual(delivered[1]["stitch_base_epoch_seq"], first_fence.epoch_seq)
            self.assertEqual(delivered[1]["epoch_id"], second_fence.epoch_id)
            self.assertEqual(delivered[1]["epoch_seq"], second_fence.epoch_seq)
            self.assertEqual(delivered[1]["asr_job_id"], second_fence.asr_job_id)
            self.assertEqual(delivered[1]["speech_segment_id"], second_fence.speech_segment_id)
            self.assertNotIn("turnId", delivered[1])

            second_handoff = await lifecycle.rollover()
            self.assertTrue(second_handoff.completed)
            third_fence = controller.current
            third_snapshot = lifecycle.candidate_stitcher_snapshot
            self.assertEqual(third_fence.epoch_seq, first_fence.epoch_seq + 2)
            self.assertEqual(third_snapshot.base.text, "we want to book a room")
            self.assertEqual(third_snapshot.base.fence, second_fence)

            await lifecycle.submit_pcm(b"\x23\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 3, "epoch 19 candidate missing")
            self.assertEqual(delivered[2]["text"], "we want to book a room tomorrow")
            self.assertEqual(delivered[2]["stitch_raw_current_text"], "a room tomorrow")
            self.assertEqual(delivered[2]["stitch_overlap_token_count"], 2)
            self.assertEqual(delivered[2]["stitch_base_epoch_seq"], second_fence.epoch_seq)
            self.assertEqual(delivered[2]["epoch_seq"], third_fence.epoch_seq)

            final = await lifecycle.finish()
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final["event"], "final_candidate")
            self.assertTrue(final["final"])
            self.assertEqual(final["epoch_seq"], third_fence.epoch_seq)
            self.assertEqual(final["stitch_base_epoch_seq"], second_fence.epoch_seq)
            self.assertEqual(final["truth_status"], "candidate_only")
        finally:
            worker.release_old_push.set()
            worker.release_successor_push.set()
            worker.release_second_init.set()
            await runtime.close()

    async def test_epoch_without_candidate_breaks_stitch_base_chain(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        worker.push_texts_by_stream_ordinal = [
            ["text from epoch one"],
            [""],
            ["new text from epoch three"],
        ]
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            first_fence = controller.current
            await lifecycle.submit_pcm(b"\x24\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 1, "epoch 17 candidate missing")
            self.assertEqual(delivered[0]["text"], "text from epoch one")

            first_handoff = await lifecycle.rollover()
            self.assertTrue(first_handoff.completed)
            second_fence = controller.current
            self.assertEqual(second_fence.epoch_seq, first_fence.epoch_seq + 1)
            self.assertEqual(lifecycle.candidate_stitcher_snapshot.base.fence, first_fence)

            # Epoch 18 emits no candidate. Its lack of accepted logical output
            # must break the base chain before epoch 19 is activated.
            second_handoff = await lifecycle.rollover()
            self.assertTrue(second_handoff.completed)
            third_fence = controller.current
            self.assertEqual(third_fence.epoch_seq, second_fence.epoch_seq + 1)
            self.assertIsNone(lifecycle.candidate_stitcher_snapshot.base)

            await lifecycle.submit_pcm(b"\x25\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 2, "epoch 19 candidate missing")
            third_candidate = delivered[1]
            self.assertEqual(third_candidate["text"], "new text from epoch three")
            self.assertEqual(third_candidate["stitch_raw_current_text"], "new text from epoch three")
            self.assertEqual(third_candidate["stitch_mode"], "current_only")
            self.assertEqual(third_candidate["stitch_reason"], "no_predecessor")
            self.assertIsNone(third_candidate["stitch_base_epoch_id"])
            self.assertIsNone(third_candidate["stitch_base_epoch_seq"])
            self.assertNotIn("text from epoch one", third_candidate["text"])
        finally:
            worker.release_old_push.set()
            worker.release_successor_push.set()
            worker.release_second_init.set()
            await runtime.close()

    async def test_candidate_base_freezes_after_old_stream_close_and_late_result_is_fenced(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        worker.blocked_push = (0, 2)
        worker.push_texts_by_stream_ordinal = [
            ["we should do this", "late predecessor replacement"],
            ["should do this now"],
        ]
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            old_fence = controller.current
            old_session = runtime.sessions[("connection-stable", "mic")]
            old_stream_id = lifecycle.local_stream_id
            await lifecycle.submit_pcm(b"\x31\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 1, "first predecessor candidate missing")
            self.assertEqual(delivered[0]["text"], "we should do this")

            await lifecycle.submit_pcm(b"\x32\x00" * 4000)
            await asyncio.wait_for(worker.blocked_push_started.wait(), 1)
            filter_calls = []
            original_filter = old_session.candidate_result_filter
            assert original_filter is not None

            def observe_filter(event):
                filter_calls.append(event["event"])
                return original_filter(event)

            old_session.candidate_result_filter = observe_filter
            rollover_task = asyncio.create_task(lifecycle.rollover())
            await asyncio.wait_for(worker.second_init.wait(), 1)
            snapshot = lifecycle.candidate_stitcher_snapshot
            self.assertEqual(snapshot.base.text, "we should do this")
            self.assertEqual(snapshot.base.fence, old_fence)
            self.assertEqual(controller.current.epoch_seq, old_fence.epoch_seq + 1)

            worker.release_blocked_push.set()
            await asyncio.wait_for(worker.successor_push_started.wait(), 1)
            drained = await asyncio.wait_for(rollover_task, 1)
            self.assertTrue(drained.completed)
            await self.wait_until(lambda: len(delivered) == 2, "successor candidate missing")
            await self.wait_until(
                lambda: worker.completed_push_streams.count(old_stream_id) >= 2,
                "late predecessor worker call did not finish",
            )
            self.assertEqual(delivered[1]["text"], "we should do this now")
            self.assertEqual(delivered[1]["stitch_base_epoch_seq"], old_fence.epoch_seq)
            self.assertEqual(delivered[1]["epoch_seq"], old_fence.epoch_seq + 1)
            self.assertEqual(delivered[1]["stitch_raw_current_text"], "should do this now")
            # The late decode is rejected by the local RequestKey fence before
            # the canonical/A6.5 callback, so it cannot mutate stitcher state.
            self.assertEqual(filter_calls, [])
            self.assertEqual(delivered[0]["text"], "we should do this")
            self.assertEqual(lifecycle.candidate_stitcher_snapshot.latest_text, "we should do this now")
        finally:
            worker.release_blocked_push.set()
            worker.release_old_push.set()
            worker.release_successor_push.set()
            worker.release_second_init.set()
            await runtime.close()

    async def test_shared_runtime_stale_reject_is_owned_by_its_lifecycle_during_peer_handoff(self):
        class SharedRuntimeWorker:
            def __init__(self):
                self.ready = False
                self.stream_ids = []
                self.blocked_stream_id = None
                self.blocked_push_started = asyncio.Event()
                self.release_blocked_push = asyncio.Event()
                self.third_init_started = asyncio.Event()
                self.release_third_init = asyncio.Event()

            async def start(self, **_options):
                self.ready = True
                return {"model_load_ms": 1.0}

            async def request(self, operation, **payload):
                stream_id = payload.get("stream_id")
                if operation == "init":
                    self.stream_ids.append(stream_id)
                    if len(self.stream_ids) == 3:
                        self.third_init_started.set()
                        await self.release_third_init.wait()
                    return {"stream_id": stream_id, "stream_state_init_wall_ms": 1.0}
                if operation == "push":
                    if stream_id == self.blocked_stream_id:
                        self.blocked_push_started.set()
                        await self.release_blocked_push.wait()
                    return {
                        "scheduler_key": payload.get("scheduler_key"),
                        "decode_wall_ms": 5.0,
                        "decode_steps_delta": 1,
                        "text": "late A candidate",
                        "language": "English",
                    }
                if operation == "close":
                    return {"closed": True}
                if operation == "finish":
                    return {
                        "scheduler_key": payload.get("scheduler_key"),
                        "decode_wall_ms": 1.0,
                        "decode_steps_delta": 1,
                        "text": "final candidate",
                        "language": "English",
                    }
                raise AssertionError(operation)

            async def close(self):
                self.ready = False
                self.release_blocked_push.set()
                self.release_third_init.set()

        worker = SharedRuntimeWorker()
        runtime = QwenStreamingRuntime(
            worker=worker, model_id="Qwen/Qwen3-ASR-1.7B", model_revision="fixed",
            gpu_memory_utilization=0.65, max_active_sessions=2, max_pending_jobs=12,
            max_backlog_chunks=4, max_stream_seconds=60.0, session_idle_ttl_seconds=120.0,
            max_context_chars=512, default_chunk_ms=250, default_language="auto",
            unfixed_chunk_num=2, unfixed_token_num=5,
        )
        await runtime.start()

        def make_peer(prefix):
            controller = LocalAcousticEpochController(
                asr_job_id=f"job-{prefix}", speech_segment_id=f"segment-{prefix}",
                initial_epoch_seq=1,
                epoch_id_allocator=SequenceAllocator(f"{prefix}-epoch-1", f"{prefix}-epoch-2"),
            )
            handoff = LocalPCMHandoffCoordinator(
                epoch_controller=controller, replay_overlap_samples=0,
                max_retained_samples=100_000, max_transition_samples=20_000,
            )
            lifecycle = QwenEpochLifecycle(
                epoch_controller=controller, pcm_handoff=handoff,
                qwen_runtime=runtime, connection_id=f"connection-{prefix}", source="mic",
                language="en", context=f"context {prefix}", chunk_size_ms=250,
            )
            return controller, handoff, lifecycle

        _controller_a, _handoff_a, lifecycle_a = make_peer("A")
        _controller_b, _handoff_b, lifecycle_b = make_peer("B")
        delivered_a = []
        async def collect_a(event):
            delivered_a.append(dict(event))
        lifecycle_a.event_sink = collect_a
        try:
            await lifecycle_a.initialize()
            await lifecycle_b.initialize()
            worker.blocked_stream_id = lifecycle_a.local_stream_id
            await lifecycle_a.submit_pcm(b"\x31\x00" * 4000)
            await asyncio.wait_for(worker.blocked_push_started.wait(), 1)

            # Simulate A's worker-side transport/session disappearing while its
            # in-flight decode still holds its captured local owner observer.
            await runtime.close_session(connection_id="connection-A", source="mic")
            peer_rollover = asyncio.create_task(lifecycle_b.rollover())
            await asyncio.wait_for(worker.third_init_started.wait(), 1)
            self.assertEqual(lifecycle_b.state, QwenEpochLifecycleState.HANDOFF_PENDING)

            worker.release_blocked_push.set()
            await self.wait_until(
                lambda: lifecycle_a.observability_snapshot()["logical_cumulative"]["stale_result_rejects"] == 1,
                "A's late decode was not counted by its owner lifecycle",
            )
            a_snapshot = lifecycle_a.observability_snapshot()
            b_snapshot = lifecycle_b.observability_snapshot()
            self.assertEqual(a_snapshot["logical_cumulative"]["local_request_freshness_rejects"], 1)
            self.assertEqual(a_snapshot["logical_cumulative"]["canonical_fence_rejects"], 0)
            self.assertEqual(a_snapshot["logical_cumulative"]["stale_result_rejects"], 1)
            self.assertEqual(a_snapshot["current_epoch"]["stale_result_rejects"], 1)
            self.assertEqual(lifecycle_a._observed_local_processed_cursor, 0)
            self.assertEqual(delivered_a, [])
            self.assertEqual(b_snapshot["logical_cumulative"]["stale_result_rejects"], 0)
            self.assertEqual(b_snapshot["last_transition"]["stale_result_rejects_start"], 0)
            self.assertEqual(b_snapshot["last_transition"]["stale_result_rejects_delta"], 0)

            worker.release_third_init.set()
            result = await asyncio.wait_for(peer_rollover, 1)
            self.assertTrue(result.completed)
            final_b = lifecycle_b.observability_snapshot()
            self.assertEqual(final_b["last_transition"]["stale_result_rejects_delta"], 0)
        finally:
            worker.release_blocked_push.set()
            worker.release_third_init.set()
            await runtime.close()

    async def test_hard_bound_holds_triggering_pcm_until_successor_and_resets_epoch_scope(self):
        worker = LifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        soft_policy = SoftRolloverPolicy(
            soft_enabled=True,
            window_samples=2,
            required_qualifying_samples=2,
            min_active_signals=2,
            cooldown_ms=500.0,
            signals=(
                SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 1, 0),
                SoftSignalDefinition("scheduler_backlog_ms", SignalClass.LOCAL_THROUGHPUT, 1, 0),
            ),
        )
        worker, controller, handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            max_stream_seconds=0.25,
            soft_rollover_policy=soft_policy,
        )
        # Simultaneous soft evidence must lose to the hard per-stream bound.
        lifecycle._soft_trigger_metrics = lambda: {
            "epoch_audio_ms": 10,
            "scheduler_backlog_ms": 10,
        }
        try:
            await lifecycle.initialize()
            await lifecycle.submit_pcm(b"\x11\x00" * 4000)  # 250 ms, exactly at hard bound
            await self.wait_until(
                lambda: lifecycle.snapshot().observed_local_processed_cursor >= 4000,
                "predecessor audio did not finish before hard bound test",
            )
            result = await lifecycle.submit_pcm(b"\x22\x00" * 1000)
            self.assertIsNone(result)
            self.assertEqual(controller.current.epoch_seq, 18)
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.ACTIVE)

            snapshot = lifecycle.observability_snapshot()
            transition = snapshot["last_transition"]
            self.assertEqual(transition["category"], "hard_bound")
            self.assertEqual(transition["reason"], "hard_bound")
            self.assertEqual(
                transition["handoff_started_pcm_accounting"]["unique_primary_admitted_samples"],
                4000,
            )
            self.assertEqual(
                transition["completed_pcm_accounting"]["unique_primary_admitted_samples"],
                5000,
            )
            self.assertEqual(transition["transition_primary_samples"], 1000)
            self.assertEqual(transition["replay_admitted_samples"], 0)
            self.assertEqual(snapshot["current_epoch"]["epoch_seq"], 18)
            self.assertEqual(snapshot["current_epoch"]["epoch_audio_ms"], 62.5)
            self.assertEqual(snapshot["logical_cumulative"]["source_received_samples"], 5000)
            self.assertEqual(snapshot["logical_cumulative"]["unique_primary_samples"], 5000)
            self.assertEqual(handoff.snapshot().transition_queued_samples, 0)
        finally:
            await runtime.close()

    async def test_soft_rollover_requires_configured_persistent_multisignal_evidence(self):
        worker = LifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        soft_policy = SoftRolloverPolicy(
            soft_enabled=True,
            window_samples=3,
            required_qualifying_samples=2,
            min_active_signals=2,
            cooldown_ms=1000.0,
            signals=(
                SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 1, 0),
                SoftSignalDefinition("scheduler_backlog_ms", SignalClass.LOCAL_THROUGHPUT, 1, 0),
            ),
        )
        worker, controller, handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            soft_rollover_policy=soft_policy,
        )
        policy_samples = iter((
            {"epoch_audio_ms": 0, "scheduler_backlog_ms": 0},
            {"epoch_audio_ms": 10, "scheduler_backlog_ms": 10},
            {"epoch_audio_ms": 10, "scheduler_backlog_ms": 10},
        ))
        lifecycle._soft_trigger_metrics = lambda: next(policy_samples)
        try:
            await lifecycle.initialize()
            await lifecycle.submit_pcm(b"\x31\x00" * 400)
            self.assertEqual(controller.current.epoch_seq, 17)
            await lifecycle.submit_pcm(b"\x32\x00" * 400)
            self.assertEqual(controller.current.epoch_seq, 17)
            self.assertEqual(lifecycle.observability_snapshot()["soft_rollover_policy"]["qualifying_samples"], 1)

            result = await lifecycle.submit_pcm(b"\x33\x00" * 400)
            self.assertIsNone(result)
            self.assertEqual(controller.current.epoch_seq, 18)
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.ACTIVE)
            snapshot = lifecycle.observability_snapshot()
            self.assertEqual(snapshot["last_transition"]["category"], "proactive_soft")
            self.assertEqual(snapshot["last_transition"]["transition_primary_samples"], 400)
            self.assertEqual(snapshot["logical_cumulative"]["source_received_samples"], 1200)
            self.assertEqual(snapshot["logical_cumulative"]["unique_primary_samples"], 1200)
            self.assertGreater(snapshot["soft_rollover_policy"]["cooldown_remaining_ms"], 0)
            self.assertEqual(handoff.snapshot().transition_queued_samples, 0)
        finally:
            await runtime.close()

    async def test_scheduler_overrun_becomes_one_emergency_transition_without_queue_growth(self):
        worker = LifecycleWorker()
        worker.release_successor_push = asyncio.Event()
        worker, controller, handoff, runtime, lifecycle = await self.make_lifecycle(worker)
        frame = b"\x41\x00" * 4000
        try:
            await lifecycle.initialize()
            await lifecycle.submit_pcm(frame)
            await asyncio.wait_for(worker.old_push_started.wait(), 1)
            for _ in range(4):
                await lifecycle.submit_pcm(frame)
            triggering_task = asyncio.create_task(lifecycle.submit_pcm(frame))
            await asyncio.wait_for(worker.second_init.wait(), 1)
            # Unblock the fenced old RPC; no old admission is retried.
            worker.release_old_push.set()
            worker.release_second_init.set()
            result = await asyncio.wait_for(triggering_task, 1)
            self.assertIsNone(result)

            snapshot = lifecycle.observability_snapshot()
            transition = snapshot["last_transition"]
            self.assertEqual(transition["category"], "emergency_backpressure")
            self.assertEqual(transition["reason"], "stream_scheduler_overrun")
            self.assertEqual(controller.current.epoch_seq, 18)
            self.assertEqual(runtime.scheduler.max_backlog_chunks, 4)
            self.assertEqual(
                transition["handoff_started_pcm_accounting"]["unique_primary_admitted_samples"],
                20000,
            )
            self.assertEqual(transition["transition_primary_samples"], 0)
            # Catch-up remains bounded and fail-closed: the rejected frame is
            # still owned by A6.2 if the fresh scheduler cannot admit it yet.
            self.assertEqual(handoff.snapshot().transition_queued_samples, 4000)
            self.assertEqual(handoff.snapshot().received_samples, 24000)
            self.assertEqual(handoff.snapshot().unique_primary_admitted_samples, 20000)
            self.assertEqual(lifecycle.state, QwenEpochLifecycleState.HANDOFF_PENDING)
            self.assertEqual(len(worker.stream_ids), 2)
        finally:
            worker.release_old_push.set()
            worker.release_successor_push.set()
            worker.release_second_init.set()
            await runtime.close()

    async def test_first_successor_partial_is_recorded_only_after_transition_primary(self):
        worker = CandidateLifecycleWorker()
        worker.release_old_push.set()
        worker.release_successor_push.set()
        worker.release_second_init.set()
        delivered = []

        async def sink(event):
            delivered.append(dict(event))

        worker, controller, _handoff, runtime, lifecycle = await self.make_lifecycle(
            worker,
            max_stream_seconds=0.25,
            event_sink=sink,
        )
        try:
            await lifecycle.initialize()
            await lifecycle.submit_pcm(b"\x51\x00" * 4000)
            await self.wait_until(lambda: len(delivered) == 1, "predecessor partial missing")
            await lifecycle.submit_pcm(b"\x52\x00" * 1000)
            self.assertEqual(controller.current.epoch_seq, 18)
            await self.wait_until(lambda: len(delivered) == 2, "successor partial missing")

            transition = lifecycle.observability_snapshot()["last_transition"]
            self.assertIsNotNone(transition["EPOCH_HANDOFF_WALL_MS"])
            self.assertIsNotNone(transition["EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS"])
            self.assertGreaterEqual(transition["EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS"], 0)
            self.assertEqual(delivered[1]["epoch_seq"], 18)
            self.assertEqual(transition["transition_primary_samples"], 1000)
        finally:
            worker.release_old_push.set()
            worker.release_successor_push.set()
            worker.release_second_init.set()
            await runtime.close()


if __name__ == "__main__":
    unittest.main()
