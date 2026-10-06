from __future__ import annotations

import asyncio
import unittest

from asr_lab.asr_epoch_controller import EpochControllerState, LocalAcousticEpochController
from asr_lab.asr_fencing import ExecutionFence
from asr_lab.asr_pcm_handoff import (
    AdmissionAck,
    AdmissionRejected,
    PCMAdmissionKind,
    PCMCapacityRejected,
    PCMHandoffError,
    LocalPCMHandoffCoordinator,
)
from asr_lab.qwen_scheduler import QwenDecodeScheduler, SchedulerError


class SequenceAllocator:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class RecordingSink:
    def __init__(self, *outcomes: object) -> None:
        self.calls: list[tuple[PCMAdmissionKind, ExecutionFence, object]] = []
        self.outcomes = list(outcomes)

    async def submit(self, kind, fence, span):
        self.calls.append((kind, fence, span))
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, AdmissionAck):
            return outcome
        if outcome is not None:
            return outcome
        return AdmissionAck(span.sample_count)


class GatedSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, kind, fence, span):
        self.calls.append((kind, fence, span))
        self.entered.set()
        await self.release.wait()
        return AdmissionAck(span.sample_count)


class SchedulerSink:
    def __init__(self, scheduler: QwenDecodeScheduler, connection_id: str, stream_id: str) -> None:
        self.scheduler = scheduler
        self.connection_id = connection_id
        self.stream_id = stream_id

    async def submit(self, kind, fence, span):
        if kind is not PCMAdmissionKind.PRIMARY:
            raise AssertionError("the existing scheduler is used only for old PRIMARY admissions")
        try:
            result = await self.scheduler.append_pcm(
                self.connection_id, self.stream_id, span.pcm16le,
            )
        except SchedulerError as exc:
            raise AdmissionRejected(exc.code) from exc
        return AdmissionAck(result["accepted_samples"])


class LocalPCMHandoffCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def make_coordinator(
        self,
        *epoch_ids: str,
        overlap: int = 0,
        retained: int = 1000,
        transition: int = 1000,
    ):
        controller = LocalAcousticEpochController(
            asr_job_id="job-a",
            speech_segment_id="segment-a",
            initial_epoch_seq=41,
            epoch_id_allocator=SequenceAllocator(*epoch_ids),
        )
        coordinator = LocalPCMHandoffCoordinator(
            epoch_controller=controller,
            replay_overlap_samples=overlap,
            max_retained_samples=retained,
            max_transition_samples=transition,
        )
        return coordinator, controller

    async def admit(self, coordinator, sink, sample_count: int) -> object:
        span = coordinator.receive_pcm(b"\x01\x00" * sample_count)
        await coordinator.admit_primary(span, sink)
        return span

    async def test_pcm_is_owned_before_pending_sink_ack(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        sink = GatedSink()
        span = coordinator.receive_pcm(b"\x01\x00" * 3)
        task = asyncio.create_task(coordinator.admit_primary(span, sink))
        await asyncio.wait_for(sink.entered.wait(), 1)

        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.retained_source_samples, 3)
        self.assertEqual(snapshot.transition_queued_samples, 0)
        self.assertEqual(snapshot.unique_primary_admitted_samples, 0)
        self.assertEqual(sink.calls[0][0], PCMAdmissionKind.PRIMARY)

        sink.release.set()
        await task
        self.assertEqual(coordinator.snapshot().transition_queued_samples, 0)

    async def test_active_unadmitted_pcm_uses_retained_not_transition_bound(self):
        coordinator, _ = self.make_coordinator("epoch-a", retained=6, transition=1)
        span = coordinator.receive_pcm(b"\x01\x00" * 4)
        snapshot = coordinator.snapshot()
        self.assertEqual(span.sample_count, 4)
        self.assertEqual(snapshot.retained_source_samples, 4)
        self.assertEqual(snapshot.transition_queued_samples, 0)

        await coordinator.admit_primary(span, RecordingSink())
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.unique_primary_admitted_samples, 4)
        self.assertEqual(snapshot.transition_queued_samples, 0)

    async def test_submission_attempt_alone_does_not_create_primary(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        sink = GatedSink()
        span = coordinator.receive_pcm(b"\x01\x00")
        task = asyncio.create_task(coordinator.admit_primary(span, sink))
        await sink.entered.wait()
        self.assertEqual(coordinator.snapshot().retained_source_samples, 1)
        self.assertEqual(coordinator.snapshot().transition_queued_samples, 0)
        self.assertEqual(coordinator.snapshot().unique_primary_admitted_cursor, 0)
        self.assertEqual(coordinator.snapshot().unique_primary_admitted_samples, 0)
        sink.release.set()
        await task

    async def test_exact_ack_marks_unique_primary_admission(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        sink = RecordingSink()
        span = coordinator.receive_pcm(b"\x01\x00" * 4)
        ack = await coordinator.admit_primary(span, sink)
        self.assertEqual(ack, AdmissionAck(4))
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.unique_primary_admitted_cursor, 4)
        self.assertEqual(snapshot.unique_primary_admitted_samples, 4)
        self.assertEqual(snapshot.transition_queued_samples, 0)

    async def test_known_failed_admission_retains_pcm_as_transition(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        span = coordinator.receive_pcm(b"\x01\x00" * 3)
        with self.assertRaises(PCMHandoffError) as raised:
            await coordinator.admit_primary(span, RecordingSink(AdmissionRejected("overrun")))
        self.assertEqual(raised.exception.code, "admission_rejected")
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.retained_source_samples, 3)
        self.assertEqual(snapshot.transition_queued_samples, 3)
        self.assertEqual(snapshot.unique_primary_admitted_samples, 0)
        self.assertFalse(coordinator.primary_admission_open)

    async def test_admission_ack_does_not_advance_processing_watermark(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        await self.admit(coordinator, RecordingSink(), 5)
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.processed_cursor, 0)
        self.assertEqual(snapshot.retained_source_samples, 5)

    async def test_source_span_cannot_be_marked_primary_twice(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        sink = RecordingSink()
        span = coordinator.receive_pcm(b"\x01\x00" * 2)
        await coordinator.admit_primary(span, sink)
        with self.assertRaises(PCMHandoffError) as raised:
            await coordinator.admit_primary(span, sink)
        self.assertEqual(raised.exception.code, "primary_span_not_pending")
        self.assertEqual(len(sink.calls), 1)

    async def test_processing_must_be_contiguous_and_within_admitted_coverage(self):
        coordinator, controller = self.make_coordinator("epoch-a")
        await self.admit(coordinator, RecordingSink(), 5)
        with self.assertRaises(PCMHandoffError) as raised:
            coordinator.mark_processed(fence=controller.current, start_sample=1, end_sample=2)
        self.assertEqual(raised.exception.code, "non_contiguous_processing_completion")
        with self.assertRaises(PCMHandoffError) as raised:
            coordinator.mark_processed(fence=controller.current, start_sample=0, end_sample=6)
        self.assertEqual(raised.exception.code, "processing_beyond_admitted_audio")

    async def test_processing_watermark_releases_only_processed_data_outside_overlap(self):
        coordinator, controller = self.make_coordinator("epoch-a", overlap=2)
        await self.admit(coordinator, RecordingSink(), 6)
        coordinator.mark_processed(fence=controller.current, start_sample=0, end_sample=4)
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.processed_cursor, 4)
        self.assertEqual(snapshot.released_source_samples, 2)
        self.assertEqual(snapshot.retained_source_samples, 4)
        self.assertEqual(snapshot.retained_range_floor, 2)

    async def test_handoff_uses_partial_source_span_for_bounded_replay_overlap(self):
        coordinator, controller = self.make_coordinator("epoch-a", "epoch-b", overlap=2)
        await self.admit(coordinator, RecordingSink(), 6)
        coordinator.mark_processed(fence=controller.current, start_sample=0, end_sample=4)
        coordinator.begin_handoff()
        old_fence = controller.current
        successor = coordinator.activate_successor()
        self.assertEqual(successor.epoch_seq, old_fence.epoch_seq + 1)
        sink = RecordingSink()
        result = await coordinator.drain_handoff(sink)
        self.assertTrue(result.completed)
        self.assertEqual([(call[2].start_sample, call[2].end_sample) for call in sink.calls], [(2, 6)])
        self.assertEqual(sink.calls[0][2].pcm16le, b"\x01\x00" * 4)
        self.assertEqual(result.replay_admitted_samples, 4)

    async def test_replay_does_not_increment_unique_primary_accounting(self):
        coordinator, controller = self.make_coordinator("epoch-a", "epoch-b")
        await self.admit(coordinator, RecordingSink(), 4)
        before = coordinator.snapshot().unique_primary_admitted_samples
        coordinator.begin_handoff()
        coordinator.activate_successor()
        await coordinator.drain_handoff(RecordingSink())
        after = coordinator.snapshot()
        self.assertEqual(after.unique_primary_admitted_samples, before)
        self.assertEqual(after.replay_admitted_samples, 4)

    async def test_replay_precedes_transition_primary_in_source_order(self):
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b")
        await self.admit(coordinator, RecordingSink(), 3)
        coordinator.begin_handoff()
        coordinator.activate_successor()
        first = coordinator.receive_pcm(b"\x02\x00" * 2)
        second = coordinator.receive_pcm(b"\x03\x00")
        sink = RecordingSink()
        result = await coordinator.drain_handoff(sink)
        self.assertTrue(result.completed)
        self.assertEqual([call[0] for call in sink.calls], [
            PCMAdmissionKind.REPLAY,
            PCMAdmissionKind.PRIMARY,
            PCMAdmissionKind.PRIMARY,
        ])
        self.assertEqual([(call[2].start_sample, call[2].end_sample) for call in sink.calls], [
            (0, 3), (3, 5), (5, 6),
        ])
        self.assertEqual(first.start_sample, 3)
        self.assertEqual(second.start_sample, 5)

    async def test_known_successor_rejection_retains_transition_and_stops_later_ranges(self):
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b")
        await self.admit(coordinator, RecordingSink(), 2)
        coordinator.begin_handoff()
        coordinator.activate_successor()
        coordinator.receive_pcm(b"\x02\x00")
        coordinator.receive_pcm(b"\x03\x00")
        sink = RecordingSink(None, AdmissionRejected("capacity"))
        result = await coordinator.drain_handoff(sink)
        self.assertFalse(result.completed)
        self.assertEqual(result.blocked_reason, "capacity")
        self.assertEqual(coordinator.snapshot().transition_queued_samples, 2)
        self.assertEqual(len(sink.calls), 2)  # one REPLAY, then the first transition only

        retry_sink = RecordingSink()
        retried = await coordinator.drain_handoff(retry_sink)
        self.assertTrue(retried.completed)
        self.assertEqual([c[2].start_sample for c in retry_sink.calls], [2, 3])

    async def test_ambiguous_admission_is_retained_and_never_blindly_retried(self):
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b")
        span = coordinator.receive_pcm(b"\x01\x00" * 2)
        with self.assertRaises(PCMHandoffError) as raised:
            await coordinator.admit_primary(span, RecordingSink(RuntimeError("lost ack")))
        self.assertEqual(raised.exception.code, "admission_unconfirmed")
        self.assertEqual(coordinator.snapshot().transition_queued_samples, 2)
        coordinator.begin_handoff()
        coordinator.activate_successor()
        successor_sink = RecordingSink()
        result = await coordinator.drain_handoff(successor_sink)
        self.assertFalse(result.completed)
        self.assertEqual(result.blocked_reason, "ambiguous_primary_admission")
        self.assertEqual(successor_sink.calls, [])
        self.assertEqual(coordinator.snapshot().retained_source_samples, 2)

    async def test_bad_ack_is_ambiguous_and_fail_closed(self):
        coordinator, _ = self.make_coordinator("epoch-a")
        span = coordinator.receive_pcm(b"\x01\x00" * 2)
        with self.assertRaises(PCMHandoffError) as raised:
            await coordinator.admit_primary(span, RecordingSink(AdmissionAck(1)))
        self.assertEqual(raised.exception.code, "admission_ack_mismatch")
        self.assertEqual(coordinator.snapshot().unique_primary_admitted_samples, 0)
        self.assertEqual(coordinator.snapshot().transition_queued_samples, 2)

    async def test_prepared_successor_is_not_current_until_activation(self):
        coordinator, controller = self.make_coordinator("epoch-a", "epoch-b")
        old = controller.current
        prepared_id = coordinator.begin_handoff()
        self.assertEqual(prepared_id, "epoch-b")
        self.assertEqual(controller.state, EpochControllerState.SUCCESSOR_PREPARED)
        self.assertEqual(controller.current, old)
        self.assertEqual(coordinator.current_fence, old)
        new = coordinator.activate_successor()
        self.assertEqual(controller.current, new)
        self.assertEqual(new.epoch_seq, old.epoch_seq + 1)

    async def test_overrun_frame_is_transition_and_not_old_primary(self):
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b")
        old_sink = RecordingSink()
        await self.admit(coordinator, old_sink, 2)
        trigger = coordinator.receive_pcm(b"\x09\x00")
        with self.assertRaises(PCMHandoffError):
            await coordinator.admit_primary(trigger, RecordingSink(AdmissionRejected("stream_scheduler_overrun")))
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.unique_primary_admitted_samples, 2)
        self.assertEqual(snapshot.transition_queued_samples, 1)
        self.assertEqual(trigger.start_sample, snapshot.unique_primary_admitted_cursor)

    async def test_real_scheduler_overrun_fence_preserves_history_and_trigger(self):
        started = asyncio.Event()
        release = asyncio.Event()
        faults = []

        async def execute(_key, _job):
            started.set()
            await release.wait()
            return {"decode_wall_ms": 1.0, "text": ""}

        async def on_result(*_args):
            return None

        async def on_fault(_connection_id, _stream_id, code, details):
            faults.append((code, details))

        scheduler = QwenDecodeScheduler(
            execute=execute,
            on_result=on_result,
            on_fault=on_fault,
            max_pending_jobs=8,
            max_backlog_chunks=1,
            max_active_streams=1,
        )
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b")
        adapter = SchedulerSink(scheduler, "connection-a", "stream-a")
        await scheduler.start()
        try:
            await scheduler.register("connection-a", "stream-a", 10)
            first = coordinator.receive_pcm(b"\x01\x00" * 160)
            await coordinator.admit_primary(first, adapter)
            await asyncio.wait_for(started.wait(), 1)
            second = coordinator.receive_pcm(b"\x02\x00" * 160)
            await coordinator.admit_primary(second, adapter)

            trigger = coordinator.receive_pcm(b"\x09\x00")
            with self.assertRaises(PCMHandoffError) as raised:
                await coordinator.admit_primary(trigger, adapter)
            self.assertEqual(raised.exception.code, "admission_rejected")
            self.assertEqual(faults[0][0], "stream_scheduler_overrun")
            self.assertEqual(scheduler.streams, {})
            self.assertEqual(scheduler.pending_jobs, 0)
            before_handoff = coordinator.snapshot()
            self.assertEqual(before_handoff.unique_primary_admitted_samples, 320)
            self.assertEqual(before_handoff.transition_queued_samples, 1)
            self.assertEqual(before_handoff.replay_retained_samples, 320)

            coordinator.begin_handoff()
            coordinator.activate_successor()
            successor_sink = RecordingSink()
            drained = await coordinator.drain_handoff(successor_sink)
            self.assertTrue(drained.completed)
            self.assertEqual([item[0] for item in successor_sink.calls], [
                PCMAdmissionKind.REPLAY,
                PCMAdmissionKind.REPLAY,
                PCMAdmissionKind.PRIMARY,
            ])
            replay_ranges = [(item[2].start_sample, item[2].end_sample) for item in successor_sink.calls[:2]]
            self.assertEqual(replay_ranges, [(0, 160), (160, 320)])
            self.assertEqual(successor_sink.calls[2][2], trigger)
            self.assertEqual(coordinator.snapshot().unique_primary_admitted_samples, 321)
            self.assertEqual(coordinator.snapshot().replay_admitted_samples, 320)
            completed_call_count = len(successor_sink.calls)
            with self.assertRaises(PCMHandoffError) as raised:
                await coordinator.drain_handoff(successor_sink)
            self.assertEqual(raised.exception.code, "successor_not_activated")
            self.assertEqual(len(successor_sink.calls), completed_call_count)
        finally:
            release.set()
            await scheduler.close()

    async def test_late_old_epoch_processing_after_activation_is_stale(self):
        coordinator, controller = self.make_coordinator("epoch-a", "epoch-b")
        await self.admit(coordinator, RecordingSink(), 3)
        old = controller.current
        coordinator.begin_handoff()
        coordinator.activate_successor()
        with self.assertRaises(PCMHandoffError) as raised:
            coordinator.mark_processed(fence=old, start_sample=0, end_sample=1)
        self.assertEqual(raised.exception.code, "stale_processing_fence")
        self.assertEqual(coordinator.snapshot().processed_cursor, 0)

    async def test_capacity_overflow_is_explicit_and_accounted_without_sink_call(self):
        coordinator, _ = self.make_coordinator("epoch-a", retained=2, transition=2)
        with self.assertRaises(PCMCapacityRejected) as raised:
            coordinator.receive_pcm(b"\x01\x00" * 3)
        self.assertEqual(raised.exception.code, "retained_capacity_exceeded")
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.received_samples, 3)
        self.assertEqual(snapshot.retained_source_samples, 0)
        self.assertEqual(snapshot.explicit_source_rejected_samples, 3)
        self.assertEqual(
            snapshot.received_samples,
            snapshot.retained_source_samples + snapshot.released_source_samples + snapshot.explicit_source_rejected_samples,
        )

    async def test_transition_capacity_applies_explicit_backpressure(self):
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b", retained=10, transition=2)
        coordinator.begin_handoff()
        coordinator.activate_successor()
        coordinator.receive_pcm(b"\x01\x00" * 2)
        with self.assertRaises(PCMCapacityRejected) as raised:
            coordinator.receive_pcm(b"\x02\x00")
        self.assertEqual(raised.exception.code, "transition_capacity_exceeded")
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.transition_queued_samples, 2)
        self.assertEqual(snapshot.max_transition_queued_samples, 2)
        self.assertEqual(snapshot.explicit_source_rejected_samples, 1)
        self.assertLessEqual(snapshot.retained_source_samples, coordinator.max_retained_samples)
        self.assertGreaterEqual(snapshot.max_retained_source_samples, snapshot.retained_source_samples)

    async def test_mixed_success_and_source_rejection_obeys_exact_accounting_balance(self):
        coordinator, controller = self.make_coordinator("epoch-a", retained=3, transition=3)
        await self.admit(coordinator, RecordingSink(), 3)
        coordinator.mark_processed(fence=controller.current, start_sample=0, end_sample=2)
        with self.assertRaises(PCMCapacityRejected):
            coordinator.receive_pcm(b"\x02\x00" * 3)
        snapshot = coordinator.snapshot()
        self.assertEqual(
            snapshot.received_samples,
            snapshot.retained_source_samples + snapshot.released_source_samples + snapshot.explicit_source_rejected_samples,
        )
        self.assertLessEqual(snapshot.unique_primary_admitted_samples, snapshot.received_samples - snapshot.explicit_source_rejected_samples)

    async def test_successor_replays_old_unprocessed_primary_history(self):
        coordinator, controller = self.make_coordinator("epoch-a", "epoch-b")
        await self.admit(coordinator, RecordingSink(), 4)
        old = controller.current
        coordinator.begin_handoff()
        coordinator.activate_successor()
        sink = RecordingSink()
        result = await coordinator.drain_handoff(sink)
        self.assertTrue(result.completed)
        self.assertEqual(sink.calls[0][0], PCMAdmissionKind.REPLAY)
        self.assertEqual(sink.calls[0][1].epoch_id, "epoch-b")
        self.assertEqual(sink.calls[0][2].pcm16le, b"\x01\x00" * 4)
        self.assertNotEqual(old.epoch_id, sink.calls[0][1].epoch_id)

    async def test_handoff_rejection_retains_trigger_for_retry_without_reordering(self):
        coordinator, _ = self.make_coordinator("epoch-a", "epoch-b")
        await self.admit(coordinator, RecordingSink(), 2)
        trigger = coordinator.receive_pcm(b"\x09\x00")
        with self.assertRaises(PCMHandoffError):
            await coordinator.admit_primary(trigger, RecordingSink(AdmissionRejected("stream_scheduler_overrun")))
        coordinator.begin_handoff()
        coordinator.activate_successor()
        first_sink = RecordingSink(None, AdmissionRejected("temporary"))
        failed = await coordinator.drain_handoff(first_sink)
        self.assertFalse(failed.completed)
        self.assertEqual(failed.blocked_reason, "temporary")
        self.assertEqual(coordinator.snapshot().transition_queued_samples, 1)
        retry_sink = RecordingSink()
        complete = await coordinator.drain_handoff(retry_sink)
        self.assertTrue(complete.completed)
        self.assertEqual(retry_sink.calls[-1][2], trigger)

    async def test_public_api_is_limited_to_local_pcm_handoff_contract(self):
        coordinator, controller = self.make_coordinator("epoch-a")
        public_names = {name for name in dir(coordinator) if not name.startswith("_")}
        self.assertEqual(public_names, {
            "activate_successor",
            "admit_primary",
            "begin_handoff",
            "current_fence",
            "drain_handoff",
            "epoch_controller",
            "mark_processed",
            "max_retained_samples",
            "max_transition_samples",
            "primary_admission_open",
            "replay_overlap_samples",
            "receive_pcm",
            "snapshot",
        })
        self.assertEqual(controller.state, EpochControllerState.ACTIVE)
        self.assertEqual(set(coordinator.snapshot().__dataclass_fields__), {
            "source_head_cursor",
            "unique_primary_admitted_cursor",
            "current_epoch_admitted_cursor",
            "processed_cursor",
            "old_processed_cursor",
            "cutover_cursor",
            "replay_start_cursor",
            "retained_range_floor",
            "retained_range_head",
            "received_samples",
            "unique_primary_admitted_samples",
            "replay_admitted_samples",
            "replay_retained_samples",
            "replay_inflight_copy_samples",
            "transition_queued_samples",
            "max_transition_queued_samples",
            "retained_source_samples",
            "max_retained_source_samples",
            "released_source_samples",
            "downstream_admission_rejected_samples",
            "explicit_source_rejected_samples",
            "handoff_in_progress",
        })

    async def test_multiple_handoffs_keep_epoch_sequence_monotonic(self):
        coordinator, controller = self.make_coordinator("epoch-a", "epoch-b", "epoch-c", overlap=1)
        await self.admit(coordinator, RecordingSink(), 4)
        coordinator.begin_handoff()
        first = coordinator.activate_successor()
        await coordinator.drain_handoff(RecordingSink())
        coordinator.mark_processed(fence=first, start_sample=0, end_sample=4)

        coordinator.begin_handoff()
        second = coordinator.activate_successor()
        self.assertEqual(second.epoch_seq, first.epoch_seq + 1)
        self.assertEqual(second.epoch_id, "epoch-c")
        result = await coordinator.drain_handoff(RecordingSink())
        self.assertTrue(result.completed)
        self.assertEqual(controller.current, second)

if __name__ == "__main__":
    unittest.main()
