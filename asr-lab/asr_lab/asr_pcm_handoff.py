"""Bounded local PCM ownership across acoustic-epoch handoff.

This coordinator retains source-coordinate PCM until it is safe to retire.
It composes A6.1's epoch controller; it does not create another epoch
authority, operate a Qwen session, or provide process-loss recovery.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from .asr_epoch_controller import EpochControllerState, LocalAcousticEpochController
from .asr_fencing import ExecutionFence, FenceOutcome


class PCMAdmissionKind(str, Enum):
    PRIMARY = "PRIMARY"
    REPLAY = "REPLAY"


class PCMRole(str, Enum):
    UNADMITTED = "UNADMITTED"
    PRIMARY = "PRIMARY"
    TRANSITION = "TRANSITION"


class AdmissionState(str, Enum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"


class PCMHandoffError(ValueError):
    """Stable local PCM handoff contract error."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(detail or code)


class AdmissionRejected(RuntimeError):
    """A sink's explicit assertion that the offered PCM was not admitted."""

    def __init__(self, code: str = "admission_rejected") -> None:
        self.code = code
        super().__init__(code)


class PCMCapacityRejected(PCMHandoffError):
    """Explicit source-side backpressure; no PCM bytes were retained/sent."""

    def __init__(self, code: str, span: "PCMSpan") -> None:
        self.span = span
        super().__init__(code)


@dataclass(frozen=True)
class PCMSpan:
    """Immutable PCM16LE over a half-open absolute source-sample interval."""

    start_sample: int
    end_sample: int
    pcm16le: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_sample, bool)
            or not isinstance(self.start_sample, int)
            or isinstance(self.end_sample, bool)
            or not isinstance(self.end_sample, int)
            or self.start_sample < 0
            or self.end_sample <= self.start_sample
            or not isinstance(self.pcm16le, bytes)
            or len(self.pcm16le) != (self.end_sample - self.start_sample) * 2
        ):
            raise PCMHandoffError("invalid_pcm_span")

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample

    def slice(self, start_sample: int, end_sample: int) -> "PCMSpan":
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or isinstance(end_sample, bool)
            or not isinstance(end_sample, int)
            or start_sample < self.start_sample
            or end_sample > self.end_sample
            or end_sample <= start_sample
        ):
            raise PCMHandoffError("invalid_pcm_slice")
        byte_start = (start_sample - self.start_sample) * 2
        byte_end = (end_sample - self.start_sample) * 2
        return PCMSpan(start_sample, end_sample, self.pcm16le[byte_start:byte_end])


@dataclass(frozen=True)
class AdmissionAck:
    accepted_samples: int


class PCMAdmissionSink(Protocol):
    async def submit(
        self, kind: PCMAdmissionKind, fence: ExecutionFence, span: PCMSpan,
    ) -> AdmissionAck: ...


@dataclass(frozen=True)
class HandoffDrainResult:
    completed: bool
    replay_admitted_samples: int
    primary_admitted_samples: int
    blocked_reason: str | None = None
    failure_stage: str | None = None


@dataclass(frozen=True)
class PCMHandoffSnapshot:
    source_head_cursor: int
    unique_primary_admitted_cursor: int
    current_epoch_admitted_cursor: int
    processed_cursor: int
    old_processed_cursor: int | None
    cutover_cursor: int | None
    replay_start_cursor: int | None
    retained_range_floor: int
    retained_range_head: int
    received_samples: int
    unique_primary_admitted_samples: int
    replay_admitted_samples: int
    replay_retained_samples: int
    replay_inflight_copy_samples: int
    transition_queued_samples: int
    max_transition_queued_samples: int
    retained_source_samples: int
    max_retained_source_samples: int
    released_source_samples: int
    downstream_admission_rejected_samples: int
    explicit_source_rejected_samples: int
    handoff_in_progress: bool


@dataclass
class _OwnedSpan:
    span: PCMSpan
    role: PCMRole
    admission_state: AdmissionState = AdmissionState.PENDING
    primary_epoch: ExecutionFence | None = None
    ambiguous_epoch: ExecutionFence | None = None


class LocalPCMHandoffCoordinator:
    """Own bounded PCM and serialize its PRIMARY/REPLAY handoff admissions.

    ``receive_pcm`` synchronously copies and accounts source bytes before any
    await or downstream submission. The epoch controller is the sole source
    of current/successor lineage. The coordinator assumes its process survives.
    """

    def __init__(
        self,
        *,
        epoch_controller: LocalAcousticEpochController,
        replay_overlap_samples: int,
        max_retained_samples: int,
        max_transition_samples: int,
    ) -> None:
        if not isinstance(epoch_controller, LocalAcousticEpochController):
            raise PCMHandoffError("invalid_epoch_controller")
        if (
            isinstance(replay_overlap_samples, bool)
            or not isinstance(replay_overlap_samples, int)
            or replay_overlap_samples < 0
        ):
            raise PCMHandoffError("invalid_replay_overlap_samples")
        if (
            isinstance(max_retained_samples, bool)
            or not isinstance(max_retained_samples, int)
            or max_retained_samples <= 0
        ):
            raise PCMHandoffError("invalid_max_retained_samples")
        if (
            isinstance(max_transition_samples, bool)
            or not isinstance(max_transition_samples, int)
            or max_transition_samples <= 0
        ):
            raise PCMHandoffError("invalid_max_transition_samples")
        if max_transition_samples > max_retained_samples:
            raise PCMHandoffError("transition_bound_exceeds_retention_bound")
        if epoch_controller.state is not EpochControllerState.ACTIVE:
            raise PCMHandoffError("epoch_controller_not_active")

        self.epoch_controller = epoch_controller
        self.replay_overlap_samples = replay_overlap_samples
        self.max_retained_samples = max_retained_samples
        self.max_transition_samples = max_transition_samples

        self._records: list[_OwnedSpan] = []
        self._source_head_cursor = 0
        self._unique_primary_cursor = 0
        self._current_epoch_admitted_cursor = 0
        self._processed_cursor = 0
        self._handoff_old_processed_cursor: int | None = None
        self._cutover_cursor: int | None = None
        self._replay_start_cursor: int | None = None
        self._replay_end_cursor: int | None = None
        self._replay_plan: list[tuple[int, int]] = []
        self._replay_acked: set[tuple[str, int, int]] = set()
        self._ambiguous_replay: set[tuple[str, int, int]] = set()
        self._successor_fence: ExecutionFence | None = None
        self._rollover_from_fence: ExecutionFence | None = None
        self._primary_open = True
        self._handoff_in_progress = False
        self._admission_in_flight = False
        self._admission_lock = asyncio.Lock()
        self._drain_lock = asyncio.Lock()

        self._received_samples = 0
        self._unique_primary_admitted_samples = 0
        self._replay_admitted_samples = 0
        self._max_transition_queued_samples = 0
        self._max_retained_source_samples = 0
        self._released_source_samples = 0
        self._downstream_rejected_samples = 0
        self._explicit_source_rejected_samples = 0
        self._source_gap_started = False
        self._replay_inflight_copy_samples = 0

    @property
    def current_fence(self) -> ExecutionFence:
        return self.epoch_controller.current

    @property
    def primary_admission_open(self) -> bool:
        return self._primary_open and not self._handoff_in_progress

    def receive_pcm(self, pcm16le: bytes) -> PCMSpan:
        """Take local ownership before a caller awaits or submits downstream."""
        if not isinstance(pcm16le, (bytes, bytearray, memoryview)):
            raise PCMHandoffError("invalid_audio")
        payload = bytes(pcm16le)
        if not payload or len(payload) % 2:
            raise PCMHandoffError("invalid_audio")

        samples = len(payload) // 2
        start = self._source_head_cursor
        end = start + samples
        span = PCMSpan(start, end, payload)
        self._source_head_cursor = end
        self._received_samples += samples

        role = PCMRole.UNADMITTED if self.primary_admission_open else PCMRole.TRANSITION
        transition_reserved = self._transition_queued_samples()
        capacity_error: str | None = None
        if self._source_gap_started:
            capacity_error = "source_after_rejected_gap"
        elif self._retained_source_samples() + self._replay_inflight_copy_samples + samples > self.max_retained_samples:
            capacity_error = "retained_capacity_exceeded"
        elif role is PCMRole.TRANSITION and transition_reserved + samples > self.max_transition_samples:
            capacity_error = "transition_capacity_exceeded"

        if capacity_error is not None:
            self._explicit_source_rejected_samples += samples
            self._source_gap_started = True
            self._primary_open = False
            raise PCMCapacityRejected(capacity_error, span)

        self._records.append(_OwnedSpan(span=span, role=role))
        self._observe_occupancy_high_water()
        return span

    async def admit_primary(self, span: PCMSpan, sink: PCMAdmissionSink) -> AdmissionAck:
        """Attempt old/current PRIMARY admission for a locally owned span."""
        async with self._admission_lock:
            record = self._find_exact_record(span)
            if not self.primary_admission_open:
                raise PCMHandoffError("primary_admission_closed")
            if record.role is not PCMRole.UNADMITTED or record.admission_state is not AdmissionState.PENDING:
                raise PCMHandoffError("primary_span_not_pending")
            if record.ambiguous_epoch is not None:
                raise PCMHandoffError("ambiguous_admission_requires_resolution")
            if record.span.start_sample != self._unique_primary_cursor:
                raise PCMHandoffError("non_contiguous_primary_source")

            fence = self.epoch_controller.current
            self._admission_in_flight = True
            record.admission_state = AdmissionState.IN_FLIGHT
            try:
                ack = await sink.submit(PCMAdmissionKind.PRIMARY, fence, record.span)
            except AdmissionRejected as exc:
                self._downstream_rejected_samples += record.span.sample_count
                record.role = PCMRole.TRANSITION
                record.admission_state = AdmissionState.REJECTED
                self._primary_open = False
                raise PCMHandoffError("admission_rejected", exc.code) from exc
            except asyncio.CancelledError:
                record.role = PCMRole.TRANSITION
                record.admission_state = AdmissionState.AMBIGUOUS
                record.ambiguous_epoch = fence
                self._primary_open = False
                raise
            except Exception as exc:
                record.role = PCMRole.TRANSITION
                record.admission_state = AdmissionState.AMBIGUOUS
                record.ambiguous_epoch = fence
                self._primary_open = False
                raise PCMHandoffError("admission_unconfirmed") from exc
            finally:
                self._admission_in_flight = False
                self._observe_occupancy_high_water()

            if not self._valid_ack(ack, record.span.sample_count):
                record.role = PCMRole.TRANSITION
                record.admission_state = AdmissionState.AMBIGUOUS
                record.ambiguous_epoch = fence
                self._primary_open = False
                raise PCMHandoffError("admission_ack_mismatch")
            if self.epoch_controller.current != fence:
                record.role = PCMRole.TRANSITION
                record.admission_state = AdmissionState.AMBIGUOUS
                record.ambiguous_epoch = fence
                self._primary_open = False
                raise PCMHandoffError("admission_fence_changed")

            self._admit_unique_primary(record, fence)
            return ack

    def begin_handoff(self) -> str:
        """Close old PRIMARY admission and reserve one A6.1 successor ID."""
        if self._admission_in_flight:
            raise PCMHandoffError("admission_in_flight")
        if self._handoff_in_progress or self._successor_fence is not None:
            raise PCMHandoffError("handoff_already_started")
        if self.epoch_controller.state is not EpochControllerState.ACTIVE:
            raise PCMHandoffError("epoch_controller_not_active")

        self._replay_plan = []
        self._replay_acked.clear()
        self._ambiguous_replay.clear()
        self._successor_fence = None
        self._handoff_old_processed_cursor = None
        self._replay_start_cursor = None
        self._replay_end_cursor = None
        self._primary_open = False
        self._handoff_in_progress = True
        for record in self._records:
            if record.primary_epoch is None and record.role is PCMRole.UNADMITTED:
                record.role = PCMRole.TRANSITION
        self._observe_occupancy_high_water()
        transition_starts = [
            record.span.start_sample for record in self._records
            if record.primary_epoch is None
        ]
        self._cutover_cursor = min(transition_starts, default=self._source_head_cursor)
        self._rollover_from_fence = self.epoch_controller.current
        self.epoch_controller.request_rollover()
        return self.epoch_controller.prepare_successor()

    def activate_successor(self) -> ExecutionFence:
        """Freeze old processing, derive bounded replay, and admit A6.1 successor."""
        if not self._handoff_in_progress or self._successor_fence is not None:
            raise PCMHandoffError("handoff_not_prepared")
        if self._admission_in_flight:
            raise PCMHandoffError("admission_in_flight")
        if self.epoch_controller.state is not EpochControllerState.SUCCESSOR_PREPARED:
            raise PCMHandoffError("successor_not_prepared")

        old_processed = self._processed_cursor
        replay_end = self._unique_primary_cursor
        retained_floor = self._retained_range_floor()
        replay_start = max(retained_floor, old_processed - self.replay_overlap_samples)
        if replay_start > replay_end:
            raise PCMHandoffError("replay_cursor_out_of_range")
        replay_plan = self._retained_primary_ranges(replay_start, replay_end)

        successor = self.epoch_controller.activate_successor()
        self._handoff_old_processed_cursor = old_processed
        self._replay_start_cursor = replay_start
        self._replay_end_cursor = replay_end
        self._replay_plan = replay_plan
        self._successor_fence = successor
        self._processed_cursor = replay_start
        self._current_epoch_admitted_cursor = replay_start
        return successor

    async def drain_handoff(
        self,
        sink: PCMAdmissionSink,
        *,
        stage_observer: Callable[[str], None] | None = None,
    ) -> HandoffDrainResult:
        """Submit all replay first, then transition PCM as unique PRIMARY.

        Known rejection is retryable only by a later explicit call. Ambiguous
        outcomes are never automatically retried because they could duplicate
        a PRIMARY admission.
        """
        async with self._drain_lock:
            successor = self._successor_fence
            if successor is None:
                raise PCMHandoffError("successor_not_activated")
            if self.epoch_controller.current != successor:
                raise PCMHandoffError("successor_fence_not_current")
            replay_this_call = 0
            primary_this_call = 0
            if stage_observer is not None:
                stage_observer("REPLAY_DRAINING")

            for start_sample, end_sample in self._replay_plan:
                key = (successor.epoch_id, start_sample, end_sample)
                if key in self._replay_acked:
                    continue
                if key in self._ambiguous_replay:
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "ambiguous_replay_admission", "replay")
                span, is_copy = self._retained_primary_span(start_sample, end_sample)
                if is_copy:
                    self._replay_inflight_copy_samples += span.sample_count
                try:
                    ack = await sink.submit(PCMAdmissionKind.REPLAY, successor, span)
                except AdmissionRejected as exc:
                    self._downstream_rejected_samples += span.sample_count
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, exc.code, "replay")
                except asyncio.CancelledError:
                    self._ambiguous_replay.add(key)
                    raise
                except Exception:
                    self._ambiguous_replay.add(key)
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "ambiguous_replay_admission", "replay")
                finally:
                    if is_copy:
                        self._replay_inflight_copy_samples -= span.sample_count
                if not self._valid_ack(ack, span.sample_count):
                    self._ambiguous_replay.add(key)
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "ambiguous_replay_ack", "replay")
                if span.start_sample != self._current_epoch_admitted_cursor:
                    raise PCMHandoffError("non_contiguous_replay_plan")
                self._replay_acked.add(key)
                self._replay_admitted_samples += span.sample_count
                replay_this_call += span.sample_count
                self._current_epoch_admitted_cursor = span.end_sample

            if self._current_epoch_admitted_cursor != self._unique_primary_cursor:
                raise PCMHandoffError("replay_does_not_reach_primary_cursor")

            if stage_observer is not None:
                stage_observer("REPLAY_DRAINED")
                stage_observer("CATCHUP_DRAINING")
            while True:
                record = self._next_transition_record()
                if record is None:
                    break
                if record.admission_state is AdmissionState.AMBIGUOUS:
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "ambiguous_primary_admission", "catchup")
                if record.span.start_sample != self._unique_primary_cursor:
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "source_gap_before_transition", "catchup")
                record.admission_state = AdmissionState.IN_FLIGHT
                self._admission_in_flight = True
                try:
                    ack = await sink.submit(PCMAdmissionKind.PRIMARY, successor, record.span)
                except AdmissionRejected as exc:
                    self._downstream_rejected_samples += record.span.sample_count
                    record.admission_state = AdmissionState.REJECTED
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, exc.code, "catchup")
                except asyncio.CancelledError:
                    record.admission_state = AdmissionState.AMBIGUOUS
                    record.ambiguous_epoch = successor
                    raise
                except Exception:
                    record.admission_state = AdmissionState.AMBIGUOUS
                    record.ambiguous_epoch = successor
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "ambiguous_primary_admission", "catchup")
                finally:
                    self._admission_in_flight = False

                if not self._valid_ack(ack, record.span.sample_count):
                    record.admission_state = AdmissionState.AMBIGUOUS
                    record.ambiguous_epoch = successor
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "ambiguous_primary_ack", "catchup")
                if self.epoch_controller.current != successor:
                    record.admission_state = AdmissionState.AMBIGUOUS
                    record.ambiguous_epoch = successor
                    return HandoffDrainResult(False, replay_this_call, primary_this_call, "primary_fence_changed", "catchup")
                self._admit_unique_primary(record, successor)
                primary_this_call += record.span.sample_count

            if stage_observer is not None:
                stage_observer("CATCHUP_DRAINED")
            self._handoff_in_progress = False
            self._primary_open = True
            self._successor_fence = None
            self._rollover_from_fence = None
            return HandoffDrainResult(True, replay_this_call, primary_this_call)

    def mark_processed(
        self, *, fence: ExecutionFence, start_sample: int, end_sample: int,
    ) -> int:
        """Advance only a contiguous current-epoch processing watermark."""
        if not isinstance(fence, ExecutionFence):
            raise PCMHandoffError("invalid_processing_fence")
        if fence != self.epoch_controller.current:
            outcome = self.epoch_controller.classify(fence)
            code = "stale_processing_fence" if outcome is FenceOutcome.STALE else "invalid_processing_fence"
            raise PCMHandoffError(code)
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or isinstance(end_sample, bool)
            or not isinstance(end_sample, int)
            or start_sample < 0
            or end_sample <= start_sample
        ):
            raise PCMHandoffError("invalid_processing_range")
        if start_sample != self._processed_cursor:
            raise PCMHandoffError("non_contiguous_processing_completion")
        if end_sample > self._current_epoch_admitted_cursor:
            raise PCMHandoffError("processing_beyond_admitted_audio")

        self._processed_cursor = end_sample
        self._release_processed_prefix()
        return self._processed_cursor

    def snapshot(self) -> PCMHandoffSnapshot:
        return PCMHandoffSnapshot(
            source_head_cursor=self._source_head_cursor,
            unique_primary_admitted_cursor=self._unique_primary_cursor,
            current_epoch_admitted_cursor=self._current_epoch_admitted_cursor,
            processed_cursor=self._processed_cursor,
            old_processed_cursor=self._handoff_old_processed_cursor,
            cutover_cursor=self._cutover_cursor,
            replay_start_cursor=self._replay_start_cursor,
            retained_range_floor=self._retained_range_floor(),
            retained_range_head=max((item.span.end_sample for item in self._records), default=self._source_head_cursor),
            received_samples=self._received_samples,
            unique_primary_admitted_samples=self._unique_primary_admitted_samples,
            replay_admitted_samples=self._replay_admitted_samples,
            replay_retained_samples=sum(
                item.span.sample_count for item in self._records if item.primary_epoch is not None
            ),
            replay_inflight_copy_samples=self._replay_inflight_copy_samples,
            transition_queued_samples=self._transition_queued_samples(),
            max_transition_queued_samples=self._max_transition_queued_samples,
            retained_source_samples=self._retained_source_samples(),
            max_retained_source_samples=self._max_retained_source_samples,
            released_source_samples=self._released_source_samples,
            downstream_admission_rejected_samples=self._downstream_rejected_samples,
            explicit_source_rejected_samples=self._explicit_source_rejected_samples,
            handoff_in_progress=self._handoff_in_progress,
        )

    def _observe_occupancy_high_water(self) -> None:
        self._max_transition_queued_samples = max(
            self._max_transition_queued_samples, self._transition_queued_samples(),
        )
        self._max_retained_source_samples = max(
            self._max_retained_source_samples, self._retained_source_samples(),
        )

    def _admit_unique_primary(self, record: _OwnedSpan, fence: ExecutionFence) -> None:
        if record.span.start_sample != self._unique_primary_cursor:
            raise PCMHandoffError("non_contiguous_primary_source")
        record.role = PCMRole.PRIMARY
        record.admission_state = AdmissionState.ACKNOWLEDGED
        record.primary_epoch = fence
        record.ambiguous_epoch = None
        self._unique_primary_cursor = record.span.end_sample
        self._current_epoch_admitted_cursor = record.span.end_sample
        self._unique_primary_admitted_samples += record.span.sample_count

    def _next_transition_record(self) -> _OwnedSpan | None:
        for record in self._records:
            if record.primary_epoch is None:
                return record
        return None

    def _find_exact_record(self, span: PCMSpan) -> _OwnedSpan:
        if not isinstance(span, PCMSpan):
            raise PCMHandoffError("invalid_pcm_span")
        for record in self._records:
            if record.span == span:
                return record
        raise PCMHandoffError("pcm_not_owned")

    def _retained_primary_ranges(self, start_sample: int, end_sample: int) -> list[tuple[int, int]]:
        if start_sample == end_sample:
            return []
        cursor = start_sample
        result: list[tuple[int, int]] = []
        for record in self._records:
            span = record.span
            if span.end_sample <= cursor or span.start_sample >= end_sample:
                continue
            if record.primary_epoch is None:
                continue
            slice_start = max(cursor, span.start_sample)
            slice_end = min(end_sample, span.end_sample)
            if slice_start != cursor:
                raise PCMHandoffError("replay_source_gap")
            result.append((slice_start, slice_end))
            cursor = slice_end
            if cursor == end_sample:
                break
        if cursor != end_sample:
            raise PCMHandoffError("replay_source_unavailable")
        return result

    def _retained_primary_span(self, start_sample: int, end_sample: int) -> tuple[PCMSpan, bool]:
        for record in self._records:
            span = record.span
            if record.primary_epoch is None or span.start_sample > start_sample or span.end_sample < end_sample:
                continue
            if span.start_sample == start_sample and span.end_sample == end_sample:
                return span, False
            return span.slice(start_sample, end_sample), True
        raise PCMHandoffError("replay_source_unavailable")

    def _release_processed_prefix(self) -> None:
        releasable_to = max(0, self._processed_cursor - self.replay_overlap_samples)
        if releasable_to <= self._retained_range_floor():
            return
        kept: list[_OwnedSpan] = []
        for record in self._records:
            span = record.span
            if record.primary_epoch is None or span.start_sample >= releasable_to:
                kept.append(record)
                continue
            if span.end_sample <= releasable_to:
                self._released_source_samples += span.sample_count
                continue
            released = releasable_to - span.start_sample
            record.span = span.slice(releasable_to, span.end_sample)
            self._released_source_samples += released
            kept.append(record)
        self._records = kept

    def _retained_range_floor(self) -> int:
        return min((item.span.start_sample for item in self._records), default=self._source_head_cursor)

    def _retained_source_samples(self) -> int:
        return sum(item.span.sample_count for item in self._records)

    def _transition_queued_samples(self) -> int:
        return sum(
            item.span.sample_count
            for item in self._records
            if item.role is PCMRole.TRANSITION
        )

    @staticmethod
    def _valid_ack(ack: object, expected_samples: int) -> bool:
        return (
            isinstance(ack, AdmissionAck)
            and not isinstance(ack.accepted_samples, bool)
            and isinstance(ack.accepted_samples, int)
            and ack.accepted_samples == expected_samples
        )
