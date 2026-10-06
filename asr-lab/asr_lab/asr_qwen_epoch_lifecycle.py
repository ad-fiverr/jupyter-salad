"""Compose canonical acoustic epochs, owned PCM handoff, and local Qwen state.

This adapter owns no epoch identity and no PCM bytes. A6.1 remains the epoch
authority, A6.2 remains the PCM owner, and Qwen receives a fresh local stream
for each canonical epoch.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping

from .asr_epoch_controller import EpochControllerState, LocalAcousticEpochController
from .asr_fencing import ExecutionFence, FenceOutcome
from .asr_candidate_stitching import (
    ASRCandidateStitcher,
    CandidateBase,
    CandidateStitcherSnapshot,
)
from .asr_pcm_handoff import (
    AdmissionAck,
    AdmissionRejected,
    HandoffDrainResult,
    LocalPCMHandoffCoordinator,
    PCMHandoffError,
    PCMAdmissionKind,
    PCMAdmissionSink,
    PCMSpan,
)
from .asr_epoch_observability import ASREpochObservability
from .asr_epoch_rollover_policy import (
    RolloverCategory,
    SoftRolloverEvaluator,
    SoftRolloverPolicy,
    choose_rollover_category,
)
from .qwen_streaming import (
    QwenStreamingRuntime,
    StreamingError,
    validate_candidate_event,
)


class QwenEpochLifecycleError(RuntimeError):
    """Fail-closed local lifecycle contract error."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(detail or code)


class QwenEpochLifecycleState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    HANDOFF_PENDING = "HANDOFF_PENDING"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class QwenEpochLifecycleSnapshot:
    state: QwenEpochLifecycleState
    canonical_fence: ExecutionFence | None
    local_stream_id: str | None
    canonical_base_cursor: int
    observed_local_processed_cursor: int
    handoff_in_progress: bool


CandidateEventSink = Callable[[dict[str, Any]], Awaitable[None]]
CandidateObserver = Callable[[ExecutionFence, dict[str, Any]], None]
StaleRejectObserver = Callable[[str], None]


@dataclass
class _QwenCandidateFenceBinding:
    """Immutable-per-session canonical provenance for candidate results."""

    epoch_controller: LocalAcousticEpochController
    _fence: ExecutionFence | None = None
    candidate_stitcher: ASRCandidateStitcher = dataclass_field(
        default_factory=ASRCandidateStitcher
    )
    base: CandidateBase | None = None
    candidate_observer: CandidateObserver | None = None
    stale_reject_observer: StaleRejectObserver | None = None
    stream_id: str | None = None

    def __post_init__(self) -> None:
        if self._fence is not None:
            self.candidate_stitcher.activate_epoch(self._fence, base=self.base)

    def bind(self, fence: ExecutionFence) -> None:
        if self._fence is not None or not isinstance(fence, ExecutionFence):
            raise QwenEpochLifecycleError("candidate_fence_binding_invalid")
        self.candidate_stitcher.activate_epoch(fence, base=self.base)
        self._fence = fence

    def filter_candidate(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        fence = self._fence
        if fence is None:
            return None
        try:
            if self.epoch_controller.classify(fence) is not FenceOutcome.ACCEPT:
                if self.stale_reject_observer is not None:
                    self.stale_reject_observer(self.stream_id or "")
                return None
        except Exception:
            if self.stale_reject_observer is not None:
                self.stale_reject_observer(self.stream_id or "")
            return None
        accepted = dict(candidate)
        accepted.update({
            "asr_job_id": fence.asr_job_id,
            "speech_segment_id": fence.speech_segment_id,
            "epoch_id": fence.epoch_id,
            "epoch_seq": fence.epoch_seq,
        })
        event = accepted.get("event")
        if not isinstance(event, str):
            return None
        try:
            validate_candidate_event(accepted, event)
        except StreamingError:
            return None
        reconciled = self.candidate_stitcher.reconcile(
            accepted,
            trusted_fence=fence,
        )
        if reconciled is not None and self.candidate_observer is not None:
            self.candidate_observer(fence, reconciled)
        return reconciled


class QwenEpochLifecycle(PCMAdmissionSink):
    """Bind A6.2's owned PCM handoff to fresh, fenced local Qwen streams.

    ``submit_pcm`` copies source PCM into A6.2 before its first await. During a
    handoff, newly received spans remain owned by A6.2 and are drained in its
    REPLAY-then-TRANSITION order. Decode completion is observed separately
    from scheduler admission and is synchronized into A6.2 only at lifecycle
    boundaries, capped by the already-acknowledged canonical cursor.
    """

    _KNOWN_ADMISSION_REJECTIONS = frozenset({
        "backend_not_ready",
        "invalid_audio",
        "stream_not_started",
        "stream_duration_limit",
        "stream_scheduler_overrun",
    })

    def __init__(
        self,
        *,
        epoch_controller: LocalAcousticEpochController,
        pcm_handoff: LocalPCMHandoffCoordinator,
        qwen_runtime: QwenStreamingRuntime,
        connection_id: str,
        source: str,
        language: str | None = None,
        context: str = "",
        chunk_size_ms: int | None = None,
        request_id: str | None = None,
        event_sink: CandidateEventSink | None = None,
        soft_rollover_policy: SoftRolloverPolicy | None = None,
        observability: ASREpochObservability | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(epoch_controller, LocalAcousticEpochController):
            raise QwenEpochLifecycleError("invalid_epoch_controller")
        if not isinstance(pcm_handoff, LocalPCMHandoffCoordinator):
            raise QwenEpochLifecycleError("invalid_pcm_handoff")
        if pcm_handoff.epoch_controller is not epoch_controller:
            raise QwenEpochLifecycleError("epoch_controller_mismatch")
        if not isinstance(connection_id, str) or not connection_id.strip():
            raise QwenEpochLifecycleError("invalid_connection_id")
        if not isinstance(source, str) or not source.strip():
            raise QwenEpochLifecycleError("invalid_source")
        if not isinstance(qwen_runtime, QwenStreamingRuntime):
            raise QwenEpochLifecycleError("invalid_qwen_runtime")

        self.epoch_controller = epoch_controller
        self.pcm_handoff = pcm_handoff
        self.qwen_runtime = qwen_runtime
        self.connection_id = connection_id
        self.source = source
        self.language = language
        self.context = context
        self.chunk_size_ms = chunk_size_ms
        self.request_id = request_id
        self.event_sink = event_sink
        if not callable(monotonic):
            raise QwenEpochLifecycleError("invalid_monotonic_clock")
        self._clock = monotonic
        self._soft_rollover = (
            SoftRolloverEvaluator(soft_rollover_policy)
            if soft_rollover_policy is not None else None
        )
        self._observability = observability or ASREpochObservability(monotonic=monotonic)

        self._state = QwenEpochLifecycleState.NEW
        self._bound_fence: ExecutionFence | None = None
        self._bound_stream_id: str | None = None
        self._seen_stream_ids: set[str] = set()
        self._candidate_stitcher = ASRCandidateStitcher()
        self._canonical_base_cursor = 0
        self._observed_local_processed_cursor = 0
        self._operation_lock = asyncio.Lock()
        self._closing = False
        self._replay_started_at: float | None = None

    @property
    def state(self) -> QwenEpochLifecycleState:
        return self._state

    @property
    def current_fence(self) -> ExecutionFence | None:
        return self._bound_fence

    @property
    def local_stream_id(self) -> str | None:
        return self._bound_stream_id

    @property
    def candidate_stitcher_snapshot(self) -> CandidateStitcherSnapshot:
        """Read-only A6.5 state for focused lifecycle verification."""
        return self._candidate_stitcher.snapshot()

    def snapshot(self) -> QwenEpochLifecycleSnapshot:
        return QwenEpochLifecycleSnapshot(
            state=self._state,
            canonical_fence=self._bound_fence,
            local_stream_id=self._bound_stream_id,
            canonical_base_cursor=self._canonical_base_cursor,
            observed_local_processed_cursor=self._observed_local_processed_cursor,
            handoff_in_progress=self.pcm_handoff.snapshot().handoff_in_progress,
        )

    async def initialize(self) -> dict[str, Any]:
        """Open the initial Qwen state under A6.1's already-current fence."""
        async with self._operation_lock:
            if self._state is not QwenEpochLifecycleState.NEW:
                raise QwenEpochLifecycleError("lifecycle_already_initialized")
            if self.epoch_controller.state is not EpochControllerState.ACTIVE:
                raise QwenEpochLifecycleError("epoch_controller_not_active")
            fence = self.epoch_controller.current
            opened, _candidate_binding = await self._open_fresh_qwen_state(
                forbidden_ids={fence.epoch_id},
                candidate_fence=fence,
            )
            self._bound_fence = fence
            self._bound_stream_id = opened["stream_id"]
            self._canonical_base_cursor = 0
            self._observed_local_processed_cursor = 0
            self._state = QwenEpochLifecycleState.ACTIVE
            self._observability.start_epoch(
                fence=fence,
                stream_id=opened["stream_id"],
                stream_init_ms=self._opened_init_ms(opened),
                state_init_ms=self._opened_state_init_ms(opened),
                now=self._clock(),
            )
            self._observe_operational_metrics()
            return {**opened, "epoch_fence": fence}

    def observability_snapshot(self) -> dict[str, Any]:
        """Return bounded scalar epoch, transition, and logical totals."""
        self._observe_operational_metrics()
        snapshot = self._observability.snapshot(now=self._clock())
        if self._soft_rollover is not None:
            snapshot["soft_rollover_policy"] = self._soft_rollover.snapshot(now=self._clock())
        else:
            snapshot["soft_rollover_policy"] = {"soft_enabled": False, "armed": False}
        return snapshot

    async def submit_pcm(self, pcm16le: bytes) -> AdmissionAck | None:
        """Own source PCM immediately, then submit it only if still live PRIMARY.

        ``None`` means A6.2 already owns this span as TRANSITION and will send
        it during the in-progress handoff. It never means the span was dropped.
        """
        if self._closing or self._state in {
            QwenEpochLifecycleState.NEW,
            QwenEpochLifecycleState.FAILED,
            QwenEpochLifecycleState.CLOSED,
        }:
            raise QwenEpochLifecycleError("lifecycle_not_accepting_pcm")

        source_fence = self.pcm_handoff.current_fence
        try:
            span = self.pcm_handoff.receive_pcm(pcm16le)
        except PCMHandoffError as exc:
            # Preserve the exact fail-closed admission stage before the service
            # can clean up this source stream. This is observational only: the
            # handoff coordinator remains the sole PCM owner and capacity policy.
            if exc.code == "transition_capacity_exceeded":
                self._observability.record_transition_blocked(
                    failure_stage="transition_capacity",
                    failure_reason=exc.code,
                    pcm_snapshot=self.pcm_handoff.snapshot(),
                )
            raise
        primary_open_at_receive = self.pcm_handoff.primary_admission_open
        if not primary_open_at_receive:
            return None

        async with self._operation_lock:
            # A rollover may have included this already-owned span as
            # TRANSITION while this caller was waiting for the lifecycle lock.
            if (
                self._state is QwenEpochLifecycleState.FAILED
                or self._bound_fence != source_fence
                or self.epoch_controller.current != source_fence
                or not self.pcm_handoff.primary_admission_open
            ):
                return None

            self._observe_operational_metrics()
            now = self._clock()
            policy_decision = None
            if self._soft_rollover is not None:
                policy_decision = self._soft_rollover.observe(
                    self._soft_trigger_metrics(), now=now,
                )
            hard_limit_ms = float(self.qwen_runtime.max_stream_seconds) * 1000.0
            current_audio_ms = self._current_stream_audio_ms()
            incoming_audio_ms = span.sample_count * 1000.0 / 16_000
            hard_bound = current_audio_ms + incoming_audio_ms > hard_limit_ms
            categories: list[RolloverCategory] = []
            if hard_bound:
                categories.append(RolloverCategory.HARD_BOUND)
            if policy_decision is not None and policy_decision.should_rollover:
                categories.append(RolloverCategory.PROACTIVE_SOFT)
            if categories:
                category = choose_rollover_category(*categories)
                details: dict[str, Any] = {
                    "observed_audio_ms": round(current_audio_ms + incoming_audio_ms, 3),
                    "hard_limit_ms": round(hard_limit_ms, 3),
                }
                if policy_decision is not None:
                    details.update({
                        "trigger_signals": policy_decision.active_signals,
                        "active_signals": policy_decision.active_signals,
                        "qualifying_samples": policy_decision.qualifying_samples,
                        "policy_window_samples": (
                            self._soft_rollover.policy.window_samples
                            if self._soft_rollover is not None else None
                        ),
                        "ignored_capacity_signals": policy_decision.ignored_capacity_signals,
                    })
                await self._rollover_locked(
                    category=category,
                    reason=("hard_bound" if category is RolloverCategory.HARD_BOUND else "proactive_soft"),
                    details=details,
                )
                # A6.2 owns this triggering frame as TRANSITION and drains it
                # exactly once into the successor.
                return None
            try:
                ack = await self.pcm_handoff.admit_primary(span, self)
            except PCMHandoffError as exc:
                if exc.code == "admission_rejected" and str(exc) == "stream_scheduler_overrun":
                    await self._rollover_locked(
                        category=RolloverCategory.EMERGENCY_BACKPRESSURE,
                        reason="stream_scheduler_overrun",
                        details={"scheduler_overrun_reason": "stream_scheduler_overrun"},
                    )
                    return None
                if exc.code != "admission_rejected":
                    self._state = QwenEpochLifecycleState.FAILED
                raise
            self._sync_processed()
            self._observe_operational_metrics()
            return ack

    async def submit(
        self, kind: PCMAdmissionKind, fence: ExecutionFence, span: PCMSpan,
    ) -> AdmissionAck:
        """A6.2 sink: enqueue exact PCM in the currently bound Qwen stream."""
        if kind not in {PCMAdmissionKind.PRIMARY, PCMAdmissionKind.REPLAY}:
            raise AdmissionRejected("invalid_admission_kind")
        if (
            self._bound_fence is None
            or self._bound_stream_id is None
            or fence != self._bound_fence
            or fence != self.epoch_controller.current
        ):
            raise AdmissionRejected("stale_qwen_epoch_binding")
        if kind is PCMAdmissionKind.REPLAY and self._state is not QwenEpochLifecycleState.HANDOFF_PENDING:
            raise AdmissionRejected("replay_outside_handoff")
        if kind is PCMAdmissionKind.PRIMARY and self._state not in {
            QwenEpochLifecycleState.ACTIVE,
            QwenEpochLifecycleState.HANDOFF_PENDING,
        }:
            raise AdmissionRejected("primary_without_qwen_epoch")

        if kind is PCMAdmissionKind.REPLAY and self._replay_started_at is None:
            self._replay_started_at = self._clock()
        elif kind is PCMAdmissionKind.PRIMARY and self._state is QwenEpochLifecycleState.HANDOFF_PENDING:
            self._finish_replay_wall()

        try:
            await self.qwen_runtime.push_audio(
                connection_id=self.connection_id,
                source=self.source,
                pcm16le=span.pcm16le,
            )
        except StreamingError as exc:
            if exc.code in self._KNOWN_ADMISSION_REJECTIONS:
                raise AdmissionRejected(exc.code) from exc
            raise
        return AdmissionAck(accepted_samples=span.sample_count)

    async def rollover(self) -> HandoffDrainResult:
        """Create fresh successor Qwen state and drain A6.2's ordered handoff."""
        async with self._operation_lock:
            return await self._rollover_locked(
                category=RolloverCategory.MANUAL_TEST,
                reason="manual_test",
                details={},
            )

    async def drain_pending_handoff(self) -> HandoffDrainResult:
        """Explicitly continue an incomplete A6.2 drain; never retries itself."""
        async with self._operation_lock:
            if (
                self._state is not QwenEpochLifecycleState.HANDOFF_PENDING
                or self._bound_fence is None
                or self._bound_stream_id is None
                or self.epoch_controller.current != self._bound_fence
            ):
                raise QwenEpochLifecycleError("handoff_not_ready_for_drain")
            failure_stage = "replay"

            def observe_stage(stage: str) -> None:
                nonlocal failure_stage
                self._observability.record_transition_stage(stage, now=self._clock())
                if stage.startswith("CATCHUP"):
                    failure_stage = "catchup"
                elif stage.startswith("REPLAY"):
                    failure_stage = "replay"

            try:
                result = await self.pcm_handoff.drain_handoff(
                    self, stage_observer=observe_stage,
                )
            except Exception as exc:
                self._observability.record_transition_failure(
                    failure_stage=failure_stage,
                    failure_reason=self._error_code(exc),
                )
                raise
            if result.completed:
                self._state = QwenEpochLifecycleState.ACTIVE
                self._sync_processed()
                self._complete_observed_transition(result)
            else:
                self._observability.record_transition_blocked(
                    failure_stage=result.failure_stage,
                    failure_reason=result.blocked_reason,
                    pcm_snapshot=self.pcm_handoff.snapshot(),
                )
                self._observe_operational_metrics()
            return result

    async def finish(self) -> dict[str, Any] | None:
        """Run natural EOS once on the current epoch; rollover never calls this."""
        if self._state is not QwenEpochLifecycleState.ACTIVE:
            raise QwenEpochLifecycleError("finish_requires_active_epoch")
        self._closing = True
        async with self._operation_lock:
            if self._state is not QwenEpochLifecycleState.ACTIVE:
                raise QwenEpochLifecycleError("finish_requires_active_epoch")
            try:
                self._sync_processed()
                result = await self.qwen_runtime.finish(
                    connection_id=self.connection_id,
                    source=self.source,
                    request_id=self.request_id,
                )
                # finish_stream drains the residual push before its final RPC;
                # the observer records that push before this synchronization.
                self._sync_processed()
                self._observe_operational_metrics()
            except Exception:
                self._state = QwenEpochLifecycleState.FAILED
                raise
            self._state = QwenEpochLifecycleState.CLOSED
            self._observability.end_epoch(end_state="NATURAL_EOS", now=self._clock())
            self._bound_fence = None
            self._bound_stream_id = None
            return result

    async def dispose(self) -> None:
        """Close local runtime state without EOS or a fabricated final result.

        Disconnects and service-side aborts are not natural end-of-speech.
        Clearing the binding before awaiting the runtime also makes repeated
        disposal idempotent and prevents lifecycle observers from advancing
        after the owner has detached this source.
        """
        if self._state is QwenEpochLifecycleState.CLOSED and self._bound_stream_id is None:
            return
        self._closing = True
        async with self._operation_lock:
            stream_id = self._bound_stream_id
            self._bound_fence = None
            self._bound_stream_id = None
            self._state = QwenEpochLifecycleState.CLOSED
            if stream_id is not None:
                await self.qwen_runtime.close_session(
                    connection_id=self.connection_id,
                    source=self.source,
                )

    async def _rollover_locked(
        self,
        *,
        category: RolloverCategory,
        reason: str,
        details: Mapping[str, Any],
    ) -> HandoffDrainResult:
        """Run the existing A6.1-A6.5 handoff once with bounded attribution."""
        if self._state is not QwenEpochLifecycleState.ACTIVE:
            raise QwenEpochLifecycleError("rollover_requires_active_epoch")
        old_fence = self.epoch_controller.current
        old_stream_id = self._bound_stream_id
        if self._bound_fence != old_fence or old_stream_id is None:
            raise QwenEpochLifecycleError("qwen_epoch_binding_mismatch")

        failure_stage = "predecessor_fence"
        self._sync_processed()
        self._replay_started_at = None
        predecessor_metrics = self._stream_snapshot(old_stream_id)
        prepared_epoch_id = self.pcm_handoff.begin_handoff()
        before_successor = self.pcm_handoff.snapshot()
        self._observability.begin_transition(
            category=category.value,
            reason=reason,
            details=details,
            cutover_cursor=before_successor.cutover_cursor,
            predecessor_stream_id=old_stream_id,
            stream_snapshot=predecessor_metrics,
            pcm_snapshot=before_successor,
            now=self._clock(),
        )
        self._state = QwenEpochLifecycleState.HANDOFF_PENDING

        # Detach the old observer before yielding to close_session so a late
        # callback cannot advance successor processing watermarks.
        self._bound_fence = None
        self._bound_stream_id = None
        self._observed_local_processed_cursor = 0
        self._canonical_base_cursor = 0
        try:
            await self.qwen_runtime.close_session(
                connection_id=self.connection_id,
                source=self.source,
            )
            self._observability.record_transition_stage("PREDECESSOR_FENCED", now=self._clock())

            # Preserve A6.5 candidate continuity and its established behavior.
            candidate_base = self._candidate_stitcher.freeze_for_successor(old_fence)
            failure_stage = "successor_open"
            self._observability.record_transition_stage("SUCCESSOR_OPENING", now=self._clock())
            opened, candidate_binding = await self._open_fresh_qwen_state(
                forbidden_ids={old_fence.epoch_id, prepared_epoch_id, old_stream_id},
                candidate_base=candidate_base,
            )
            self._observability.record_transition_stage("SUCCESSOR_OPENED", now=self._clock())
            successor_stream_id = opened["stream_id"]
            try:
                failure_stage = "successor_activation"
                successor = self.pcm_handoff.activate_successor()
                if (
                    successor.epoch_id != prepared_epoch_id
                    or successor.asr_job_id != old_fence.asr_job_id
                    or successor.speech_segment_id != old_fence.speech_segment_id
                    or successor.epoch_seq != old_fence.epoch_seq + 1
                ):
                    raise QwenEpochLifecycleError("invalid_successor_lineage")
                successor_pcm = self.pcm_handoff.snapshot()
                replay_start = successor_pcm.replay_start_cursor
                if replay_start is None:
                    raise QwenEpochLifecycleError("successor_replay_cursor_missing")
                candidate_binding.bind(successor)
                self._bound_fence = successor
                self._bound_stream_id = successor_stream_id
                self._canonical_base_cursor = replay_start
                self._observed_local_processed_cursor = 0
                replay_samples = max(
                    0,
                    successor_pcm.unique_primary_admitted_cursor - replay_start,
                )
                # The epoch-local view resets only after A6.1 admits this
                # canonical successor. Logical PCM totals remain cumulative.
                self._observability.start_epoch(
                    fence=successor,
                    stream_id=successor_stream_id,
                    stream_init_ms=self._opened_init_ms(opened),
                    state_init_ms=self._opened_state_init_ms(opened),
                    replay_audio_samples=replay_samples,
                    now=self._clock(),
                )
                self._observability.record_transition_stage("SUCCESSOR_ACTIVATED", now=self._clock())
            except Exception:
                self._bound_fence = None
                self._bound_stream_id = None
                await self.qwen_runtime.close_session(
                    connection_id=self.connection_id,
                    source=self.source,
                )
                raise

            failure_stage = "replay"

            def observe_stage(stage: str) -> None:
                nonlocal failure_stage
                self._observability.record_transition_stage(stage, now=self._clock())
                if stage.startswith("CATCHUP"):
                    failure_stage = "catchup"
                elif stage.startswith("REPLAY"):
                    failure_stage = "replay"

            result = await self.pcm_handoff.drain_handoff(
                self, stage_observer=observe_stage,
            )
            if result.completed:
                self._state = QwenEpochLifecycleState.ACTIVE
                self._sync_processed()
                self._complete_observed_transition(result)
            else:
                self._observability.record_transition_blocked(
                    failure_stage=result.failure_stage,
                    failure_reason=result.blocked_reason,
                    pcm_snapshot=self.pcm_handoff.snapshot(),
                )
                self._observe_operational_metrics()
            return result
        except BaseException as exc:
            # Retain a bounded timing/failure record while preserving the
            # existing lifecycle's fail-safe state and retry semantics.
            self._observability.record_transition_failure(
                failure_stage=failure_stage,
                failure_reason=self._error_code(exc),
            )
            if self._observability.snapshot(now=self._clock())["last_transition"].get("EPOCH_HANDOFF_WALL_MS") is None:
                self._observability.complete_transition(
                    success=False,
                    failure_reason=self._error_code(exc),
                    stream_snapshot=self._stream_snapshot(self._bound_stream_id),
                    global_snapshot=self._global_scheduler_snapshot(),
                    pcm_snapshot=self.pcm_handoff.snapshot(),
                    now=self._clock(),
                )
            raise

    def _complete_observed_transition(self, result: HandoffDrainResult) -> None:
        self._finish_replay_wall()
        self._observe_operational_metrics()
        self._observability.complete_transition(
            success=True,
            replay_admitted_samples=result.replay_admitted_samples,
            transition_primary_samples=result.primary_admitted_samples,
            stream_snapshot=self._stream_snapshot(self._bound_stream_id),
            global_snapshot=self._global_scheduler_snapshot(),
            pcm_snapshot=self.pcm_handoff.snapshot(),
            now=self._clock(),
        )
        self._observability.record_transition_stage("ACTIVE", now=self._clock())
        if self._soft_rollover is not None:
            self._soft_rollover.mark_rollover_succeeded(now=self._clock())

    def _observe_operational_metrics(self) -> None:
        self._observability.observe(
            stream_snapshot=self._stream_snapshot(self._bound_stream_id),
            global_snapshot=self._global_scheduler_snapshot(),
            pcm_snapshot=self.pcm_handoff.snapshot(),
            now=self._clock(),
        )

    def _stream_snapshot(self, stream_id: str | None) -> Mapping[str, Any] | None:
        if stream_id is None:
            return None
        scheduler = getattr(self.qwen_runtime, "scheduler", None)
        snapshot = getattr(scheduler, "stream_snapshot", None)
        if not callable(snapshot):
            return None
        try:
            value = snapshot(self.connection_id, stream_id)
        except Exception:
            return None
        return value if isinstance(value, Mapping) else None

    def _global_scheduler_snapshot(self) -> Mapping[str, Any] | None:
        scheduler = getattr(self.qwen_runtime, "scheduler", None)
        snapshot = getattr(scheduler, "snapshot", None)
        if not callable(snapshot):
            return None
        try:
            value = snapshot()
        except Exception:
            return None
        return value if isinstance(value, Mapping) else None

    def _soft_trigger_metrics(self) -> dict[str, Any]:
        stream = self._stream_snapshot(self._bound_stream_id) or {}
        global_metrics = self._global_scheduler_snapshot() or {}
        return {
            "epoch_audio_ms": stream.get("accepted_audio_ms"),
            "scheduler_backlog_ms": stream.get("scheduler_backlog_audio_ms"),
            "stream_lag_ms": stream.get("stream_lag_ms"),
            "scheduler_wait_p95_ms": stream.get("scheduler_wait_p95_ms"),
            "decode_wall_p95_ms": stream.get("decode_wall_p95_ms"),
            "global_backlog_ms": global_metrics.get("qwen_scheduler_backlog_ms"),
            "global_active_stream_count": global_metrics.get("active_stream_count"),
            "global_pending_decode_count": global_metrics.get("pending_decode_count"),
        }

    def _current_stream_audio_ms(self) -> float:
        stream = self._stream_snapshot(self._bound_stream_id) or {}
        value = stream.get("accepted_audio_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
        pcm = self.pcm_handoff.snapshot()
        return max(
            0,
            pcm.current_epoch_admitted_cursor - self._canonical_base_cursor,
        ) * 1000.0 / 16_000

    def _observe_candidate(self, fence: ExecutionFence, event: dict[str, Any]) -> None:
        if (
            event.get("event") != "partial_candidate"
            or not isinstance(event.get("text"), str)
            or not event["text"].strip()
            or fence != self._bound_fence
            or fence != self.epoch_controller.current
            or self._bound_stream_id is None
        ):
            return
        canonical_cursor = self._canonical_base_cursor + self._observed_local_processed_cursor
        self._observability.record_first_partial_after_rollover(
            canonical_cursor=canonical_cursor,
            now=self._clock(),
        )

    def _observe_stale_candidate(self, stream_id: str) -> None:
        if stream_id in self._seen_stream_ids:
            self._observability.record_stale_result_reject(
                gate="canonical_execution_fence_stale",
                stream_id=stream_id,
            )

    def _observe_local_stale_reject(self, stream_id: str, gate: str) -> None:
        if stream_id in self._seen_stream_ids:
            self._observability.record_stale_result_reject(
                gate=gate,
                stream_id=stream_id,
            )

    def _finish_replay_wall(self) -> None:
        if self._replay_started_at is None:
            return
        self._observability.record_replay_wall(
            replay_wall_ms=max(0.0, self._clock() - self._replay_started_at) * 1000.0,
        )
        self._replay_started_at = None

    @staticmethod
    def _opened_init_ms(opened: Mapping[str, Any]) -> float | None:
        value = opened.get("FIRST_STREAM_INIT_MS")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    @staticmethod
    def _opened_state_init_ms(opened: Mapping[str, Any]) -> float | None:
        value = opened.get("FIRST_STREAM_STATE_INIT_WALL_MS")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    @staticmethod
    def _error_code(exc: BaseException) -> str:
        code = getattr(exc, "code", None)
        if isinstance(code, str):
            return code[:120]
        return type(exc).__name__

    async def _open_fresh_qwen_state(
        self,
        *,
        forbidden_ids: set[str],
        candidate_fence: ExecutionFence | None = None,
        candidate_base: CandidateBase | None = None,
    ) -> tuple[dict[str, Any], _QwenCandidateFenceBinding]:
        candidate_binding = _QwenCandidateFenceBinding(
            epoch_controller=self.epoch_controller,
            candidate_stitcher=self._candidate_stitcher,
            base=candidate_base,
            candidate_observer=self._observe_candidate,
            stale_reject_observer=self._observe_stale_candidate,
        )
        if candidate_fence is not None:
            candidate_binding.bind(candidate_fence)
        opened = await self.qwen_runtime.open_session(
            connection_id=self.connection_id,
            source=self.source,
            language=self.language,
            context=self.context,
            chunk_size_ms=self.chunk_size_ms,
            request_id=self.request_id,
            event_sink=self.event_sink,
            candidate_result_filter=candidate_binding.filter_candidate,
            processed_audio_observer=self._observe_processed,
            local_stale_reject_observer=self._observe_local_stale_reject,
        )
        stream_id = opened.get("stream_id") if isinstance(opened, dict) else None
        if (
            not isinstance(stream_id, str)
            or not stream_id
            or stream_id in forbidden_ids
            or stream_id in self._seen_stream_ids
        ):
            await self.qwen_runtime.close_session(
                connection_id=self.connection_id,
                source=self.source,
            )
            raise QwenEpochLifecycleError("local_stream_id_not_fresh")
        self._seen_stream_ids.add(stream_id)
        candidate_binding.stream_id = stream_id
        return opened, candidate_binding

    def _observe_processed(self, stream_id: str, cursor_end_samples: int) -> None:
        """Record successful local Qwen progress without mutating A6.2."""
        if (
            not isinstance(cursor_end_samples, int)
            or isinstance(cursor_end_samples, bool)
            or cursor_end_samples < 0
            or stream_id != self._bound_stream_id
            or self._bound_fence is None
            or self._bound_fence != self.epoch_controller.current
        ):
            return
        self._observed_local_processed_cursor = max(
            self._observed_local_processed_cursor,
            cursor_end_samples,
        )

    def _sync_processed(self) -> int:
        """Flush observed progress only through A6.2's admitted cursor."""
        fence = self._bound_fence
        if (
            fence is None
            or self._bound_stream_id is None
            or fence != self.epoch_controller.current
            or self._state not in {
                QwenEpochLifecycleState.ACTIVE,
                QwenEpochLifecycleState.HANDOFF_PENDING,
            }
        ):
            return self.pcm_handoff.snapshot().processed_cursor

        snapshot = self.pcm_handoff.snapshot()
        target = min(
            self._canonical_base_cursor + self._observed_local_processed_cursor,
            snapshot.current_epoch_admitted_cursor,
        )
        if target > snapshot.processed_cursor:
            return self.pcm_handoff.mark_processed(
                fence=fence,
                start_sample=snapshot.processed_cursor,
                end_sample=target,
            )
        return snapshot.processed_cursor
