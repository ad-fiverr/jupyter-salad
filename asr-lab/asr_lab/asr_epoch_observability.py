"""Bounded, content-free observability for local Qwen acoustic epochs."""
from __future__ import annotations

import math
import re
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Mapping


_STREAM_METRICS = (
    "accepted_audio_ms",
    "scheduler_pending_jobs",
    "scheduler_backlog_audio_ms",
    "scheduler_max_backlog_audio_ms",
    "stream_lag_ms",
    "stream_lag_max_ms",
    "scheduler_wait_p50_ms",
    "scheduler_wait_p95_ms",
    "decode_wall_p50_ms",
    "decode_wall_p95_ms",
    "decode_steps_delta_total",
)
_GLOBAL_METRICS = (
    "active_stream_count",
    "pending_decode_count",
    "qwen_scheduler_backlog_ms",
    "qwen_scheduler_max_stream_backlog_ms",
    "qwen_scheduler_stream_lag_ms",
    "qwen_scheduler_max_stream_lag_ms",
    "qwen_scheduler_wait_p50_ms",
    "qwen_scheduler_wait_p95_ms",
    "qwen_decode_wall_p50_ms",
    "qwen_decode_wall_p95_ms",
)
_PCM_METRICS = (
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
)
_DETAIL_KEYS = frozenset({
    "trigger_signals",
    "qualifying_samples",
    "policy_window_samples",
    "observed_audio_ms",
    "hard_limit_ms",
    "scheduler_overrun_reason",
    "active_signals",
    "ignored_capacity_signals",
})
_STALE_REJECT_GATES = {
    "canonical_execution_fence_stale": "canonical_fence_rejects",
    "scheduler_request_key_stale": "local_request_freshness_rejects",
    "runtime_scheduler_key_stale": "local_request_freshness_rejects",
    "runtime_session_key_stale": "local_request_freshness_rejects",
}
_EPOCH_HISTORY_LIMIT = 8
_TRANSITION_HISTORY_LIMIT = 8
_TRANSITION_STAGE_EVENT_LIMIT = 16
_TRANSITION_STAGES = frozenset({
    "HANDOFF_STARTED", "PREDECESSOR_FENCED", "SUCCESSOR_OPENING",
    "SUCCESSOR_OPENED", "SUCCESSOR_ACTIVATED", "REPLAY_DRAINING",
    "REPLAY_DRAINED", "CATCHUP_DRAINING", "CATCHUP_DRAINED", "ACTIVE",
})
_FAILURE_STAGES = frozenset({
    "predecessor_fence", "successor_open", "successor_activation",
    "replay", "catchup", "transition_capacity", "handoff",
})


def _number(value: Any) -> int | float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return value


def _selected(snapshot: Mapping[str, Any] | None, keys: tuple[str, ...]) -> dict[str, int | float | None]:
    source = snapshot or {}
    if isinstance(source, Mapping):
        return {key: _number(source.get(key)) for key in keys}
    return {key: _number(getattr(source, key, None)) for key in keys}


class ASREpochObservability:
    """Keep bounded epoch/transition history and constant-size logical totals."""

    def __init__(self, *, monotonic: Callable[[], float]) -> None:
        self._clock = monotonic
        self._current_epoch: dict[str, Any] | None = None
        self._last_transition: dict[str, Any] | None = None
        self._epoch_history: deque[dict[str, Any]] = deque(maxlen=_EPOCH_HISTORY_LIMIT)
        self._transition_history: deque[dict[str, Any]] = deque(maxlen=_TRANSITION_HISTORY_LIMIT)
        self._transition_started_at: float | None = None
        self._cumulative: dict[str, int | float] = {
            "rollover_attempt_count": 0,
            "rollover_success_count": 0,
            "rollover_failure_count": 0,
            "rollover_handoff_wall_total_ms": 0.0,
            "source_received_samples": 0,
            "unique_primary_samples": 0,
            "replay_admitted_samples": 0,
            "downstream_rejected_samples": 0,
            "explicit_source_rejected_samples": 0,
            "stale_result_rejects": 0,
            "canonical_fence_rejects": 0,
            "local_request_freshness_rejects": 0,
        }
        self._stale_gate_counts = {gate: 0 for gate in _STALE_REJECT_GATES}

    def start_epoch(
        self,
        *,
        fence: Any,
        stream_id: str,
        stream_init_ms: float | None,
        state_init_ms: float | None = None,
        replay_audio_samples: int = 0,
        now: float | None = None,
    ) -> None:
        started_at = self._clock() if now is None else now
        if not math.isfinite(started_at):
            raise ValueError("invalid_epoch_observability_clock")
        if self._current_epoch is not None:
            previous = self._current_epoch
            previous["lifecycle_state"] = "ROLLED_OVER"
            previous["ended_at_monotonic"] = started_at
            previous["epoch_duration_ms"] = round(
                max(0.0, started_at - previous["started_at_monotonic"]) * 1000.0, 3,
            )
            if self._last_transition is not None:
                previous["end_transition_seq"] = self._last_transition.get("transition_seq")
                previous["end_reason"] = self._last_transition.get("reason")
        self._current_epoch = {
            "epoch_id": fence.epoch_id,
            "epoch_seq": fence.epoch_seq,
            "asr_job_id": fence.asr_job_id,
            "speech_segment_id": fence.speech_segment_id,
            "local_stream_id": stream_id,
            "lifecycle_state": "ACTIVE",
            "started_at_monotonic": started_at,
            "epoch_age_ms": 0.0,
            "epoch_audio_ms": 0.0,
            "stream_init_ms": _number(stream_init_ms),
            "state_init_ms": _number(state_init_ms),
            "replay_audio_ms": max(0.0, replay_audio_samples * 1000.0 / 16_000),
            "replay_wall_ms": 0.0 if replay_audio_samples == 0 else None,
            "scheduler_pending_jobs": 0,
            "scheduler_backlog_audio_ms": 0.0,
            "scheduler_max_backlog_audio_ms": 0.0,
            "stream_lag_ms": 0.0,
            "stream_lag_max_ms": 0.0,
            "scheduler_wait_p50_ms": None,
            "scheduler_wait_p95_ms": None,
            "decode_wall_p50_ms": None,
            "decode_wall_p95_ms": None,
            "decode_steps_delta_total": 0,
            "stale_result_rejects": 0,
            "canonical_fence_rejects": 0,
            "local_request_freshness_rejects": 0,
            "stale_result_reject_gates": {gate: 0 for gate in _STALE_REJECT_GATES},
            "pcm": _selected(None, _PCM_METRICS),
            "global_scheduler": _selected(None, _GLOBAL_METRICS),
            # qwen-asr 0.0.6 does not expose a version-stable, read-only
            # accumulated-audio duration contract in this runtime.
            "QWEN_AUDIO_ACCUM_MS": None,
        }
        self._epoch_history.append(self._current_epoch)
        if self._last_transition is not None:
            self._last_transition["successor_epoch_id"] = fence.epoch_id
            self._last_transition["successor_epoch_seq"] = fence.epoch_seq
            self._last_transition["successor_local_stream_id"] = stream_id

    def observe(
        self,
        *,
        stream_snapshot: Mapping[str, Any] | None,
        global_snapshot: Mapping[str, Any] | None,
        pcm_snapshot: Any,
        now: float | None = None,
    ) -> None:
        observed_at = self._clock() if now is None else now
        if not math.isfinite(observed_at):
            raise ValueError("invalid_epoch_observability_clock")
        stream_values = _selected(stream_snapshot, _STREAM_METRICS)
        global_values = _selected(global_snapshot, _GLOBAL_METRICS)
        pcm_values = _selected(pcm_snapshot, _PCM_METRICS)
        self._observe_cumulative(pcm_snapshot)
        if self._current_epoch is not None:
            epoch = self._current_epoch
            epoch["epoch_age_ms"] = round(
                max(0.0, observed_at - epoch["started_at_monotonic"]) * 1000.0, 3,
            )
            for key, value in stream_values.items():
                if value is not None:
                    target_key = "epoch_audio_ms" if key == "accepted_audio_ms" else key
                    if target_key in {"scheduler_max_backlog_audio_ms", "stream_lag_max_ms"}:
                        epoch[target_key] = max(float(epoch.get(target_key) or 0), float(value))
                    else:
                        epoch[target_key] = value
            epoch["pcm"] = pcm_values
            epoch.update(self._pcm_occupancy_ms(pcm_values))
            epoch["global_scheduler"] = global_values
        if self._last_transition is not None:
            self._last_transition["latest_pcm_accounting"] = pcm_values
            self._last_transition["global_scheduler"] = global_values
            self._update_transition_pcm_totals(self._last_transition, pcm_snapshot)

    def record_stale_result_reject(self, *, gate: str, stream_id: str) -> None:
        source_counter = _STALE_REJECT_GATES.get(gate)
        if source_counter is None or not isinstance(stream_id, str) or not stream_id:
            raise ValueError("invalid_stale_result_reject")
        self._cumulative[source_counter] += 1
        self._cumulative["stale_result_rejects"] = (
            self._cumulative["canonical_fence_rejects"]
            + self._cumulative["local_request_freshness_rejects"]
        )
        self._stale_gate_counts[gate] += 1
        if self._current_epoch is not None and self._current_epoch["local_stream_id"] == stream_id:
            self._current_epoch[source_counter] += 1
            self._current_epoch["stale_result_rejects"] = (
                self._current_epoch["canonical_fence_rejects"]
                + self._current_epoch["local_request_freshness_rejects"]
            )
            self._current_epoch["stale_result_reject_gates"][gate] += 1
        transition = self._last_transition
        transition_stream = transition is not None and (
            transition["state"] == "HANDOFF_PENDING"
            or (
                transition["state"] in {"ACTIVE", "FAILED_OR_INCOMPLETE"}
                and transition.get("predecessor_stream_id") == stream_id
            )
        )
        if transition is not None and transition_stream:
            transition["stale_result_rejects_end"] = self._cumulative["stale_result_rejects"]
            transition["stale_result_rejects_delta"] = (
                transition["stale_result_rejects_end"]
                - transition["stale_result_rejects_start"]
            )
            transition["stale_result_reject_gate_counts"][gate] += 1

    def record_replay_wall(self, *, replay_wall_ms: float) -> None:
        if not math.isfinite(replay_wall_ms) or replay_wall_ms < 0:
            raise ValueError("invalid_epoch_replay_wall")
        if self._current_epoch is not None and self._current_epoch["replay_wall_ms"] is None:
            self._current_epoch["replay_wall_ms"] = round(replay_wall_ms, 3)

    def end_epoch(self, *, end_state: str, now: float | None = None) -> None:
        if end_state != "NATURAL_EOS" or self._current_epoch is None:
            return
        ended_at = self._clock() if now is None else now
        if not math.isfinite(ended_at):
            raise ValueError("invalid_epoch_observability_clock")
        epoch = self._current_epoch
        epoch["lifecycle_state"] = end_state
        epoch["ended_at_monotonic"] = ended_at
        epoch["epoch_duration_ms"] = round(
            max(0.0, ended_at - epoch["started_at_monotonic"]) * 1000.0, 3,
        )

    def begin_transition(
        self,
        *,
        category: str,
        reason: str,
        details: Mapping[str, Any],
        cutover_cursor: int | None,
        predecessor_stream_id: str | None = None,
        stream_snapshot: Mapping[str, Any] | None,
        pcm_snapshot: Any,
        now: float | None = None,
    ) -> None:
        started_at = self._clock() if now is None else now
        if not math.isfinite(started_at):
            raise ValueError("invalid_epoch_observability_clock")
        safe_details: dict[str, Any] = {}
        for key in _DETAIL_KEYS:
            value = details.get(key)
            if isinstance(value, str):
                safe_details[key] = self._safe_reason(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                safe_details[key] = _number(value)
            elif isinstance(value, (tuple, list)):
                safe_details[key] = [
                    self._safe_reason(item) for item in value[:16] if isinstance(item, str)
                ]
        pcm_values = _selected(pcm_snapshot, _PCM_METRICS)
        predecessor = self._current_epoch or {}
        safe_category = self._safe_reason(category)
        safe_transition_reason = self._safe_reason(reason)
        self._last_transition = {
            "schema_version": 1,
            "transition_seq": int(self._cumulative["rollover_attempt_count"]) + 1,
            "category": safe_category,
            "reason": safe_transition_reason,
            "EPOCH_ROLLOVER_REASON": safe_transition_reason,
            "details": safe_details,
            "state": "HANDOFF_PENDING",
            "stage": "HANDOFF_STARTED",
            "last_completed_stage": None,
            "stage_events": [],
            "handoff_attempt_count": 0,
            "started_at_monotonic": started_at,
            "rollover_started_at_monotonic": started_at,
            "successor_open_started_at_monotonic": None,
            "successor_open_completed_at_monotonic": None,
            "successor_activated_at_monotonic": None,
            "replay_completed_at_monotonic": None,
            "catchup_completed_at_monotonic": None,
            "handoff_completed_at_monotonic": None,
            "cutover_cursor": _number(cutover_cursor),
            "EPOCH_HANDOFF_WALL_MS": None,
            "EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS": None,
            "replay_admitted_samples": 0,
            "transition_primary_samples": 0,
            "failure_stage": None,
            "failure_reason": None,
            "last_blocked_stage": None,
            "last_blocked_reason": None,
            "PCM_LOST": None,
            "PRIMARY_DUP": None,
            "handoff_started_pcm_accounting": pcm_values,
            "latest_pcm_accounting": pcm_values,
            "predecessor_scheduler": _selected(stream_snapshot, _STREAM_METRICS),
            "predecessor_epoch_id": predecessor.get("epoch_id"),
            "predecessor_epoch_seq": predecessor.get("epoch_seq"),
            "predecessor_local_stream_id": predecessor.get("local_stream_id", predecessor_stream_id),
            "successor_epoch_id": None,
            "successor_epoch_seq": None,
            "predecessor_stream_id": predecessor_stream_id,
            "stale_result_rejects_start": self._cumulative["stale_result_rejects"],
            "stale_result_rejects_end": self._cumulative["stale_result_rejects"],
            "stale_result_rejects_delta": 0,
            "stale_result_reject_gate_counts": {gate: 0 for gate in _STALE_REJECT_GATES},
        }
        self._transition_history.append(self._last_transition)
        self._transition_started_at = started_at
        self._cumulative["rollover_attempt_count"] += 1
        if self._current_epoch is not None:
            self._current_epoch["lifecycle_state"] = "ROLLOVER_STARTED"
        self.record_transition_stage("HANDOFF_STARTED", now=started_at)

    def record_transition_stage(self, stage: str, *, now: float | None = None) -> None:
        transition = self._last_transition
        if transition is None or stage not in _TRANSITION_STAGES:
            return
        observed_at = self._clock() if now is None else now
        if not math.isfinite(observed_at):
            raise ValueError("invalid_epoch_observability_clock")
        started_at = self._transition_started_at
        elapsed_ms = (
            round(max(0.0, observed_at - started_at) * 1000.0, 3)
            if started_at is not None else 0.0
        )
        if stage == "REPLAY_DRAINING":
            transition["handoff_attempt_count"] += 1
        events = transition["stage_events"]
        events.append({
            "stage": stage,
            "elapsed_ms": elapsed_ms,
            "monotonic_at": round(observed_at, 6),
        })
        if len(events) > _TRANSITION_STAGE_EVENT_LIMIT:
            del events[:len(events) - _TRANSITION_STAGE_EVENT_LIMIT]
        transition["stage"] = stage
        if stage in {
            "PREDECESSOR_FENCED", "SUCCESSOR_OPENED", "SUCCESSOR_ACTIVATED",
            "REPLAY_DRAINED", "CATCHUP_DRAINED", "ACTIVE",
        }:
            transition["last_completed_stage"] = stage
        timestamp_fields = {
            "HANDOFF_STARTED": "rollover_started_at_monotonic",
            "SUCCESSOR_OPENING": "successor_open_started_at_monotonic",
            "SUCCESSOR_OPENED": "successor_open_completed_at_monotonic",
            "SUCCESSOR_ACTIVATED": "successor_activated_at_monotonic",
            "REPLAY_DRAINED": "replay_completed_at_monotonic",
            "CATCHUP_DRAINED": "catchup_completed_at_monotonic",
            "ACTIVE": "handoff_completed_at_monotonic",
        }
        timestamp_field = timestamp_fields.get(stage)
        if timestamp_field is not None:
            transition[timestamp_field] = observed_at
        if stage == "PREDECESSOR_FENCED" and self._current_epoch is not None:
            self._current_epoch["lifecycle_state"] = "FENCED"
        if stage == "ACTIVE" and self._current_epoch is not None:
            self._current_epoch["lifecycle_state"] = "ACTIVE"

    def record_transition_blocked(
        self,
        *,
        failure_stage: str | None,
        failure_reason: str | None,
        pcm_snapshot: Any,
    ) -> None:
        transition = self._last_transition
        if transition is None:
            return
        stage = failure_stage if failure_stage in _FAILURE_STAGES else "handoff"
        reason = self._safe_reason(failure_reason)
        transition["failure_stage"] = stage
        transition["failure_reason"] = reason
        transition["last_blocked_stage"] = stage
        transition["last_blocked_reason"] = reason
        self._update_transition_pcm_totals(transition, pcm_snapshot)

    def record_transition_failure(self, *, failure_stage: str, failure_reason: str) -> None:
        transition = self._last_transition
        if transition is None:
            return
        stage = failure_stage if failure_stage in _FAILURE_STAGES else "handoff"
        transition["failure_stage"] = stage
        transition["failure_reason"] = self._safe_reason(failure_reason)
        if self._current_epoch is not None:
            self._current_epoch["lifecycle_state"] = "HANDOFF_FAILED"

    def record_first_partial_after_rollover(
        self, *, canonical_cursor: int, now: float | None = None,
    ) -> float | None:
        transition = self._last_transition
        started_at = self._transition_started_at
        if (
            transition is None
            or started_at is None
            or transition["state"] not in {"HANDOFF_PENDING", "ACTIVE"}
        ):
            return None
        cutover = transition.get("cutover_cursor")
        if (
            not isinstance(canonical_cursor, int)
            or isinstance(canonical_cursor, bool)
            or not isinstance(cutover, int)
            or canonical_cursor <= cutover
        ):
            return None
        if transition["EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS"] is not None:
            return None
        observed_at = self._clock() if now is None else now
        value = round(max(0.0, observed_at - started_at) * 1000.0, 3)
        transition["EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS"] = value
        return value

    def complete_transition(
        self,
        *,
        success: bool,
        replay_admitted_samples: int = 0,
        transition_primary_samples: int = 0,
        failure_reason: str | None = None,
        stream_snapshot: Mapping[str, Any] | None = None,
        global_snapshot: Mapping[str, Any] | None = None,
        pcm_snapshot: Any,
        now: float | None = None,
    ) -> float | None:
        transition = self._last_transition
        started_at = self._transition_started_at
        if transition is None or started_at is None:
            return None
        completed_at = self._clock() if now is None else now
        if not math.isfinite(completed_at):
            raise ValueError("invalid_epoch_observability_clock")
        handoff_ms = round(max(0.0, completed_at - started_at) * 1000.0, 3)
        transition["EPOCH_HANDOFF_WALL_MS"] = handoff_ms
        transition["state"] = "ACTIVE" if success else "FAILED_OR_INCOMPLETE"
        if success:
            transition["failure_stage"] = None
            transition["failure_reason"] = None
        else:
            transition["failure_reason"] = self._safe_reason(failure_reason)
        transition["replay_admitted_samples"] = max(0, int(replay_admitted_samples))
        transition["transition_primary_samples"] = max(0, int(transition_primary_samples))
        transition["successor_scheduler"] = _selected(stream_snapshot, _STREAM_METRICS)
        transition["global_scheduler"] = _selected(global_snapshot, _GLOBAL_METRICS)
        transition["completed_pcm_accounting"] = _selected(pcm_snapshot, _PCM_METRICS)
        transition["stale_result_rejects_end"] = self._cumulative["stale_result_rejects"]
        transition["stale_result_rejects_delta"] = max(
            0,
            transition["stale_result_rejects_end"] - transition["stale_result_rejects_start"],
        )
        transition["latest_pcm_accounting"] = transition["completed_pcm_accounting"]
        self._update_transition_pcm_totals(transition, pcm_snapshot)
        continuity = self._pcm_continuity(pcm_snapshot)
        transition["PCM_LOST"] = continuity["PCM_LOST"]
        transition["PRIMARY_DUP"] = continuity["PRIMARY_DUP"]
        if success and self._current_epoch is not None:
            self._current_epoch["lifecycle_state"] = "ACTIVE"
        self._cumulative["rollover_handoff_wall_total_ms"] += handoff_ms
        if success:
            self._cumulative["rollover_success_count"] += 1
        else:
            self._cumulative["rollover_failure_count"] += 1
        self._observe_cumulative(pcm_snapshot)
        return handoff_ms

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        observed_at = self._clock() if now is None else now
        current = None if self._current_epoch is None else deepcopy(self._current_epoch)
        if current is not None:
            age_end = current.get("ended_at_monotonic", observed_at)
            current["epoch_age_ms"] = round(
                max(0.0, age_end - current["started_at_monotonic"]) * 1000.0, 3,
            )
            current.pop("started_at_monotonic", None)
            current.pop("ended_at_monotonic", None)
            current["QWEN_AUDIO_ACCUM_MS"] = None
        transition = None if self._last_transition is None else deepcopy(self._last_transition)
        if transition is not None:
            self._prepare_transition_snapshot(transition, observed_at)
        epoch_history = [
            self._prepare_epoch_snapshot(deepcopy(epoch), observed_at)
            for epoch in self._epoch_history
        ]
        transition_history = [
            self._prepare_transition_snapshot(deepcopy(item), observed_at)
            for item in self._transition_history
        ]
        if current is not None:
            current.update({
                "CURRENT_EPOCH_ID": current["epoch_id"],
                "CURRENT_EPOCH_SEQ": current["epoch_seq"],
                "EPOCH_AUDIO_MS": current["epoch_audio_ms"],
                "EPOCH_AGE_MS": current["epoch_age_ms"],
                "EPOCH_STATE_INIT_MS": current["state_init_ms"],
                "EPOCH_REPLAY_AUDIO_MS": current["replay_audio_ms"],
                "EPOCH_REPLAY_WALL_MS": current["replay_wall_ms"],
                "EPOCH_DECODE_WALL_P50_MS": current["decode_wall_p50_ms"],
                "EPOCH_DECODE_WALL_P95_MS": current["decode_wall_p95_ms"],
                "EPOCH_SCHEDULER_WAIT_P50_MS": current["scheduler_wait_p50_ms"],
                "EPOCH_SCHEDULER_WAIT_P95_MS": current["scheduler_wait_p95_ms"],
                "STALE_RESULT_REJECTS": current["stale_result_rejects"],
            })
        logical = dict(self._cumulative)
        logical["EPOCH_ROLLOVER_COUNT"] = self._cumulative["rollover_success_count"]
        logical["EPOCH_ROLLOVER_TOTAL_MS"] = self._cumulative["rollover_handoff_wall_total_ms"]
        logical["stale_result_reject_gates"] = dict(self._stale_gate_counts)
        return {
            "schema_version": 1,
            "current_epoch": current,
            "last_transition": transition,
            "epoch_history": epoch_history,
            "transition_history": transition_history,
            "logical_cumulative": logical,
            "QWEN_AUDIO_ACCUM_MS": None,
        }

    @staticmethod
    def _prepare_epoch_snapshot(epoch: dict[str, Any], observed_at: float) -> dict[str, Any]:
        started_at = epoch.pop("started_at_monotonic", None)
        ended_at = epoch.pop("ended_at_monotonic", None)
        if isinstance(ended_at, (int, float)):
            epoch_start = ended_at if started_at is None else started_at
            epoch["epoch_age_ms"] = round(
                max(0.0, ended_at - float(epoch_start)) * 1000.0, 3,
            )
        elif isinstance(started_at, (int, float)):
            epoch["epoch_age_ms"] = round(
                max(0.0, observed_at - float(started_at)) * 1000.0, 3,
            )
        epoch["QWEN_AUDIO_ACCUM_MS"] = None
        return epoch

    def _prepare_transition_snapshot(self, transition: dict[str, Any], observed_at: float) -> dict[str, Any]:
        transition.pop("started_at_monotonic", None)
        if (
            transition.get("EPOCH_HANDOFF_WALL_MS") is None
            and transition.get("transition_seq") == (self._last_transition or {}).get("transition_seq")
            and self._transition_started_at is not None
        ):
            transition["handoff_elapsed_ms"] = round(
                max(0.0, observed_at - self._transition_started_at) * 1000.0, 3,
            )
        transition["QWEN_AUDIO_ACCUM_MS"] = None
        return transition

    @staticmethod
    def _safe_reason(value: str | None) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", value):
            return "unspecified"
        return value

    @staticmethod
    def _update_transition_pcm_totals(transition: dict[str, Any], pcm_snapshot: Any) -> None:
        current = _selected(pcm_snapshot, _PCM_METRICS)
        started = transition.get("handoff_started_pcm_accounting") or {}
        for key, target in (
            ("replay_admitted_samples", "replay_admitted_samples"),
            ("unique_primary_admitted_samples", "transition_primary_samples"),
        ):
            end = current.get(key)
            begin = started.get(key)
            if isinstance(end, (int, float)) and isinstance(begin, (int, float)):
                transition[target] = max(0, int(end) - int(begin))
        transition.update(ASREpochObservability._pcm_occupancy_ms(current))

    @staticmethod
    def _pcm_occupancy_ms(values: Mapping[str, Any]) -> dict[str, float | None]:
        def to_ms(key: str) -> float | None:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                return None
            return round(float(value) * 1000.0 / 16_000.0, 3)

        return {
            "transition_buffer_current_ms": to_ms("transition_queued_samples"),
            "transition_buffer_max_ms": to_ms("max_transition_queued_samples"),
            "retained_pcm_current_ms": to_ms("retained_source_samples"),
            "retained_pcm_max_ms": to_ms("max_retained_source_samples"),
        }

    @staticmethod
    def _pcm_continuity(pcm_snapshot: Any) -> dict[str, int | None]:
        values = _selected(pcm_snapshot, (
            "received_samples", "unique_primary_admitted_samples",
            "transition_queued_samples", "explicit_source_rejected_samples",
        ))
        received = values["received_samples"]
        primary = values["unique_primary_admitted_samples"]
        queued = values["transition_queued_samples"]
        rejected = values["explicit_source_rejected_samples"]
        if not all(isinstance(value, (int, float)) for value in (received, primary, queued, rejected)):
            return {"PCM_LOST": None, "PRIMARY_DUP": None}
        if received == primary and queued == 0 and rejected == 0:
            return {"PCM_LOST": 0, "PRIMARY_DUP": 0}
        return {"PCM_LOST": None, "PRIMARY_DUP": None}

    def _observe_cumulative(self, pcm_snapshot: Any) -> None:
        if pcm_snapshot is None:
            return
        for source, target in (
            ("received_samples", "source_received_samples"),
            ("unique_primary_admitted_samples", "unique_primary_samples"),
            ("replay_admitted_samples", "replay_admitted_samples"),
            ("downstream_admission_rejected_samples", "downstream_rejected_samples"),
            ("explicit_source_rejected_samples", "explicit_source_rejected_samples"),
        ):
            value = _number(getattr(pcm_snapshot, source, None) if not isinstance(pcm_snapshot, Mapping) else pcm_snapshot.get(source))
            if value is not None:
                self._cumulative[target] = int(value)
