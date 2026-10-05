"""Bounded, content-free observability for local Qwen acoustic epochs."""
from __future__ import annotations

import math
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
    "retained_source_samples",
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
    """Keep only the active epoch, latest transition, and constant-size totals."""

    def __init__(self, *, monotonic: Callable[[], float]) -> None:
        self._clock = monotonic
        self._current_epoch: dict[str, Any] | None = None
        self._last_transition: dict[str, Any] | None = None
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
        self._current_epoch = {
            "epoch_id": fence.epoch_id,
            "epoch_seq": fence.epoch_seq,
            "asr_job_id": fence.asr_job_id,
            "speech_segment_id": fence.speech_segment_id,
            "local_stream_id": stream_id,
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
        }
        if self._last_transition is not None:
            self._last_transition["successor_epoch_id"] = fence.epoch_id
            self._last_transition["successor_epoch_seq"] = fence.epoch_seq

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
            epoch["global_scheduler"] = global_values
        if self._last_transition is not None:
            self._last_transition["latest_pcm_accounting"] = pcm_values
            self._last_transition["global_scheduler"] = global_values

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
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                safe_details[key] = value
            elif isinstance(value, (tuple, list)):
                safe_details[key] = [str(item)[:80] for item in value[:16]]
        pcm_values = _selected(pcm_snapshot, _PCM_METRICS)
        self._last_transition = {
            "category": category,
            "reason": reason[:120],
            "EPOCH_ROLLOVER_REASON": reason[:120],
            "details": safe_details,
            "state": "HANDOFF_PENDING",
            "started_at_monotonic": started_at,
            "cutover_cursor": _number(cutover_cursor),
            "EPOCH_HANDOFF_WALL_MS": None,
            "EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS": None,
            "replay_admitted_samples": 0,
            "transition_primary_samples": 0,
            "failure_reason": None,
            "handoff_started_pcm_accounting": pcm_values,
            "latest_pcm_accounting": pcm_values,
            "predecessor_scheduler": _selected(stream_snapshot, _STREAM_METRICS),
            "successor_epoch_id": None,
            "successor_epoch_seq": None,
            "predecessor_stream_id": predecessor_stream_id,
            "stale_result_rejects_start": self._cumulative["stale_result_rejects"],
            "stale_result_rejects_end": self._cumulative["stale_result_rejects"],
            "stale_result_rejects_delta": 0,
            "stale_result_reject_gate_counts": {gate: 0 for gate in _STALE_REJECT_GATES},
        }
        self._transition_started_at = started_at
        self._cumulative["rollover_attempt_count"] += 1

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
        transition["failure_reason"] = None if success else (failure_reason or "unspecified")[:120]
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
            current["epoch_age_ms"] = round(
                max(0.0, observed_at - current["started_at_monotonic"]) * 1000.0, 3,
            )
            current.pop("started_at_monotonic", None)
        transition = None if self._last_transition is None else deepcopy(self._last_transition)
        if transition is not None:
            transition.pop("started_at_monotonic", None)
            if transition["EPOCH_HANDOFF_WALL_MS"] is None and self._transition_started_at is not None:
                transition["handoff_elapsed_ms"] = round(
                    max(0.0, observed_at - self._transition_started_at) * 1000.0, 3,
                )
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
            "current_epoch": current,
            "last_transition": transition,
            "logical_cumulative": logical,
        }

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
