"""Deterministic, bounded local trigger policy for acoustic epoch rollover.

This policy is deliberately uncalibrated by default: production code must
inject a complete policy explicitly. Device/global-capacity signals can be
observed, but never participate in a soft trigger.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RolloverCategory(str, Enum):
    MANUAL_TEST = "manual_test"
    PROACTIVE_SOFT = "proactive_soft"
    HARD_BOUND = "hard_bound"
    EMERGENCY_BACKPRESSURE = "emergency_backpressure"


class SignalClass(str, Enum):
    ACOUSTIC_HISTORY = "acoustic_history"
    LOCAL_THROUGHPUT = "local_throughput"
    DEVICE_GLOBAL_CAPACITY = "device_global_capacity"


_ALLOWED_SIGNAL_CLASSES = {
    "epoch_audio_ms": SignalClass.ACOUSTIC_HISTORY,
    "scheduler_backlog_ms": SignalClass.LOCAL_THROUGHPUT,
    "stream_lag_ms": SignalClass.LOCAL_THROUGHPUT,
    "scheduler_wait_p95_ms": SignalClass.LOCAL_THROUGHPUT,
    "decode_wall_p95_ms": SignalClass.LOCAL_THROUGHPUT,
    "global_backlog_ms": SignalClass.DEVICE_GLOBAL_CAPACITY,
    "global_active_stream_count": SignalClass.DEVICE_GLOBAL_CAPACITY,
    "global_pending_decode_count": SignalClass.DEVICE_GLOBAL_CAPACITY,
}
MAX_POLICY_WINDOW_SAMPLES = 4096


@dataclass(frozen=True)
class SoftSignalDefinition:
    name: str
    signal_class: SignalClass
    enter_threshold: float
    clear_threshold: float

    def __post_init__(self) -> None:
        if self.name not in _ALLOWED_SIGNAL_CLASSES:
            raise ValueError("unknown_soft_rollover_signal")
        if self.signal_class is not _ALLOWED_SIGNAL_CLASSES[self.name]:
            raise ValueError("soft_rollover_signal_class_mismatch")
        if (
            not math.isfinite(self.enter_threshold)
            or not math.isfinite(self.clear_threshold)
            or self.clear_threshold < 0
            or self.enter_threshold <= self.clear_threshold
        ):
            raise ValueError("invalid_soft_rollover_hysteresis")

    @property
    def triggerable(self) -> bool:
        return self.signal_class in {
            SignalClass.ACOUSTIC_HISTORY,
            SignalClass.LOCAL_THROUGHPUT,
        }


@dataclass(frozen=True)
class SoftRolloverPolicy:
    soft_enabled: bool
    window_samples: int
    required_qualifying_samples: int
    min_active_signals: int
    cooldown_ms: float
    signals: tuple[SoftSignalDefinition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.soft_enabled, bool):
            raise ValueError("soft_rollover_enabled_must_be_boolean")
        if (
            isinstance(self.window_samples, bool)
            or not isinstance(self.window_samples, int)
            or not 1 <= self.window_samples <= MAX_POLICY_WINDOW_SAMPLES
        ):
            raise ValueError("soft_rollover_window_must_be_finite_and_bounded")
        if (
            isinstance(self.required_qualifying_samples, bool)
            or not isinstance(self.required_qualifying_samples, int)
            or not 2 <= self.required_qualifying_samples <= self.window_samples
        ):
            raise ValueError("invalid_soft_rollover_qualifying_sample_count")
        if (
            isinstance(self.min_active_signals, bool)
            or not isinstance(self.min_active_signals, int)
            or self.min_active_signals < 2
        ):
            raise ValueError("soft_rollover_requires_multiple_signals")
        if not math.isfinite(self.cooldown_ms) or self.cooldown_ms < 0:
            raise ValueError("invalid_soft_rollover_cooldown")
        names = [signal.name for signal in self.signals]
        if len(names) != len(set(names)):
            raise ValueError("duplicate_soft_rollover_signal")
        triggerable_count = sum(signal.triggerable for signal in self.signals)
        if self.soft_enabled and triggerable_count < self.min_active_signals:
            raise ValueError("insufficient_triggerable_soft_rollover_signals")

    @classmethod
    def from_optional_config(
        cls,
        *,
        soft_enabled: bool | None,
        window_samples: int | None,
        required_qualifying_samples: int | None,
        min_active_signals: int | None,
        cooldown_ms: float | None,
        signals: tuple[SoftSignalDefinition, ...] | None,
    ) -> "SoftRolloverPolicy | None":
        """Return no policy for absent/incomplete config; never invent defaults."""
        values = (
            soft_enabled,
            window_samples,
            required_qualifying_samples,
            min_active_signals,
            cooldown_ms,
            signals,
        )
        if any(value is None for value in values):
            return None
        return cls(
            soft_enabled=soft_enabled,
            window_samples=window_samples,
            required_qualifying_samples=required_qualifying_samples,
            min_active_signals=min_active_signals,
            cooldown_ms=cooldown_ms,
            signals=signals,
        )


@dataclass(frozen=True)
class SoftPolicyDecision:
    should_rollover: bool
    qualifying: bool
    qualifying_samples: int
    active_signals: tuple[str, ...]
    ignored_capacity_signals: tuple[str, ...]
    cooldown_active: bool


class SoftRolloverEvaluator:
    """Evaluate bounded policy samples with hysteresis and explicit cooldown."""

    def __init__(self, policy: SoftRolloverPolicy) -> None:
        self.policy = policy
        self._samples: deque[bool] = deque(maxlen=policy.window_samples)
        self._active: dict[str, bool] = {signal.name: False for signal in policy.signals}
        self._cooldown_until = 0.0

    def observe(self, metrics: Mapping[str, Any], *, now: float) -> SoftPolicyDecision:
        if not math.isfinite(now):
            raise ValueError("invalid_soft_rollover_clock")
        ignored_capacity: list[str] = []
        active: list[str] = []
        if self.policy.soft_enabled:
            for signal in self.policy.signals:
                value = metrics.get(signal.name)
                if not signal.triggerable:
                    if self._valid_metric(value) and value >= signal.enter_threshold:
                        ignored_capacity.append(signal.name)
                    self._active[signal.name] = False
                    continue
                if not self._valid_metric(value):
                    # Missing/invalid evidence fails closed instead of allowing
                    # a stale active signal to qualify a later sample.
                    self._active[signal.name] = False
                elif value >= signal.enter_threshold:
                    self._active[signal.name] = True
                elif value <= signal.clear_threshold:
                    self._active[signal.name] = False
                if self._active[signal.name]:
                    active.append(signal.name)

        qualifying = self.policy.soft_enabled and len(active) >= self.policy.min_active_signals
        self._samples.append(bool(qualifying))
        cooldown_active = now < self._cooldown_until
        return SoftPolicyDecision(
            should_rollover=(
                qualifying
                and sum(self._samples) >= self.policy.required_qualifying_samples
                and not cooldown_active
            ),
            qualifying=bool(qualifying),
            qualifying_samples=sum(self._samples),
            active_signals=tuple(sorted(active)),
            ignored_capacity_signals=tuple(sorted(ignored_capacity)),
            cooldown_active=cooldown_active,
        )

    def mark_rollover_succeeded(self, *, now: float) -> None:
        """Reset history and start cooldown only after successor becomes ACTIVE."""
        if not math.isfinite(now):
            raise ValueError("invalid_soft_rollover_clock")
        self._samples.clear()
        for name in self._active:
            self._active[name] = False
        self._cooldown_until = now + self.policy.cooldown_ms / 1000.0

    def snapshot(self, *, now: float) -> dict[str, Any]:
        return {
            "soft_enabled": self.policy.soft_enabled,
            "policy_window_samples": self.policy.window_samples,
            "policy_samples_retained": len(self._samples),
            "qualifying_samples": sum(self._samples),
            "active_signals": sorted(name for name, active in self._active.items() if active),
            "cooldown_remaining_ms": round(max(0.0, self._cooldown_until - now) * 1000.0, 3),
            "triggerable_signals": sorted(
                signal.name for signal in self.policy.signals if signal.triggerable
            ),
            "non_triggering_capacity_signals": sorted(
                signal.name for signal in self.policy.signals if not signal.triggerable
            ),
        }

    @staticmethod
    def _valid_metric(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )


def choose_rollover_category(*categories: RolloverCategory) -> RolloverCategory:
    """Deterministic cause precedence when more than one cause is observable."""
    if not categories:
        raise ValueError("rollover_category_required")
    priority = {
        RolloverCategory.EMERGENCY_BACKPRESSURE: 4,
        RolloverCategory.HARD_BOUND: 3,
        RolloverCategory.MANUAL_TEST: 2,
        RolloverCategory.PROACTIVE_SOFT: 1,
    }
    return max(categories, key=priority.__getitem__)
