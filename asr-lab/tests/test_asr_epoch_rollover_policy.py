from __future__ import annotations

import unittest

from asr_lab.asr_epoch_rollover_policy import (
    MAX_POLICY_WINDOW_SAMPLES,
    RolloverCategory,
    SignalClass,
    SoftRolloverEvaluator,
    SoftRolloverPolicy,
    SoftSignalDefinition,
    choose_rollover_category,
)


def policy(*, window=3, required=2, cooldown=100.0, signals=None):
    return SoftRolloverPolicy(
        soft_enabled=True,
        window_samples=window,
        required_qualifying_samples=required,
        min_active_signals=2,
        cooldown_ms=cooldown,
        signals=tuple(signals or (
            SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 100.0, 80.0),
            SoftSignalDefinition("scheduler_backlog_ms", SignalClass.LOCAL_THROUGHPUT, 50.0, 25.0),
        )),
    )


class SoftRolloverPolicyTests(unittest.TestCase):
    def test_absent_or_incomplete_policy_config_is_unarmed(self):
        self.assertIsNone(SoftRolloverPolicy.from_optional_config(
            soft_enabled=None,
            window_samples=None,
            required_qualifying_samples=None,
            min_active_signals=None,
            cooldown_ms=None,
            signals=None,
        ))
        self.assertIsNone(SoftRolloverPolicy.from_optional_config(
            soft_enabled=True,
            window_samples=3,
            required_qualifying_samples=None,
            min_active_signals=2,
            cooldown_ms=10.0,
            signals=(),
        ))

    def test_unbounded_invalid_and_duplicate_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite_and_bounded"):
            policy(window=MAX_POLICY_WINDOW_SAMPLES + 1)
        with self.assertRaisesRegex(ValueError, "qualifying_sample_count"):
            policy(required=1)
        with self.assertRaisesRegex(ValueError, "multiple_signals"):
            SoftRolloverPolicy(True, 2, 2, 1, 0.0, (
                SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 2, 1),
                SoftSignalDefinition("scheduler_backlog_ms", SignalClass.LOCAL_THROUGHPUT, 2, 1),
            ))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            policy(signals=(
                SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 2, 1),
                SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 3, 2),
            ))
        with self.assertRaisesRegex(ValueError, "class_mismatch"):
            SoftSignalDefinition("epoch_audio_ms", SignalClass.LOCAL_THROUGHPUT, 2, 1)

    def test_single_spike_never_triggers_and_requires_current_multisignal_evidence(self):
        evaluator = SoftRolloverEvaluator(policy(required=2))
        decision = evaluator.observe(
            {"epoch_audio_ms": 120, "scheduler_backlog_ms": 0}, now=0,
        )
        self.assertFalse(decision.qualifying)
        self.assertFalse(decision.should_rollover)
        self.assertEqual(decision.active_signals, ("epoch_audio_ms",))

    def test_persistent_multisignal_fires_once_then_cooldown_suppresses_soft(self):
        evaluator = SoftRolloverEvaluator(policy(window=4, required=2, cooldown=100.0))
        metrics = {"epoch_audio_ms": 120, "scheduler_backlog_ms": 60}
        first = evaluator.observe(metrics, now=1.0)
        second = evaluator.observe(metrics, now=1.1)
        self.assertFalse(first.should_rollover)
        self.assertTrue(second.should_rollover)
        self.assertEqual(second.qualifying_samples, 2)

        evaluator.mark_rollover_succeeded(now=1.2)
        during_cooldown = evaluator.observe(metrics, now=1.25)
        self.assertFalse(during_cooldown.should_rollover)
        self.assertTrue(during_cooldown.cooldown_active)
        self.assertEqual(during_cooldown.qualifying_samples, 1)
        self.assertEqual(evaluator.snapshot(now=1.25)["policy_samples_retained"], 1)

    def test_hysteresis_preserves_middle_band_and_clears_at_clear_threshold(self):
        evaluator = SoftRolloverEvaluator(policy(required=3))
        evaluator.observe({"epoch_audio_ms": 110, "scheduler_backlog_ms": 60}, now=0.0)
        middle = evaluator.observe({"epoch_audio_ms": 90, "scheduler_backlog_ms": 40}, now=0.1)
        cleared = evaluator.observe({"epoch_audio_ms": 80, "scheduler_backlog_ms": 25}, now=0.2)
        self.assertIn("epoch_audio_ms", middle.active_signals)
        self.assertIn("scheduler_backlog_ms", middle.active_signals)
        self.assertEqual(cleared.active_signals, ())

    def test_device_global_capacity_is_observable_but_cannot_trigger(self):
        configured = policy(signals=(
            SoftSignalDefinition("epoch_audio_ms", SignalClass.ACOUSTIC_HISTORY, 100, 80),
            SoftSignalDefinition("scheduler_backlog_ms", SignalClass.LOCAL_THROUGHPUT, 50, 25),
            SoftSignalDefinition("global_backlog_ms", SignalClass.DEVICE_GLOBAL_CAPACITY, 10, 5),
        ))
        evaluator = SoftRolloverEvaluator(configured)
        decision = evaluator.observe({
            "epoch_audio_ms": 0,
            "scheduler_backlog_ms": 0,
            "global_backlog_ms": 100,
        }, now=0.0)
        self.assertFalse(decision.should_rollover)
        self.assertEqual(decision.active_signals, ())
        self.assertEqual(decision.ignored_capacity_signals, ("global_backlog_ms",))
        self.assertIn(
            "global_backlog_ms",
            evaluator.snapshot(now=0.0)["non_triggering_capacity_signals"],
        )

    def test_qualifying_window_is_bounded_and_missing_data_fails_closed(self):
        evaluator = SoftRolloverEvaluator(policy(window=2, required=2))
        high = {"epoch_audio_ms": 120, "scheduler_backlog_ms": 60}
        evaluator.observe(high, now=0.0)
        evaluator.observe({"epoch_audio_ms": 120}, now=0.1)
        decision = evaluator.observe(high, now=0.2)
        self.assertEqual(decision.qualifying_samples, 1)
        self.assertFalse(decision.should_rollover)
        self.assertEqual(evaluator.snapshot(now=0.2)["policy_samples_retained"], 2)

    def test_trigger_category_precedence_is_deterministic(self):
        self.assertIs(
            choose_rollover_category(
                RolloverCategory.PROACTIVE_SOFT,
                RolloverCategory.HARD_BOUND,
                RolloverCategory.EMERGENCY_BACKPRESSURE,
            ),
            RolloverCategory.EMERGENCY_BACKPRESSURE,
        )
        self.assertIs(
            choose_rollover_category(RolloverCategory.PROACTIVE_SOFT, RolloverCategory.MANUAL_TEST),
            RolloverCategory.MANUAL_TEST,
        )


if __name__ == "__main__":
    unittest.main()
