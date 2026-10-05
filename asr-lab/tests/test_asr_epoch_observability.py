from __future__ import annotations

import unittest

from asr_lab.asr_epoch_observability import ASREpochObservability
from asr_lab.asr_fencing import ExecutionFence


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class ASREpochObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.recorder = ASREpochObservability(monotonic=self.clock)
        self.fence = ExecutionFence("job-1", "epoch-1", 1, "speech-1")

    def test_initial_epoch_snapshot_separates_local_global_and_pcm_metrics(self):
        self.recorder.start_epoch(
            fence=self.fence,
            stream_id="local-stream-1",
            stream_init_ms=4.5,
            now=0.0,
        )
        self.recorder.observe(
            stream_snapshot={
                "accepted_audio_ms": 125.0,
                "scheduler_pending_jobs": 2,
                "scheduler_max_backlog_audio_ms": 80.0,
                "stream_lag_ms": 55.0,
                "stream_lag_max_ms": 60.0,
                "scheduler_wait_p95_ms": 3.5,
                "decode_wall_p95_ms": 9.0,
                "decode_steps_delta_total": 3,
                "text": "must not be copied",
            },
            global_snapshot={
                "active_stream_count": 6,
                "pending_decode_count": 4,
                "qwen_scheduler_backlog_ms": 250.0,
            },
            pcm_snapshot={
                "source_head_cursor": 5000,
                "unique_primary_admitted_cursor": 4000,
                "current_epoch_admitted_cursor": 4000,
                "processed_cursor": 3000,
                "received_samples": 5000,
                "unique_primary_admitted_samples": 4000,
                "replay_admitted_samples": 0,
                "transition_queued_samples": 1000,
            },
            now=1.0,
        )
        snapshot = self.recorder.snapshot(now=1.5)
        current = snapshot["current_epoch"]
        self.assertEqual(current["epoch_id"], "epoch-1")
        self.assertEqual(current["epoch_audio_ms"], 125.0)
        self.assertEqual(current["epoch_age_ms"], 1500.0)
        self.assertEqual(current["EPOCH_AUDIO_MS"], 125.0)
        self.assertEqual(current["EPOCH_STATE_INIT_MS"], None)
        self.assertEqual(current["scheduler_wait_p95_ms"], 3.5)
        self.assertEqual(current["global_scheduler"]["active_stream_count"], 6)
        self.assertEqual(current["pcm"]["transition_queued_samples"], 1000)
        self.assertEqual(snapshot["logical_cumulative"]["source_received_samples"], 5000)
        self.assertNotIn("text", current)

    def test_epoch_reset_preserves_logical_totals_and_replay_accounting(self):
        self.recorder.start_epoch(fence=self.fence, stream_id="stream-1", stream_init_ms=2, now=0)
        self.recorder.observe(
            stream_snapshot={"accepted_audio_ms": 200},
            global_snapshot={},
            pcm_snapshot={"received_samples": 3200, "unique_primary_admitted_samples": 3200},
            now=0.5,
        )
        successor = ExecutionFence("job-1", "epoch-2", 2, "speech-1")
        self.recorder.start_epoch(
            fence=successor,
            stream_id="stream-2",
            stream_init_ms=3,
            replay_audio_samples=800,
            now=1.0,
        )
        self.recorder.observe(
            stream_snapshot={"accepted_audio_ms": 50},
            global_snapshot={},
            pcm_snapshot={"received_samples": 4000, "unique_primary_admitted_samples": 4000},
            now=1.2,
        )
        snapshot = self.recorder.snapshot(now=1.2)
        self.assertEqual(snapshot["current_epoch"]["epoch_id"], "epoch-2")
        self.assertEqual(snapshot["current_epoch"]["epoch_audio_ms"], 50)
        self.assertEqual(snapshot["current_epoch"]["replay_audio_ms"], 50.0)
        self.recorder.record_replay_wall(replay_wall_ms=7.5)
        snapshot = self.recorder.snapshot(now=1.2)
        self.assertEqual(snapshot["current_epoch"]["EPOCH_REPLAY_WALL_MS"], 7.5)
        self.assertEqual(snapshot["logical_cumulative"]["source_received_samples"], 4000)
        self.assertEqual(snapshot["logical_cumulative"]["unique_primary_samples"], 4000)

    def test_snapshot_nested_values_cannot_mutate_internal_accounting(self):
        self.recorder.start_epoch(
            fence=self.fence,
            stream_id="local-stream-1",
            stream_init_ms=4.5,
            now=0.0,
        )
        self.recorder.observe(
            stream_snapshot={},
            global_snapshot={},
            pcm_snapshot={"received_samples": 1600},
            now=1.0,
        )
        self.recorder.begin_transition(
            category="manual_test",
            reason="manual_test",
            details={"trigger_signals": ["epoch_audio_ms"]},
            cutover_cursor=800,
            stream_snapshot={},
            pcm_snapshot={"received_samples": 1600},
            now=2.0,
        )

        exposed = self.recorder.snapshot(now=2.1)
        exposed["current_epoch"]["pcm"]["received_samples"] = 999
        exposed["last_transition"]["details"]["trigger_signals"].append("forged")

        verified = self.recorder.snapshot(now=2.2)
        self.assertEqual(verified["current_epoch"]["pcm"]["received_samples"], 1600)
        self.assertEqual(
            verified["last_transition"]["details"]["trigger_signals"],
            ["epoch_audio_ms"],
        )

    def test_transition_handoff_and_first_partial_metrics_are_precise_and_one_shot(self):
        self.recorder.start_epoch(fence=self.fence, stream_id="stream-1", stream_init_ms=2, now=0)
        self.recorder.begin_transition(
            category="hard_bound",
            reason="hard_bound",
            details={
                "hard_limit_ms": 60000,
                "active_signals": ["private-safe-signal"],
                "transcript": "must not be retained",
            },
            cutover_cursor=4000,
            stream_snapshot={"decode_wall_p95_ms": 8.0},
            pcm_snapshot={"received_samples": 5000, "unique_primary_admitted_samples": 4000},
            now=1.0,
        )
        successor = ExecutionFence("job-1", "epoch-2", 2, "speech-1")
        self.recorder.start_epoch(
            fence=successor,
            stream_id="stream-2",
            stream_init_ms=3,
            replay_audio_samples=500,
            now=1.1,
        )
        self.assertIsNone(self.recorder.record_first_partial_after_rollover(canonical_cursor=4000, now=1.2))
        first = self.recorder.record_first_partial_after_rollover(canonical_cursor=4200, now=1.35)
        second = self.recorder.record_first_partial_after_rollover(canonical_cursor=4300, now=1.5)
        self.assertEqual(first, 350.0)
        self.assertIsNone(second)
        elapsed = self.recorder.complete_transition(
            success=True,
            replay_admitted_samples=500,
            transition_primary_samples=1000,
            stream_snapshot={"accepted_audio_ms": 93.75},
            global_snapshot={"active_stream_count": 1},
            pcm_snapshot={
                "received_samples": 5000,
                "unique_primary_admitted_samples": 5000,
                "replay_admitted_samples": 500,
            },
            now=1.75,
        )
        snapshot = self.recorder.snapshot(now=2.0)
        transition = snapshot["last_transition"]
        self.assertEqual(elapsed, 750.0)
        self.assertEqual(transition["EPOCH_HANDOFF_WALL_MS"], 750.0)
        self.assertEqual(transition["EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS"], 350.0)
        self.assertEqual(transition["state"], "ACTIVE")
        self.assertEqual(transition["replay_admitted_samples"], 500)
        self.assertEqual(transition["transition_primary_samples"], 1000)
        self.assertEqual(snapshot["logical_cumulative"]["rollover_success_count"], 1)
        rendered = repr(snapshot)
        self.assertNotIn("must not be retained", rendered)
        self.assertNotIn("transcript", rendered)

    def test_stale_result_rejects_and_failed_transition_are_retained(self):
        self.recorder.start_epoch(fence=self.fence, stream_id="stream-1", stream_init_ms=None, now=0)
        self.recorder.record_stale_result_reject(
            gate="canonical_execution_fence_stale", stream_id="stream-1",
        )
        self.recorder.begin_transition(
            category="emergency_backpressure",
            reason="stream_scheduler_overrun",
            details={"scheduler_overrun_reason": "stream_scheduler_overrun"},
            cutover_cursor=0,
            predecessor_stream_id="stream-1",
            stream_snapshot={},
            pcm_snapshot={},
            now=1,
        )
        self.recorder.record_stale_result_reject(
            gate="scheduler_request_key_stale", stream_id="stream-1",
        )
        self.recorder.complete_transition(
            success=False,
            failure_reason="successor_init_failed",
            pcm_snapshot={},
            now=1.25,
        )
        snapshot = self.recorder.snapshot(now=2)
        self.assertEqual(snapshot["current_epoch"]["stale_result_rejects"], 2)
        self.assertEqual(snapshot["logical_cumulative"]["stale_result_rejects"], 2)
        self.assertEqual(snapshot["logical_cumulative"]["rollover_failure_count"], 1)
        self.assertEqual(snapshot["last_transition"]["failure_reason"], "successor_init_failed")
        self.assertEqual(snapshot["last_transition"]["EPOCH_HANDOFF_WALL_MS"], 250.0)
        self.assertEqual(snapshot["last_transition"]["stale_result_rejects_start"], 1)
        self.assertEqual(snapshot["last_transition"]["stale_result_rejects_end"], 2)
        self.assertEqual(snapshot["last_transition"]["stale_result_rejects_delta"], 1)

    def test_stale_reject_counts_are_local_and_transition_tracks_predecessor_only(self):
        self.recorder.start_epoch(fence=self.fence, stream_id="stream-a", stream_init_ms=None, now=0)
        self.recorder.record_stale_result_reject(
            gate="canonical_execution_fence_stale", stream_id="stream-a",
        )
        self.recorder.record_stale_result_reject(
            gate="canonical_execution_fence_stale", stream_id="stream-a",
        )
        self.recorder.record_stale_result_reject(
            gate="scheduler_request_key_stale", stream_id="stream-a",
        )
        self.recorder.record_stale_result_reject(
            gate="runtime_scheduler_key_stale", stream_id="stream-a",
        )
        self.recorder.record_stale_result_reject(
            gate="runtime_session_key_stale", stream_id="stream-a",
        )
        before = self.recorder.snapshot(now=0)["logical_cumulative"]
        self.assertEqual(before["canonical_fence_rejects"], 2)
        self.assertEqual(before["local_request_freshness_rejects"], 3)
        self.assertEqual(before["stale_result_rejects"], 5)

        self.recorder.begin_transition(
            category="manual_test", reason="manual_test", details={}, cutover_cursor=0,
            predecessor_stream_id="stream-a", stream_snapshot={}, pcm_snapshot={}, now=1,
        )
        self.recorder.record_stale_result_reject(
            gate="scheduler_request_key_stale", stream_id="stream-a",
        )
        self.recorder.record_stale_result_reject(
            gate="runtime_session_key_stale", stream_id="stream-a",
        )
        pending = self.recorder.snapshot(now=1.1)["last_transition"]
        self.assertEqual(pending["stale_result_rejects_start"], 5)
        self.assertEqual(pending["stale_result_rejects_end"], 7)
        self.assertEqual(pending["stale_result_rejects_delta"], 2)
        self.assertEqual(pending["stale_result_reject_gate_counts"]["scheduler_request_key_stale"], 1)

        successor = ExecutionFence("job-1", "epoch-2", 2, "speech-1")
        self.recorder.start_epoch(
            fence=successor, stream_id="stream-b", stream_init_ms=1, now=1.1,
        )
        self.recorder.complete_transition(success=True, pcm_snapshot={}, now=1.2)
        self.recorder.record_stale_result_reject(
            gate="runtime_scheduler_key_stale", stream_id="stream-a",
        )
        after_predecessor = self.recorder.snapshot(now=1.3)["last_transition"]
        self.assertEqual(after_predecessor["stale_result_rejects_end"], 8)
        self.assertEqual(after_predecessor["stale_result_rejects_delta"], 3)

        self.recorder.record_stale_result_reject(
            gate="runtime_session_key_stale", stream_id="stream-b",
        )
        final = self.recorder.snapshot(now=1.4)
        self.assertEqual(final["logical_cumulative"]["stale_result_rejects"], 9)
        self.assertEqual(final["current_epoch"]["stale_result_rejects"], 1)
        self.assertEqual(
            final["current_epoch"]["stale_result_reject_gates"]["runtime_session_key_stale"], 1,
        )
        self.assertEqual(
            final["current_epoch"]["stale_result_reject_gates"]["canonical_execution_fence_stale"], 0,
        )
        self.assertEqual(final["last_transition"]["stale_result_rejects_end"], 8)
        self.assertEqual(final["last_transition"]["stale_result_rejects_delta"], 3)

    def test_first_partial_can_arrive_after_handoff_wall_completes(self):
        self.recorder.start_epoch(fence=self.fence, stream_id="stream-1", stream_init_ms=1, now=0)
        self.recorder.begin_transition(
            category="manual_test",
            reason="manual_test",
            details={},
            cutover_cursor=1000,
            stream_snapshot={},
            pcm_snapshot={},
            now=1.0,
        )
        successor = ExecutionFence("job-1", "epoch-2", 2, "speech-1")
        self.recorder.start_epoch(fence=successor, stream_id="stream-2", stream_init_ms=1, now=1.1)
        self.recorder.complete_transition(success=True, pcm_snapshot={}, now=1.5)
        value = self.recorder.record_first_partial_after_rollover(canonical_cursor=1001, now=1.75)
        self.assertEqual(value, 750.0)
        self.assertEqual(
            self.recorder.snapshot(now=1.8)["last_transition"]["EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS"],
            750.0,
        )


if __name__ == "__main__":
    unittest.main()
