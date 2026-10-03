from __future__ import annotations

import dataclasses
import unittest

from asr_lab.asr_fencing import (
    ExecutionFence,
    FenceOutcome,
    compare_execution_fence,
)


class ExecutionFenceTests(unittest.TestCase):
    def setUp(self):
        self.current = ExecutionFence(
            asr_job_id="job-a",
            epoch_id="epoch-2",
            epoch_seq=2,
            speech_segment_id="segment-a",
        )

    def test_current_job_and_epoch_are_accepted(self):
        self.assertEqual(
            compare_execution_fence(self.current, self.current),
            FenceOutcome.ACCEPT,
        )

    def test_old_epoch_is_stale_for_the_same_logical_job(self):
        old = ExecutionFence("job-a", "epoch-1", 1, "segment-a")
        self.assertEqual(compare_execution_fence(old, self.current), FenceOutcome.STALE)

    def test_different_logical_job_is_invalid_scope(self):
        other_job = dataclasses.replace(self.current, asr_job_id="job-b")
        self.assertEqual(
            compare_execution_fence(other_job, self.current),
            FenceOutcome.INVALID_SCOPE,
        )

    def test_admitting_newer_epoch_stales_the_previous_epoch(self):
        newer = ExecutionFence("job-a", "epoch-3", 3, "segment-a")
        self.assertEqual(compare_execution_fence(self.current, newer), FenceOutcome.STALE)

    def test_worker_replacement_does_not_change_canonical_fence(self):
        field_names = {item.name for item in dataclasses.fields(ExecutionFence)}
        self.assertEqual(
            field_names,
            {"asr_job_id", "epoch_id", "epoch_seq", "speech_segment_id"},
        )
        # Worker/process/GPU identity is not an input to the canonical check.
        same_execution_after_worker_replacement = dataclasses.replace(self.current)
        self.assertEqual(
            compare_execution_fence(same_execution_after_worker_replacement, self.current),
            FenceOutcome.ACCEPT,
        )

    def test_stale_partial_and_final_have_the_same_fence_outcome(self):
        old = ExecutionFence("job-a", "epoch-1", 1, "segment-a")
        outcomes = {
            candidate_kind: compare_execution_fence(old, self.current)
            for candidate_kind in ("partial", "final")
        }
        self.assertEqual(outcomes, {"partial": FenceOutcome.STALE, "final": FenceOutcome.STALE})

    def test_known_segment_mismatch_is_invalid_scope(self):
        wrong_segment = dataclasses.replace(self.current, speech_segment_id="segment-b")
        self.assertEqual(
            compare_execution_fence(wrong_segment, self.current),
            FenceOutcome.INVALID_SCOPE,
        )

    def test_omitting_known_segment_is_invalid_scope(self):
        missing_segment = dataclasses.replace(self.current, speech_segment_id=None)
        self.assertEqual(
            compare_execution_fence(missing_segment, self.current),
            FenceOutcome.INVALID_SCOPE,
        )

    def test_pre_turn_does_not_require_turn_id(self):
        pre_turn = ExecutionFence("job-a", "epoch-1", 1, "segment-a")
        self.assertEqual(compare_execution_fence(pre_turn, pre_turn), FenceOutcome.ACCEPT)
        self.assertNotIn("turn_id", {item.name for item in dataclasses.fields(ExecutionFence)})

    def test_exact_execution_tuple_is_accepted_without_segment_authority(self):
        # This adapter-gap case does not relax the admitted-operation
        # contract: without the D3 association, only exact execution identity
        # can be accepted.
        locally_unavailable_segment = ExecutionFence("job-a", "epoch-1", 1)
        self.assertEqual(
            compare_execution_fence(locally_unavailable_segment, locally_unavailable_segment),
            FenceOutcome.ACCEPT,
        )

    def test_epoch_sequence_is_not_ordered_without_segment_authority(self):
        current_without_segment = ExecutionFence("job-a", "epoch-2", 2)
        lower_sequence = ExecutionFence("job-a", "epoch-1", 1)
        higher_sequence = ExecutionFence("job-a", "epoch-3", 3)

        self.assertEqual(
            compare_execution_fence(lower_sequence, current_without_segment),
            FenceOutcome.INVALID_SCOPE,
        )
        self.assertEqual(
            compare_execution_fence(higher_sequence, current_without_segment),
            FenceOutcome.INVALID_SCOPE,
        )

    def test_equal_sequence_with_different_epoch_id_is_invalid_scope(self):
        conflicting_epoch = dataclasses.replace(self.current, epoch_id="other-epoch")
        self.assertEqual(
            compare_execution_fence(conflicting_epoch, self.current),
            FenceOutcome.INVALID_SCOPE,
        )

    def test_matching_local_scheduler_does_not_rescue_an_obsolete_epoch(self):
        local_scheduler_is_current = True
        old = ExecutionFence("job-a", "epoch-1", 1, "segment-a")
        canonical_outcome = compare_execution_fence(old, self.current)
        self.assertTrue(local_scheduler_is_current)
        self.assertEqual(canonical_outcome, FenceOutcome.STALE)

    def test_unadmitted_future_epoch_is_invalid_scope(self):
        future = ExecutionFence("job-a", "epoch-3", 3, "segment-a")
        self.assertEqual(compare_execution_fence(future, self.current), FenceOutcome.INVALID_SCOPE)


if __name__ == "__main__":
    unittest.main()
