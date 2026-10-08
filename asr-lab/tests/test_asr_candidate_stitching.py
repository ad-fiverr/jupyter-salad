from __future__ import annotations

import unittest

from asr_lab.asr_candidate_stitching import (
    ASRCandidateStitcher,
    CandidateBase,
    STITCH_POLICY_VERSION,
)
from asr_lab.asr_fencing import ExecutionFence


def fence(job: str = "job-1", segment: str = "segment-1", epoch: str = "epoch-1", seq: int = 1):
    return ExecutionFence(job, epoch, seq, segment)


def candidate(
    text: str,
    *,
    revision: int = 1,
    event: str = "partial_candidate",
    final: bool = False,
) -> dict[str, object]:
    return {
        "event": event,
        "text": text,
        "revision": revision,
        "final": final,
        "provisional": True,
        "candidate_only": True,
        "truth_status": "candidate_only",
        "replace": True,
        "stream_id": "stream-current",
        "request_id": "request-current",
        "QWEN_SCHEDULER_REVISION": 9,
    }


class ASRCandidateStitchingTests(unittest.TestCase):
    def make_stitcher(self, base_text: str | None = "earlier words"):
        base_fence = fence(epoch="epoch-1", seq=1)
        current_fence = fence(epoch="epoch-2", seq=2)
        base = CandidateBase(base_text, base_fence) if base_text is not None else None
        stitcher = ASRCandidateStitcher()
        stitcher.activate_epoch(current_fence, base=base)
        return stitcher, base_fence, current_fence

    def test_without_base_is_current_only_and_preserves_raw_candidate(self):
        stitcher, _, current = self.make_stitcher(None)
        raw = "  raw\t  current text  "
        result = stitcher.reconcile(candidate(raw), trusted_fence=current)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["text"], raw)
        self.assertEqual(result["stitch_raw_current_text"], raw)
        self.assertEqual(result["stitch_mode"], "current_only")
        self.assertEqual(result["stitch_reason"], "no_predecessor")
        self.assertIsNone(result["stitch_base_epoch_id"])
        self.assertEqual(result["stitch_anchor_continuity"], "none")
        self.assertIsNone(result["stitch_anchor_epoch_distance"])
        self.assertIsNone(result["stitch_empty_epoch_count"])

    def test_freeze_without_current_candidate_carries_only_verified_immediate_anchor(self):
        stitcher, base_fence, current = self.make_stitcher("older epoch candidate")
        self.assertIsNone(stitcher.snapshot().latest_text)
        carried = stitcher.freeze_for_successor(current)
        self.assertIsNotNone(carried)
        assert carried is not None
        self.assertEqual(carried.fence, base_fence)
        self.assertEqual(carried.lineage_tail_fence, current)
        self.assertEqual(carried.empty_epoch_count, 1)

    def test_largest_exact_suffix_prefix_overlap_wins(self):
        stitcher, base_fence, current = self.make_stitcher("alpha beta gamma")
        result = stitcher.reconcile(candidate("beta gamma delta"), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "alpha beta gamma delta")
        self.assertEqual(result["stitch_mode"], "exact_overlap")
        self.assertEqual(result["stitch_overlap_token_count"], 2)
        self.assertEqual(result["stitch_base_epoch_id"], base_fence.epoch_id)
        self.assertEqual(result["stitch_base_epoch_seq"], base_fence.epoch_seq)
        self.assertEqual(result["stitch_anchor_continuity"], "immediate_predecessor")
        self.assertEqual(result["stitch_anchor_epoch_distance"], 1)
        self.assertEqual(result["stitch_empty_epoch_count"], 0)

    def test_anchor_survives_two_verified_empty_epochs_and_reports_distance(self):
        anchor_fence = fence(epoch="epoch-1", seq=1)
        first_empty = fence(epoch="epoch-2", seq=2)
        second_empty = fence(epoch="epoch-3", seq=3)
        current = fence(epoch="epoch-4", seq=4)
        stitcher = ASRCandidateStitcher()
        stitcher.activate_epoch(first_empty, base=CandidateBase("we want to book", anchor_fence))

        carried_once = stitcher.freeze_for_successor(first_empty)
        self.assertIsNotNone(carried_once)
        assert carried_once is not None
        self.assertEqual(carried_once.fence, anchor_fence)
        self.assertEqual(carried_once.lineage_tail_fence, first_empty)
        self.assertEqual(carried_once.empty_epoch_count, 1)

        stitcher.activate_epoch(second_empty, base=carried_once)
        carried_twice = stitcher.freeze_for_successor(second_empty)
        self.assertIsNotNone(carried_twice)
        assert carried_twice is not None
        self.assertEqual(carried_twice.fence, anchor_fence)
        self.assertEqual(carried_twice.lineage_tail_fence, second_empty)
        self.assertEqual(carried_twice.empty_epoch_count, 2)

        stitcher.activate_epoch(current, base=carried_twice)
        result = stitcher.reconcile(candidate("to book a room"), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "we want to book a room")
        self.assertEqual(result["stitch_mode"], "exact_overlap")
        self.assertEqual(result["stitch_overlap_token_count"], 2)
        self.assertEqual(result["stitch_base_epoch_id"], anchor_fence.epoch_id)
        self.assertEqual(result["stitch_base_epoch_seq"], anchor_fence.epoch_seq)
        self.assertEqual(result["stitch_anchor_continuity"], "verified_empty_epochs")
        self.assertEqual(result["stitch_anchor_epoch_distance"], 3)
        self.assertEqual(result["stitch_empty_epoch_count"], 2)

    def test_rejected_candidate_breaks_empty_chain_fail_closed(self):
        stitcher, _base_fence, current = self.make_stitcher("verified anchor")
        rejected = candidate("candidate must not extend this chain")
        rejected["revision"] = True
        self.assertIsNone(stitcher.reconcile(rejected, trusted_fence=current))
        self.assertEqual(stitcher.snapshot().last_rejection_reason, "invalid_candidate_revision")
        self.assertIsNone(stitcher.freeze_for_successor(current))

    def test_final_candidate_does_not_seed_a_successor_anchor(self):
        stitcher, _, current = self.make_stitcher("prior words")
        final = candidate("terminal words", event="final_candidate", final=True)
        self.assertIsNotNone(stitcher.reconcile(final, trusted_fence=current))
        self.assertIsNone(stitcher.freeze_for_successor(current))

    def test_strong_overlap_removes_only_the_current_prefix(self):
        stitcher, _, current = self.make_stitcher("I would like to book")
        result = stitcher.reconcile(
            candidate("like to book a table for four"),
            trusted_fence=current,
        )
        assert result is not None
        self.assertEqual(result["text"], "I would like to book a table for four")
        self.assertEqual(result["stitch_overlap_token_count"], 3)

    def test_zero_overlap_appends_with_single_join_separator(self):
        stitcher, _, current = self.make_stitcher("alpha beta")
        result = stitcher.reconcile(candidate("gamma   delta"), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "alpha beta gamma   delta")
        self.assertEqual(result["stitch_mode"], "no_overlap")
        self.assertEqual(result["stitch_overlap_token_count"], 0)

    def test_one_token_overlap_is_ambiguous_and_keeps_duplicate(self):
        stitcher, _, current = self.make_stitcher("we need")
        result = stitcher.reconcile(candidate("need more"), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "we need need more")
        self.assertEqual(result["stitch_mode"], "ambiguous_weak_overlap")
        self.assertEqual(result["stitch_overlap_token_count"], 1)

    def test_comparison_uses_nfkc_and_casefold_but_emission_stays_raw(self):
        base = "ＦＯＯ Café"
        current_text = "foo cafe\u0301   NOW"
        stitcher, _, current = self.make_stitcher(base)
        result = stitcher.reconcile(candidate(current_text), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "ＦＯＯ Café NOW")
        self.assertEqual(result["stitch_raw_current_text"], current_text)
        self.assertEqual(result["stitch_overlap_token_count"], 2)

    def test_punctuation_diacritics_numbers_and_symbols_are_not_stripped(self):
        cases = [
            ("hello, world", "world! next"),
            ("café", "cafe next"),
            ("room 204", "205 please"),
            ("use +", "plus now"),
        ]
        for base_text, current_text in cases:
            with self.subTest(base=base_text, current=current_text):
                stitcher, _, current = self.make_stitcher(base_text)
                result = stitcher.reconcile(candidate(current_text), trusted_fence=current)
                assert result is not None
                self.assertEqual(result["stitch_mode"], "no_overlap")
                self.assertEqual(result["stitch_overlap_token_count"], 0)
                self.assertEqual(result["text"], f"{base_text} {current_text}")

    def test_internal_current_whitespace_is_preserved_after_boundary_join(self):
        stitcher, _, current = self.make_stitcher("we want to")
        current_text = "want to   order\t now  please"
        result = stitcher.reconcile(candidate(current_text), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "we want to order\t now  please")
        self.assertEqual(result["stitch_raw_current_text"], current_text)

    def test_empty_base_is_current_only_and_empty_current_keeps_valid_base(self):
        empty_base, _, current = self.make_stitcher("")
        result = empty_base.reconcile(candidate("new words"), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "new words")
        self.assertEqual(result["stitch_reason"], "empty_base")

        with_base, base_fence, current = self.make_stitcher("stable base")
        result = with_base.reconcile(candidate(""), trusted_fence=current)
        assert result is not None
        self.assertEqual(result["text"], "stable base")
        self.assertEqual(result["stitch_mode"], "base_only_empty_current")
        self.assertEqual(result["stitch_raw_current_text"], "")
        self.assertEqual(result["stitch_base_epoch_id"], base_fence.epoch_id)

    def test_wrong_job_segment_and_non_immediate_epoch_fail_to_current_only(self):
        base_fence = fence(epoch="epoch-1", seq=1)
        cases = [
            fence(job="other-job", epoch="epoch-2", seq=2),
            fence(segment="other-segment", epoch="epoch-2", seq=2),
            fence(epoch="epoch-3", seq=3),
        ]
        for current in cases:
            with self.subTest(current=current):
                stitcher = ASRCandidateStitcher()
                stitcher.activate_epoch(
                    current,
                    base=CandidateBase("base text", base_fence),
                )
                result = stitcher.reconcile(candidate("base text plus"), trusted_fence=current)
                assert result is not None
                self.assertEqual(result["text"], "base text plus")
                self.assertEqual(result["stitch_mode"], "current_only")
                self.assertEqual(result["stitch_reason"], "lineage_gap")
                self.assertEqual(result["stitch_anchor_continuity"], "none")
                self.assertIsNone(result["stitch_anchor_epoch_distance"])
                self.assertIsNone(result["stitch_empty_epoch_count"])

    def test_higher_revision_recomputes_against_same_frozen_base(self):
        stitcher, _, current = self.make_stitcher("red green blue")
        first = stitcher.reconcile(
            candidate("green blue yellow", revision=3),
            trusted_fence=current,
        )
        second = stitcher.reconcile(
            candidate("blue cyan", revision=4),
            trusted_fence=current,
        )
        assert first is not None and second is not None
        self.assertEqual(first["text"], "red green blue yellow")
        self.assertEqual(second["text"], "red green blue blue cyan")
        self.assertEqual(stitcher.snapshot().base.text, "red green blue")

    def test_lower_revision_is_suppressed(self):
        stitcher, _, current = self.make_stitcher()
        stitcher.reconcile(candidate("first", revision=4), trusted_fence=current)
        self.assertIsNone(stitcher.reconcile(candidate("older", revision=3), trusted_fence=current))
        self.assertEqual(stitcher.snapshot().last_rejection_reason, "obsolete_revision")

    def test_equal_revision_identical_event_is_idempotent_and_collision_fails_closed(self):
        stitcher, _, current = self.make_stitcher()
        first = candidate("same text", revision=5)
        self.assertIsNotNone(stitcher.reconcile(first, trusted_fence=current))
        self.assertIsNone(stitcher.reconcile(dict(first), trusted_fence=current))
        self.assertEqual(stitcher.snapshot().last_rejection_reason, "duplicate_candidate")

        collision = candidate("different text", revision=5)
        self.assertIsNone(stitcher.reconcile(collision, trusted_fence=current))
        self.assertEqual(stitcher.snapshot().last_rejection_reason, "revision_text_collision")

    def test_same_revision_partial_to_final_still_emits(self):
        stitcher, _, current = self.make_stitcher("previous words here")
        partial = candidate("words here today", revision=7)
        final = candidate(
            "words here today",
            revision=7,
            event="final_candidate",
            final=True,
        )
        self.assertIsNotNone(stitcher.reconcile(partial, trusted_fence=current))
        result = stitcher.reconcile(final, trusted_fence=current)
        assert result is not None
        self.assertTrue(result["final"])
        self.assertEqual(result["text"], "previous words here today")
        self.assertIsNone(stitcher.reconcile(final, trusted_fence=current))
        self.assertEqual(stitcher.snapshot().last_rejection_reason, "duplicate_candidate")

    def test_candidate_fields_and_provisional_semantics_are_preserved(self):
        stitcher, base_fence, current = self.make_stitcher("previous phrase")
        source = candidate("phrase continues")
        source["asr_job_id"] = "job-current"
        source["speech_segment_id"] = "segment-current"
        source["epoch_id"] = current.epoch_id
        source["epoch_seq"] = current.epoch_seq
        source["custom_metric"] = 42
        result = stitcher.reconcile(source, trusted_fence=current)
        assert result is not None
        for key in source:
            self.assertIn(key, result)
        self.assertEqual(result["custom_metric"], 42)
        self.assertEqual(result["asr_job_id"], current.asr_job_id)
        self.assertEqual(result["speech_segment_id"], current.speech_segment_id)
        self.assertEqual(result["epoch_id"], current.epoch_id)
        self.assertEqual(result["epoch_seq"], current.epoch_seq)
        self.assertEqual(result["stitch_base_epoch_id"], base_fence.epoch_id)
        self.assertNotIn("turnId", result)
        self.assertTrue(result["provisional"])
        self.assertTrue(result["candidate_only"])
        self.assertEqual(result["truth_status"], "candidate_only")
        self.assertTrue(result["replace"])
        self.assertEqual(result["stitch_policy_version"], STITCH_POLICY_VERSION)
        self.assertEqual(result["stitch_anchor_continuity"], "immediate_predecessor")
        self.assertEqual(result["stitch_anchor_epoch_distance"], 1)
        self.assertEqual(result["stitch_empty_epoch_count"], 0)

    def test_current_fence_mismatch_fails_closed(self):
        stitcher, _, current = self.make_stitcher()
        other = fence(job="other-job", epoch="epoch-2", seq=2)
        self.assertIsNone(stitcher.reconcile(candidate("text"), trusted_fence=other))
        self.assertEqual(stitcher.snapshot().last_rejection_reason, "current_fence_mismatch")


if __name__ == "__main__":
    unittest.main()
