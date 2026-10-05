"""Deterministic, provisional stitching of adjacent Qwen epoch candidates.

This module owns transcript-text reconciliation only. It does not establish
job/segment/epoch identity, freshness, PCM ownership, EOS, or business truth.
Callers must pass a trusted ``ExecutionFence`` after their local and canonical
freshness gates have accepted the result.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .asr_fencing import ExecutionFence


STITCH_POLICY_VERSION = "a6.5-exact-boundary-v1"


@dataclass(frozen=True)
class CandidateBase:
    """A previously emitted logical candidate and its trusted producer fence."""

    text: str
    fence: ExecutionFence


@dataclass(frozen=True)
class CandidateStitcherSnapshot:
    """Read-only state projection for lifecycle tests and diagnostics."""

    current_fence: ExecutionFence | None
    base: CandidateBase | None
    latest_text: str | None
    latest_revision: int | None
    highest_revision: int | None
    last_rejection_reason: str | None


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _normalized_token(token: str) -> str:
    """Normalize comparison tokens only; emitted text always remains raw."""
    return unicodedata.normalize("NFKC", token).casefold()


def _largest_exact_overlap(base_text: str, current_text: str) -> int:
    base_tokens = [_normalized_token(token) for token, _, _ in _token_spans(base_text)]
    current_tokens = [_normalized_token(token) for token, _, _ in _token_spans(current_text)]
    for count in range(min(len(base_tokens), len(current_tokens)), 0, -1):
        if base_tokens[-count:] == current_tokens[:count]:
            return count
    return 0


def _joined(base_text: str, current_suffix: str) -> str:
    """Keep the frozen base and current internal whitespace; normalize join."""
    if not current_suffix:
        return base_text
    if not base_text:
        return current_suffix
    return f"{base_text} {current_suffix.lstrip()}"


class ASRCandidateStitcher:
    """Reconcile candidate revisions against one frozen predecessor epoch.

    A new epoch always re-evaluates each accepted revision against the same
    frozen base. The previous stitched revision is never used as the next base.
    """

    def __init__(self) -> None:
        self._current_fence: ExecutionFence | None = None
        self._base: CandidateBase | None = None
        self._latest: CandidateBase | None = None
        self._latest_revision: int | None = None
        self._highest_revision: int | None = None
        self._last_raw_text: str | None = None
        self._last_event: str | None = None
        self._last_final: bool | None = None
        self._last_rejection_reason: str | None = None

    def activate_epoch(
        self,
        fence: ExecutionFence,
        *,
        base: CandidateBase | None = None,
    ) -> None:
        """Bind text reconciliation to the trusted current epoch."""
        if not isinstance(fence, ExecutionFence):
            raise TypeError("candidate_stitcher_fence_invalid")
        if base is not None and not isinstance(base, CandidateBase):
            raise TypeError("candidate_stitcher_base_invalid")
        self._current_fence = fence
        self._base = base
        self._latest = None
        self._latest_revision = None
        self._highest_revision = None
        self._last_raw_text = None
        self._last_event = None
        self._last_final = None
        self._last_rejection_reason = None

    def freeze_for_successor(self, fence: ExecutionFence) -> CandidateBase | None:
        """Freeze this epoch's latest candidate for a successor.

        If this epoch emitted no accepted logical candidate, its successor
        starts without a stitching base. Do not carry a base across the gap.
        """
        if self._current_fence != fence:
            raise ValueError("candidate_stitcher_freeze_fence_mismatch")
        return self._latest

    def reconcile(
        self,
        candidate: dict[str, Any],
        *,
        trusted_fence: ExecutionFence,
    ) -> dict[str, Any] | None:
        """Return a provisional logical candidate, or ``None`` fail-closed."""
        self._last_rejection_reason = None
        if self._current_fence is None or trusted_fence != self._current_fence:
            return self._reject("current_fence_mismatch")
        if not isinstance(candidate, dict):
            return self._reject("invalid_candidate")

        raw_text = candidate.get("text")
        revision = candidate.get("revision")
        event = candidate.get("event")
        final = candidate.get("final")
        if not isinstance(raw_text, str):
            return self._reject("invalid_candidate_text")
        if (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
        ):
            return self._reject("invalid_candidate_revision")
        if (
            event not in {"partial_candidate", "final_candidate"}
            or final is not (event == "final_candidate")
        ):
            return self._reject("invalid_candidate_event")

        if self._highest_revision is not None:
            if revision < self._highest_revision:
                return self._reject("obsolete_revision")
            if revision == self._highest_revision:
                if raw_text != self._last_raw_text:
                    return self._reject("revision_text_collision")
                if event == self._last_event and final == self._last_final:
                    return self._reject("duplicate_candidate")
                if self._last_final is True:
                    return self._reject("candidate_after_final")
                if not (
                    self._last_event == "partial_candidate"
                    and self._last_final is False
                    and event == "final_candidate"
                    and final is True
                ):
                    return self._reject("revision_event_collision")
            elif self._last_final is True:
                return self._reject("candidate_after_final")

        logical_text, mode, reason, overlap = self._stitch_text(raw_text, trusted_fence)
        result = dict(candidate)
        result["text"] = logical_text
        result.update({
            "asr_job_id": trusted_fence.asr_job_id,
            "speech_segment_id": trusted_fence.speech_segment_id,
            "epoch_id": trusted_fence.epoch_id,
            "epoch_seq": trusted_fence.epoch_seq,
        })
        base = self._base
        result.update({
            "stitch_policy_version": STITCH_POLICY_VERSION,
            "stitch_mode": mode,
            "stitch_reason": reason,
            "stitch_overlap_token_count": overlap,
            "stitch_raw_current_text": raw_text,
            "stitch_base_epoch_id": base.fence.epoch_id if base is not None else None,
            "stitch_base_epoch_seq": base.fence.epoch_seq if base is not None else None,
        })

        self._highest_revision = revision
        self._latest_revision = revision
        self._last_raw_text = raw_text
        self._last_event = event
        self._last_final = final
        self._latest = CandidateBase(text=logical_text, fence=trusted_fence)
        return result

    def snapshot(self) -> CandidateStitcherSnapshot:
        return CandidateStitcherSnapshot(
            current_fence=self._current_fence,
            base=self._base,
            latest_text=self._latest.text if self._latest is not None else None,
            latest_revision=self._latest_revision,
            highest_revision=self._highest_revision,
            last_rejection_reason=self._last_rejection_reason,
        )

    def _stitch_text(
        self,
        raw_text: str,
        current_fence: ExecutionFence,
    ) -> tuple[str, str, str, int]:
        base = self._base
        if base is None:
            return raw_text, "current_only", "no_predecessor", 0
        if not base.text.strip():
            return raw_text, "current_only", "empty_base", 0
        if (
            not current_fence.asr_job_id
            or not current_fence.speech_segment_id
            or current_fence.asr_job_id != base.fence.asr_job_id
            or current_fence.speech_segment_id != base.fence.speech_segment_id
            or current_fence.epoch_seq != base.fence.epoch_seq + 1
        ):
            return raw_text, "current_only", "lineage_gap", 0
        if not raw_text.strip():
            return base.text, "base_only_empty_current", "empty_current", 0

        overlap = _largest_exact_overlap(base.text, raw_text)
        if overlap >= 2:
            spans = _token_spans(raw_text)
            # Start exactly at the first unmatched token: discard only the
            # matched current prefix and its boundary separator.
            unmatched = raw_text[spans[overlap][1]:] if overlap < len(spans) else ""
            return _joined(base.text, unmatched), "exact_overlap", "exact_suffix_prefix", overlap
        if overlap == 1:
            return (
                _joined(base.text, raw_text),
                "ambiguous_weak_overlap",
                "single_token_overlap_preserved",
                overlap,
            )
        return _joined(base.text, raw_text), "no_overlap", "no_exact_overlap", 0

    def _reject(self, reason: str) -> None:
        self._last_rejection_reason = reason
        return None
