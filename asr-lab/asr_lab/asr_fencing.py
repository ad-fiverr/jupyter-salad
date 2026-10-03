"""Canonical ASR job/epoch fencing, independent of worker scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ExecutionFence:
    """Identity of one admitted acoustic execution.

    ``speech_segment_id`` is optional here only because this local comparator
    may not yet receive the authoritative D3 association. D3 still requires
    that identity on an admitted canonical PRE-TURN ASR operation. The current
    fence must come from ASR control; a candidate cannot establish or replace
    that authority.
    """

    asr_job_id: str
    epoch_id: str
    epoch_seq: int
    speech_segment_id: str | None = None


class FenceOutcome(str, Enum):
    ACCEPT = "ACCEPT"
    STALE = "STALE"
    INVALID_SCOPE = "INVALID_SCOPE"


def compare_execution_fence(
    candidate: ExecutionFence,
    current: ExecutionFence,
) -> FenceOutcome:
    """Compare a result's fence with the authoritative current execution.

    This is a pure canonical check. Callers must also apply any local adapter
    fence (for example, Qwen's scheduler revision) where that adapter is used.
    Passing the local check cannot override a non-ACCEPT canonical outcome.
    """

    if candidate.asr_job_id != current.asr_job_id:
        return FenceOutcome.INVALID_SCOPE

    if current.speech_segment_id is None:
        # D4 orders epochs only within one speech segment. Without the current
        # authoritative segment, sequence numbers have no proven shared
        # ordering domain; accept only the exact execution tuple.
        if (
            candidate.epoch_id != current.epoch_id
            or candidate.epoch_seq != current.epoch_seq
        ):
            return FenceOutcome.INVALID_SCOPE
    else:
        # Only a known current D3 association can establish the D4 ordering
        # domain. A missing or different candidate association cannot replace
        # that authority.
        if candidate.speech_segment_id != current.speech_segment_id:
            return FenceOutcome.INVALID_SCOPE

        if candidate.epoch_seq < current.epoch_seq:
            return FenceOutcome.STALE
        if candidate.epoch_seq > current.epoch_seq:
            # The candidate describes an execution ASR control has not
            # admitted as current in this validation context.
            return FenceOutcome.INVALID_SCOPE
        if candidate.epoch_id != current.epoch_id:
            return FenceOutcome.INVALID_SCOPE

    return FenceOutcome.ACCEPT
