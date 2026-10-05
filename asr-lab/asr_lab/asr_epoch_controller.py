"""Local controller for admitted ASR acoustic-epoch lineage.

This module owns only local epoch identity and lifecycle.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable
from uuid import uuid4

from .asr_fencing import ExecutionFence, FenceOutcome, compare_execution_fence


class EpochControllerState(str, Enum):
    ACTIVE = "ACTIVE"
    ROLLOVER_REQUESTED = "ROLLOVER_REQUESTED"
    SUCCESSOR_PREPARED = "SUCCESSOR_PREPARED"


class EpochControllerError(ValueError):
    """Deterministic error raised for invalid controller input or transition."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(detail or code)


EpochIdAllocator = Callable[[], str]


def _default_epoch_id_allocator() -> str:
    return uuid4().hex


class LocalAcousticEpochController:
    """Own the current local acoustic epoch for admitted job/segment work.

    ``initial_epoch_seq`` is required and has no product-wide default because
    the accepted design leaves that initial policy open. This controller
    advances the sequence from the supplied seed exactly once when a
    prepared successor is activated.
    """

    def __init__(
        self,
        *,
        asr_job_id: str,
        speech_segment_id: str,
        initial_epoch_seq: int,
        epoch_id_allocator: EpochIdAllocator | None = None,
    ) -> None:
        self._validate_non_empty_string(asr_job_id, "invalid_job_id")
        self._validate_non_empty_string(speech_segment_id, "invalid_speech_segment_id")
        if isinstance(initial_epoch_seq, bool) or not isinstance(initial_epoch_seq, int):
            raise EpochControllerError("invalid_initial_epoch_seq")

        allocator = (
            _default_epoch_id_allocator
            if epoch_id_allocator is None
            else epoch_id_allocator
        )
        if not callable(allocator):
            raise EpochControllerError("invalid_epoch_id_allocator")

        self._epoch_id_allocator = allocator
        self._allocated_epoch_ids: set[str] = set()
        initial_epoch_id = self._allocate_fresh_epoch_id()
        self._allocated_epoch_ids.add(initial_epoch_id)
        self._current = ExecutionFence(
            asr_job_id=asr_job_id,
            epoch_id=initial_epoch_id,
            epoch_seq=initial_epoch_seq,
            speech_segment_id=speech_segment_id,
        )
        self._prepared_epoch_id: str | None = None
        self._state = EpochControllerState.ACTIVE

    @property
    def current(self) -> ExecutionFence:
        """The currently active local acoustic execution lineage."""

        return self._current

    @property
    def prepared_epoch_id(self) -> str | None:
        """Reserved successor ID; it is not an admitted fence until activation."""

        return self._prepared_epoch_id

    @property
    def state(self) -> EpochControllerState:
        return self._state

    def request_rollover(self) -> None:
        if self._state is not EpochControllerState.ACTIVE:
            raise self._transition_error("request_rollover")
        self._state = EpochControllerState.ROLLOVER_REQUESTED

    def prepare_successor(self) -> str:
        if self._state is not EpochControllerState.ROLLOVER_REQUESTED:
            raise self._transition_error("prepare_successor")

        epoch_id = self._allocate_fresh_epoch_id()
        self._allocated_epoch_ids.add(epoch_id)
        self._prepared_epoch_id = epoch_id
        self._state = EpochControllerState.SUCCESSOR_PREPARED
        return epoch_id

    def activate_successor(self) -> ExecutionFence:
        if self._state is not EpochControllerState.SUCCESSOR_PREPARED:
            raise self._transition_error("activate_successor")
        epoch_id = self._prepared_epoch_id
        if epoch_id is None:
            raise EpochControllerError("prepared_successor_missing")

        self._current = ExecutionFence(
            asr_job_id=self._current.asr_job_id,
            epoch_id=epoch_id,
            epoch_seq=self._current.epoch_seq + 1,
            speech_segment_id=self._current.speech_segment_id,
        )
        self._prepared_epoch_id = None
        self._state = EpochControllerState.ACTIVE
        return self._current

    def classify(self, candidate: ExecutionFence) -> FenceOutcome:
        """Classify a local candidate using the accepted A2 fence comparator."""

        if not isinstance(candidate, ExecutionFence):
            raise EpochControllerError("invalid_candidate_fence")
        return compare_execution_fence(candidate, self._current)

    @staticmethod
    def _validate_non_empty_string(value: object, code: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise EpochControllerError(code)

    def _allocate_fresh_epoch_id(self) -> str:
        epoch_id = self._epoch_id_allocator()
        self._validate_non_empty_string(epoch_id, "invalid_epoch_id")
        if epoch_id in self._allocated_epoch_ids:
            raise EpochControllerError("epoch_id_reused")
        return epoch_id

    def _transition_error(self, operation: str) -> EpochControllerError:
        return EpochControllerError(
            "invalid_transition",
            f"invalid_transition:{self._state.value}:{operation}",
        )
