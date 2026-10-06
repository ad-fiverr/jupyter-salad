"""Per-WebSocket ownership for Qwen's local rolling acoustic lifecycle."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .asr_epoch_controller import LocalAcousticEpochController
from .asr_pcm_handoff import HandoffDrainResult, LocalPCMHandoffCoordinator, PCMHandoffError
from .asr_qwen_epoch_lifecycle import (
    QwenEpochLifecycle,
    QwenEpochLifecycleError,
    QwenEpochLifecycleState,
)
from .config import Settings
from .qwen_streaming import QwenStreamingRuntime, StreamingError, validate_candidate_event


ASR_LAB_LOCAL_INITIAL_EPOCH_SEQ = 0
CandidateEventSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class _SourceLifecycle:
    source: str
    lifecycle: QwenEpochLifecycle | None = None
    public_stream_id: str | None = None
    disposed: bool = False


class QwenServiceLifecycleRegistry:
    """Own one bounded A6 lifecycle per active source on a WebSocket."""

    def __init__(
        self,
        *,
        runtime: QwenStreamingRuntime,
        settings: Settings,
        connection_id: str,
        event_sink: CandidateEventSink,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.connection_id = connection_id
        self.event_sink = event_sink
        self._sources: dict[str, _SourceLifecycle] = {}
        self._closed = False

    def has_active_source(self, source: str) -> bool:
        entry = self._sources.get(source)
        return bool(entry is not None and not entry.disposed)

    def source_observability_snapshot(self, source: str) -> dict[str, Any] | None:
        """Capture bounded lifecycle evidence before source cleanup detaches it."""
        entry = self._sources.get(source)
        if entry is None or entry.disposed or entry.lifecycle is None:
            return None
        return {
            "qwen_public_stream_id": entry.public_stream_id,
            "qwen_local_stream_id": entry.lifecycle.local_stream_id,
            "qwen_epoch_observability": entry.lifecycle.observability_snapshot(),
        }

    async def open_source(
        self,
        *,
        source: str,
        language: str | None = None,
        context: str = "",
        model_chunk_ms: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise StreamingError("stream_connection_closed")
        if self.has_active_source(source):
            raise StreamingError("stream_already_active")

        entry = _SourceLifecycle(source=source)
        self._sources[source] = entry

        async def forward_candidate(candidate: dict[str, Any]) -> None:
            lifecycle = entry.lifecycle
            local_stream_id = candidate.get("stream_id")
            if (
                entry.disposed
                or self._sources.get(source) is not entry
                or lifecycle is None
                or not isinstance(local_stream_id, str)
                or local_stream_id != lifecycle.local_stream_id
                or not isinstance(entry.public_stream_id, str)
            ):
                return
            await self.event_sink(self._project_candidate(entry, candidate, local_stream_id))

        job_id = f"asr-lab-job-{uuid4().hex}"
        segment_id = f"asr-lab-segment-{uuid4().hex}"
        controller = LocalAcousticEpochController(
            asr_job_id=job_id,
            speech_segment_id=segment_id,
            initial_epoch_seq=ASR_LAB_LOCAL_INITIAL_EPOCH_SEQ,
        )
        transition_samples = max(
            1,
            math.ceil(float(self.settings.max_chunk_seconds) * 16_000),
        )
        max_retained_samples = (
            math.ceil(float(self.runtime.max_stream_seconds) * 16_000)
            + transition_samples
        )
        handoff = LocalPCMHandoffCoordinator(
            epoch_controller=controller,
            replay_overlap_samples=0,
            max_retained_samples=max_retained_samples,
            max_transition_samples=transition_samples,
        )
        lifecycle = QwenEpochLifecycle(
            epoch_controller=controller,
            pcm_handoff=handoff,
            qwen_runtime=self.runtime,
            connection_id=self.connection_id,
            source=source,
            language=language,
            context=context,
            chunk_size_ms=model_chunk_ms,
            request_id=request_id,
            event_sink=forward_candidate,
        )
        entry.lifecycle = lifecycle
        try:
            opened = await lifecycle.initialize()
        except Exception as exc:
            entry.disposed = True
            self._sources.pop(source, None)
            try:
                await lifecycle.dispose()
            except Exception:
                pass
            raise self._as_streaming_error(exc) from exc

        local_stream_id = opened.get("stream_id")
        if not isinstance(local_stream_id, str) or not local_stream_id:
            entry.disposed = True
            self._sources.pop(source, None)
            await lifecycle.dispose()
            raise StreamingError("invalid_stream_started_event")
        entry.public_stream_id = local_stream_id
        response = dict(opened)
        response.pop("epoch_fence", None)
        response["qwen_local_stream_id"] = local_stream_id
        response["qwen_epoch_observability"] = lifecycle.observability_snapshot()
        return response

    async def open_default_source(self, source: str) -> dict[str, Any] | None:
        if self.has_active_source(source):
            return None
        return await self.open_source(source=source)

    async def submit_pcm(self, *, source: str, pcm16le: bytes) -> None:
        entry = self._sources.get(source)
        if entry is None or entry.disposed or entry.lifecycle is None:
            raise StreamingError("stream_not_started")
        lifecycle = entry.lifecycle
        try:
            if lifecycle.state is QwenEpochLifecycleState.HANDOFF_PENDING:
                # A later source frame is one explicit, bounded opportunity
                # to finish an earlier handoff. A still-pending frame remains
                # owned by A6.2 as TRANSITION when submit_pcm is called below.
                await lifecycle.drain_pending_handoff()
            await lifecycle.submit_pcm(pcm16le)
        except Exception as exc:
            raise self._as_streaming_error(exc) from exc

    async def rollover_source(self, source: str) -> HandoffDrainResult:
        """Drive one internal manual/test rollover on an active logical source."""
        entry = self._sources.get(source)
        if entry is None or entry.disposed or entry.lifecycle is None:
            raise StreamingError("stream_not_started")
        try:
            return await entry.lifecycle.rollover()
        except Exception as exc:
            raise self._as_streaming_error(exc) from exc

    async def finish(self, *, source: str, request_id: str | None = None) -> dict[str, Any] | None:
        entry = self._sources.get(source)
        if entry is None or entry.disposed or entry.lifecycle is None:
            return None
        lifecycle = entry.lifecycle
        local_stream_id = lifecycle.local_stream_id
        if lifecycle.state is QwenEpochLifecycleState.HANDOFF_PENDING:
            try:
                drained = await lifecycle.drain_pending_handoff()
            except Exception as exc:
                raise self._as_streaming_error(exc) from exc
            if not drained.completed:
                raise StreamingError("stream_handoff_incomplete")
        try:
            result = await lifecycle.finish()
        except Exception as exc:
            raise self._as_streaming_error(exc) from exc

        if result is None:
            self._detach(entry)
            return None
        if not isinstance(local_stream_id, str) or result.get("stream_id") != local_stream_id:
            self._detach(entry)
            raise StreamingError("stream_fenced")
        candidate = self._project_candidate(entry, result, local_stream_id)
        if request_id is not None:
            candidate["request_id"] = request_id
            candidate["client_request_id"] = request_id
        self._detach(entry)
        return candidate

    async def abort_source(self, source: str) -> None:
        entry = self._sources.get(source)
        if entry is None:
            return
        self._detach(entry)
        lifecycle = entry.lifecycle
        if lifecycle is not None:
            try:
                await lifecycle.dispose()
            except (StreamingError, QwenEpochLifecycleError, PCMHandoffError):
                raise
            except Exception as exc:
                raise self._as_streaming_error(exc) from exc

    async def close_connection(self) -> None:
        if self._closed:
            return
        self._closed = True
        entries = list(self._sources.values())
        self._sources.clear()
        for entry in entries:
            entry.disposed = True
        try:
            for entry in entries:
                if entry.lifecycle is not None:
                    try:
                        await entry.lifecycle.dispose()
                    except Exception:
                        # Runtime connection closure below is the final cleanup
                        # boundary for any already-failed local session.
                        pass
        finally:
            await self.runtime.close_connection(self.connection_id)

    def _project_candidate(
        self,
        entry: _SourceLifecycle,
        candidate: dict[str, Any],
        local_stream_id: str,
    ) -> dict[str, Any]:
        event = candidate.get("event")
        if event not in {"partial_candidate", "final_candidate"}:
            raise StreamingError("invalid_candidate_event")
        projected = dict(validate_candidate_event(candidate, event))
        projected["qwen_local_stream_id"] = local_stream_id
        projected["stream_id"] = entry.public_stream_id
        lifecycle = entry.lifecycle
        if lifecycle is not None:
            projected["qwen_epoch_observability"] = lifecycle.observability_snapshot()
        return projected

    def _detach(self, entry: _SourceLifecycle) -> None:
        entry.disposed = True
        if self._sources.get(entry.source) is entry:
            self._sources.pop(entry.source, None)

    @staticmethod
    def _as_streaming_error(exc: Exception) -> StreamingError:
        if isinstance(exc, StreamingError):
            return exc
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code:
            return StreamingError(code)
        return StreamingError("stream_lifecycle_failed")
