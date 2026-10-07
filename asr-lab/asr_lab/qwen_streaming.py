"""Bounded, connection-owned Qwen streaming sessions over a private worker RPC."""
from __future__ import annotations

import asyncio
import base64
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Awaitable, Callable, Protocol

from .qwen_scheduler import (
    DecodeJob,
    DispatchResult,
    QwenDecodeScheduler,
    RequestKey,
    SchedulerError,
)

STREAMING_CLASS = "accumulated-audio-pseudostreaming"
ALLOWED_CHUNK_MS = (50, 100, 150, 200, 250, 500, 1000, 2000)
AUDIO_PUSH_INTERVAL_MS = 100
QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian",
    "hu": "Hungarian", "mk": "Macedonian",
}


class WorkerRPC(Protocol):
    async def start(self, **options: Any) -> dict[str, Any]: ...
    async def request(self, operation: str, **payload: Any) -> dict[str, Any]: ...
    async def close(self) -> None: ...


ProcessedAudioObserver = Callable[[str, int], None]
CandidateResultFilter = Callable[[dict[str, Any]], dict[str, Any] | None]
LocalStaleRejectObserver = Callable[[str, str], None]


class StreamingError(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


def validate_candidate_event(payload: dict[str, Any], expected_event: str) -> dict[str, Any]:
    """Enforce that experimental Qwen output can never be serialized as truth."""
    expected_final = expected_event == "final_candidate"
    if (
        expected_event not in {"partial_candidate", "final_candidate"}
        or payload.get("event") != expected_event
        or payload.get("truth_status") != "candidate_only"
        or payload.get("provisional") is not True
        or payload.get("final") is not expected_final
        or payload.get("type") == "transcript"
        or (expected_final and payload.get("candidate_only") is not True)
    ):
        raise StreamingError("invalid_candidate_event")
    return payload


@dataclass
class StreamSession:
    owner_connection_id: str
    source: str
    stream_id: str
    request_id: str | None
    language: str
    chunk_size_ms: int
    context: str
    started_at: float
    last_activity_at: float
    max_stream_seconds: float
    event_sink: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    candidate_result_filter: CandidateResultFilter | None = None
    processed_audio_observer: ProcessedAudioObserver | None = None
    event_queue: asyncio.Queue[dict[str, Any]] = dataclass_field(default_factory=lambda: asyncio.Queue(maxsize=32))
    first_audio_at: float | None = None
    stream_init_ms: float | None = None
    stream_state_init_wall_ms: float | None = None
    decode_wall_samples: list[float] = dataclass_field(default_factory=list)
    first_decode_ready_ms: float | None = None
    first_scheduler_wait_ms: float | None = None
    first_decode_start_ms: float | None = None
    audio_samples: int = 0
    revision: int = 0
    partial_count: int = 0
    current_text: str = ""
    current_language: str | None = None
    first_partial_at: float | None = None
    last_partial_at: float | None = None
    cumulative_decode_wall_ms: float = 0.0
    scheduler_metrics: deque[dict[str, Any]] = dataclass_field(default_factory=lambda: deque(maxlen=256))
    latest_scheduler_key: RequestKey | None = None


def _decode_ordinal_metrics(session: StreamSession) -> dict[str, float | None]:
    samples = session.decode_wall_samples
    first = samples[0] if samples else None
    second = samples[1] if len(samples) > 1 else None
    steady = samples[2:]
    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.9999)))
        return round(ordered[index], 3)
    steady_p50 = percentile(steady, 0.50)
    steady_p95 = percentile(steady, 0.95)
    return {
        "EPOCH_FIRST_DECODE_WALL_MS": round(first, 3) if first is not None else None,
        "EPOCH_SECOND_DECODE_WALL_MS": round(second, 3) if second is not None else None,
        "EPOCH_STEADY_DECODE_WALL_P50_MS": steady_p50,
        "EPOCH_STEADY_DECODE_WALL_P95_MS": steady_p95,
        "FIRST_DECODE_WALL_MS": round(first, 3) if first is not None else None,
        "SECOND_DECODE_WALL_MS": round(second, 3) if second is not None else None,
        "STEADY_DECODE_WALL_P50_MS": steady_p50,
        "STEADY_DECODE_WALL_P95_MS": steady_p95,
    }


def token_revision_metrics(previous: str, current: str) -> tuple[float | None, float | None]:
    """Return changed/removed old-token fraction and unchanged old-prefix fraction.

    Appended new tokens are not counted as revisions. A token is stable only
    while it remains in the exact common prefix of two complete transcript states.
    """
    old_tokens = previous.split()
    new_tokens = current.split()
    if not old_tokens:
        return None, None
    stable_prefix = 0
    for old, new in zip(old_tokens, new_tokens):
        if old != new:
            break
        stable_prefix += 1
    stable_fraction = stable_prefix / len(old_tokens)
    return 1.0 - stable_fraction, stable_fraction


class QwenStreamingRuntime:
    """Owns the Qwen process and per-(connection, source) streaming states."""

    def __init__(
        self,
        *,
        worker: WorkerRPC,
        model_id: str,
        model_revision: str,
        gpu_memory_utilization: float,
        max_active_sessions: int,
        max_stream_seconds: float,
        session_idle_ttl_seconds: float,
        max_context_chars: int,
        default_chunk_ms: int,
        default_language: str,
        unfixed_chunk_num: int,
        unfixed_token_num: int,
        max_pending_jobs: int = 24,
        max_backlog_chunks: int = 4,
    ) -> None:
        self.worker = worker
        self.model_id = model_id
        self.model_revision = model_revision
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_active_sessions = max_active_sessions
        self.max_stream_seconds = max_stream_seconds
        self.session_idle_ttl_seconds = session_idle_ttl_seconds
        self.max_context_chars = max_context_chars
        self.default_chunk_ms = default_chunk_ms
        self.default_language = default_language
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.sessions: dict[tuple[str, str], StreamSession] = {}
        self.sessions_by_stream_id: dict[tuple[str, str], StreamSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._sweeper: asyncio.Task[None] | None = None
        self._closing = False
        self._worker_metrics: list[dict[str, Any]] = []
        self.scheduler = QwenDecodeScheduler(
            execute=self._execute_scheduled,
            on_result=self._scheduled_result,
            on_fault=self._scheduler_fault,
            max_pending_jobs=max_pending_jobs,
            max_backlog_chunks=max_backlog_chunks,
            max_active_streams=max_active_sessions,
        )

    @property
    def ready(self) -> bool:
        return bool(getattr(self.worker, "ready", False)) and not self._closing

    @property
    def worker_metrics(self) -> list[dict[str, Any]]:
        return list(self._worker_metrics)

    @property
    def runtime_provenance(self) -> dict[str, Any] | None:
        if not self._worker_metrics:
            return None
        provenance = self._worker_metrics[0].get("runtime_provenance")
        return dict(provenance) if isinstance(provenance, dict) else None

    @property
    def queued_jobs(self) -> int:
        return self.scheduler.pending_jobs

    @property
    def _capacity_progress_generation(self) -> int:
        return self.scheduler._capacity_progress_generation

    async def _wait_for_capacity_progress(self, after_generation: int, timeout_seconds: float) -> bool:
        return await self.scheduler._wait_for_capacity_progress(after_generation, timeout_seconds)

    @property
    def active_sessions(self) -> int:
        return len(self.sessions)

    async def start(self) -> None:
        report = await self.worker.start(
            model_id=self.model_id,
            model_revision=self.model_revision,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_active_sessions=self.max_active_sessions,
            warmup_chunk_ms=self.default_chunk_ms,
            unfixed_chunk_num=self.unfixed_chunk_num,
            unfixed_token_num=self.unfixed_token_num,
        )
        self._worker_metrics = [report] if report else []
        await self.scheduler.start()
        self._sweeper = asyncio.create_task(self._expire_loop(), name="qwen-stream-expiry")

    async def open_session(
        self,
        *,
        connection_id: str,
        source: str,
        language: str | None = None,
        context: str = "",
        model_chunk_ms: int | None = None,
        chunk_size_ms: int | None = None,
        request_id: str | None = None,
        event_sink: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        candidate_result_filter: CandidateResultFilter | None = None,
        processed_audio_observer: ProcessedAudioObserver | None = None,
        local_stale_reject_observer: LocalStaleRejectObserver | None = None,
    ) -> dict[str, Any]:
        if not self.ready:
            raise StreamingError("backend_not_ready")
        language_value = self.default_language if language is None else language
        for supplied_chunk in (model_chunk_ms, chunk_size_ms):
            if supplied_chunk is not None and (
                isinstance(supplied_chunk, bool) or not isinstance(supplied_chunk, int)
                or supplied_chunk not in ALLOWED_CHUNK_MS
            ):
                raise StreamingError("invalid_stream_chunk")
        if model_chunk_ms is not None and chunk_size_ms is not None and model_chunk_ms != chunk_size_ms:
            raise StreamingError("conflicting_model_chunk")
        selected_chunk_ms = model_chunk_ms if model_chunk_ms is not None else chunk_size_ms
        chunk_ms = self.default_chunk_ms if selected_chunk_ms is None else selected_chunk_ms
        if (isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int)
                or chunk_ms not in ALLOWED_CHUNK_MS):
            raise StreamingError("invalid_stream_chunk")
        if len(context) > self.max_context_chars:
            raise StreamingError("invalid_context")
        key = (connection_id, source)
        async with self._sessions_lock:
            await self._expire_locked(time.perf_counter())
            if key in self.sessions:
                raise StreamingError("stream_already_active")
            if len(self.sessions) >= self.max_active_sessions:
                raise StreamingError("stream_capacity_exceeded")
            stream_id = uuid.uuid4().hex
            now = time.perf_counter()
            try:
                init_started = time.perf_counter()
                init_reply = await self.worker.request(
                    "init",
                    stream_id=stream_id,
                    chunk_size_sec=chunk_ms / 1000.0,
                    language=QWEN_LANGUAGE_NAMES.get(language_value),
                    context=context,
                    unfixed_chunk_num=self.unfixed_chunk_num,
                    unfixed_token_num=self.unfixed_token_num,
                )
                init_finished = time.perf_counter()
                state_init_ms = init_reply.get("stream_state_init_wall_ms") if isinstance(init_reply, dict) else None
                await self.scheduler.register(
                    connection_id,
                    stream_id,
                    chunk_ms,
                    local_stale_reject_observer=local_stale_reject_observer,
                )
            except Exception:
                try:
                    await self.worker.request("close", stream_id=stream_id)
                except Exception:
                    pass
                raise
            session = StreamSession(
                owner_connection_id=connection_id,
                source=source,
                stream_id=stream_id,
                request_id=request_id,
                language=language_value,
                chunk_size_ms=chunk_ms,
                context=context,
                started_at=now,
                last_activity_at=now,
                max_stream_seconds=self.max_stream_seconds,
                event_sink=event_sink,
                candidate_result_filter=candidate_result_filter,
                processed_audio_observer=processed_audio_observer,
                stream_init_ms=(init_finished - init_started) * 1000.0,
                stream_state_init_wall_ms=float(state_init_ms) if isinstance(state_init_ms, (int, float)) and not isinstance(state_init_ms, bool) else None,
            )
            self.sessions[key] = session
            self.sessions_by_stream_id[(connection_id, stream_id)] = session
        return {
            "event": "stream_started",
            "backend": "qwen3_asr",
            "stream_id": stream_id,
            "request_id": request_id,
            "transcript_mode": "STREAMING_PARTIALS",
            "streaming_class": STREAMING_CLASS,
            "audio_push_interval_ms": AUDIO_PUSH_INTERVAL_MS,
            "model_chunk_ms": chunk_ms,
            "model_decode_chunk_ms": chunk_ms,
            "effective_max_backlog_ms": self.scheduler.max_backlog_chunks * chunk_ms,
            "qwen_max_backlog_chunks": self.scheduler.max_backlog_chunks,
            "FIRST_STREAM_INIT_MS": round(session.stream_init_ms or 0.0, 3),
            "FIRST_STREAM_STATE_INIT_WALL_MS": round(session.stream_state_init_wall_ms, 3) if session.stream_state_init_wall_ms is not None else None,
            "FIRST_STREAM_INIT_RPC_OVERHEAD_MS": round(max(0.0, session.stream_init_ms - session.stream_state_init_wall_ms), 3) if session.stream_init_ms is not None and session.stream_state_init_wall_ms is not None else None,
            "language": language_value,
        }

    async def push_audio(
        self, *, connection_id: str, source: str, pcm16le: bytes,
        allow_capacity_wait: bool = False,
    ) -> dict[str, Any] | None:
        key = (connection_id, source)
        session = self.sessions.get(key)
        if session is None or session.owner_connection_id != connection_id:
            raise StreamingError("stream_not_started")
        if not pcm16le or len(pcm16le) % 2:
            raise StreamingError("invalid_audio")
        samples = len(pcm16le) // 2
        if session.audio_samples + samples > session.max_stream_seconds * 16_000:
            await self.close_session(connection_id=connection_id, source=source)
            raise StreamingError("stream_duration_limit")
        received_at = time.perf_counter()
        try:
            await self.scheduler.append_pcm(
                connection_id,
                session.stream_id,
                pcm16le,
                received_at=received_at,
                allow_capacity_wait=allow_capacity_wait,
            )
        except SchedulerError as exc:
            if not (allow_capacity_wait and exc.code == "stream_scheduler_capacity_wait"):
                await self.close_session(connection_id=connection_id, source=source)
            raise StreamingError(exc.code, exc.details) from exc
        session.audio_samples += samples
        session.last_activity_at = received_at
        return None

    async def _execute_scheduled(self, key: RequestKey, job: DecodeJob) -> dict[str, Any]:
        session = self.sessions_by_stream_id.get((key.connection_id, key.stream_id))
        if session is None:
            raise StreamingError("stream_fenced")
        session.latest_scheduler_key = key
        scheduler_key = {
            "connection_id": key.connection_id,
            "stream_id": key.stream_id,
            "scheduler_revision": key.scheduler_revision,
        }
        payload: dict[str, Any] = {
            "stream_id": key.stream_id,
            "connection_id": key.connection_id,
            "scheduler_revision": key.scheduler_revision,
            "scheduler_key": scheduler_key,
        }
        if job.kind == "push":
            payload["pcm16le_base64"] = base64.b64encode(job.pcm16le).decode("ascii")
        reply = await self.worker.request(job.kind, **payload)
        if reply.get("scheduler_key") != scheduler_key:
            raise StreamingError("invalid_worker_response")
        delta = reply.get("decode_steps_delta", 0)
        if isinstance(delta, bool) or not isinstance(delta, int) or delta not in (0, 1):
            raise StreamingError("invalid_worker_response")
        return reply

    async def _scheduled_result(
        self, key: RequestKey, job: DecodeJob, reply: dict[str, Any], metrics: dict[str, Any],
    ) -> None:
        if not self.scheduler.is_current(key):
            self._observe_local_stale_reject(job, key, "runtime_scheduler_key_stale")
            return
        session = self.sessions_by_stream_id.get((key.connection_id, key.stream_id))
        if session is None or session.latest_scheduler_key != key:
            self._observe_local_stale_reject(job, key, "runtime_session_key_stale")
            return
        ready = time.perf_counter()
        stream_queue = self.scheduler.streams.get((key.connection_id, key.stream_id))
        if session.first_audio_at is None and stream_queue is not None:
            session.first_audio_at = stream_queue.first_audio_at
        stream_metrics = self.scheduler.stream_snapshot(key.connection_id, key.stream_id) or {}
        session.last_activity_at = ready
        decode_wall_ms = max(0.0, float(reply.get("decode_wall_ms", 0.0)))
        step_delta = metrics.get("qwen_decode_steps_delta", 0)
        if step_delta > 0:
            session.cumulative_decode_wall_ms += decode_wall_ms
            session.decode_wall_samples.append(decode_wall_ms)
            if session.first_decode_ready_ms is None:
                origin = metrics.get("first_audio_at")
                ready_at = metrics.get("job_ready_at")
                dispatch_at = metrics.get("dispatch_at")
                if origin is not None and ready_at is not None and dispatch_at is not None:
                    session.first_decode_ready_ms = max(0.0, (ready_at - origin) * 1000.0)
                    session.first_scheduler_wait_ms = max(0.0, (dispatch_at - ready_at) * 1000.0)
                    session.first_decode_start_ms = max(0.0, (dispatch_at - origin) * 1000.0)
        session.scheduler_metrics.append(dict(metrics))
        text = reply.get("text")
        if not isinstance(text, str):
            raise StreamingError("invalid_worker_response")
        if job.kind == "push" and session.processed_audio_observer is not None:
            session.processed_audio_observer(session.stream_id, job.cursor_end_samples)
        if not text.strip() or text == session.current_text:
            return None

        revision_rate, stability = token_revision_metrics(session.current_text, text)
        session.revision += 1
        session.partial_count += 1
        if session.first_partial_at is None:
            session.first_partial_at = ready
        interval_ms = (
            max(0.0, (ready - session.last_partial_at) * 1000)
            if session.last_partial_at is not None else None
        )
        session.last_partial_at = ready
        session.current_text = text
        language = reply.get("language")
        if isinstance(language, str) and language:
            session.current_language = language
        cursor_ms = job.cursor_end_samples * 1000.0 / 16_000
        duration_ms = max(cursor_ms, 1.0)
        event = {
            "event": "partial_candidate",
            "schema_version": 2,
            "backend": "qwen3_asr",
            "transcript_mode": "STREAMING_PARTIALS",
            "streaming_class": STREAMING_CLASS,
            "text": text,
            "language": session.current_language,
            "final": False,
            "provisional": True,
            "truth_status": "candidate_only",
            "replace": True,
            "stream_id": session.stream_id,
            "source": session.source,
            "speaker": "you" if session.source == "mic" else "them",
            "revision": session.revision,
            "audio_cursor_ms": round(cursor_ms, 2),
            "QWEN_NEW_AUDIO_MS": (
                round((len(job.pcm16le) // 2) * 1000.0 / 16_000, 3)
                if job.kind == "push" else None
            ),
            "EPOCH_AUDIO_ACCUMULATED_MS": round(cursor_ms, 3),
            # qwen-asr's internal accumulated tensor is not exposed through a
            # version-stable read-only API, so do not infer it from the cursor.
            "QWEN_AUDIO_ACCUM_MS": None,
            "request_id": session.request_id,
            "PARTIAL_COUNT": session.partial_count,
            "SERVER_FIRST_PARTIAL_MS": round(((session.first_partial_at if session.first_partial_at is not None else ready) - (session.first_audio_at if session.first_audio_at is not None else session.started_at)) * 1000, 2),
            "FIRST_PARTIAL_MS": round(((session.first_partial_at if session.first_partial_at is not None else ready) - (session.first_audio_at if session.first_audio_at is not None else session.started_at)) * 1000, 2),
            "EFFECTIVE_MAX_BACKLOG_MS": session.chunk_size_ms * self.scheduler.max_backlog_chunks,
            "QWEN_MAX_BACKLOG_CHUNKS": self.scheduler.max_backlog_chunks,
            "QWEN_DECODE_SLO_TARGET_MS": 100,
            "QWEN_DECODE_SLO_VIOLATION": None if step_delta <= 0 else bool(metrics.get("qwen_decode_slo_violation", False)),
            **_decode_ordinal_metrics(session),
            "PARTIAL_UPDATE_INTERVAL_MS": round(interval_ms, 2) if interval_ms is not None else None,
            "PARTIAL_REVISION_RATE": round(revision_rate, 4) if revision_rate is not None else None,
            "PARTIAL_STABILITY": round(stability, 4) if stability is not None else None,
            "SERVER_CHUNK_TO_PARTIAL_MS": round((ready - job.ready_at) * 1000, 2),
            "QWEN_DECODE_CALL_WALL_MS": round(decode_wall_ms, 2) if step_delta > 0 else None,
            "QWEN_CUMULATIVE_DECODE_WALL_MS": round(session.cumulative_decode_wall_ms, 2),
            "QWEN_STREAM_RTF": round(session.cumulative_decode_wall_ms / duration_ms, 4),
            "AUDIO_DURATION_MS": round(cursor_ms, 2),
            "MODEL_DECODE_CHUNK_MS": session.chunk_size_ms,
            "FIRST_STREAM_INIT_MS": round(session.stream_init_ms or 0.0, 3),
            "FIRST_STREAM_STATE_INIT_WALL_MS": round(session.stream_state_init_wall_ms, 3) if session.stream_state_init_wall_ms is not None else None,
            "FIRST_STREAM_INIT_RPC_OVERHEAD_MS": round(max(0.0, session.stream_init_ms - session.stream_state_init_wall_ms), 3) if session.stream_init_ms is not None and session.stream_state_init_wall_ms is not None else None,
            "FIRST_AUDIO_TO_FIRST_DECODE_READY_MS": round(session.first_decode_ready_ms, 3) if session.first_decode_ready_ms is not None else None,
            "FIRST_SCHEDULER_WAIT_MS": round(session.first_scheduler_wait_ms, 3) if session.first_scheduler_wait_ms is not None else None,
            "FIRST_AUDIO_TO_FIRST_DECODE_START_MS": round(session.first_decode_start_ms, 3) if session.first_decode_start_ms is not None else None,
            "AUDIO_PUSH_INTERVAL_MS": AUDIO_PUSH_INTERVAL_MS,
            "QWEN_SCHEDULER_WAIT_MS": metrics.get("qwen_scheduler_wait_ms"),
            "QWEN_SCHEDULER_REVISION": key.scheduler_revision,
            "QWEN_DECODE_BACKLOG_MS": metrics.get("qwen_decode_backlog_ms"),
            "QWEN_DECODE_BACKLOG_MAX_MS": stream_metrics.get("scheduler_max_backlog_audio_ms"),
            "STREAM_LAG_MS": metrics.get("stream_lag_ms"),
            "STREAM_LAG_MAX_MS": stream_metrics.get("stream_lag_max_ms"),
            "QWEN_DECODE_STEPS_DELTA": metrics.get("qwen_decode_steps_delta", 0),
            "PENDING_DECODE_COUNT": metrics.get("pending_decode_count"),
            "ACTIVE_STREAM_COUNT": self.scheduler.active_stream_count,
            "DECODE_OVERRUN": bool(metrics.get("decode_overrun", False)),
        }
        event = self._filter_candidate_result(session, event, "partial_candidate")
        if event is None:
            return None
        if session.event_sink is not None:
            await session.event_sink(event)
        else:
            session.event_queue.put_nowait(event)

    async def finish(
        self, *, connection_id: str, source: str, server_eos_at: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        key = (connection_id, source)
        session = self.sessions.get(key)
        if session is None or session.owner_connection_id != connection_id:
            raise StreamingError("stream_not_started")
        eos_at = time.perf_counter() if server_eos_at is None else server_eos_at
        try:
            dispatch = await self.scheduler.finish_stream(connection_id, session.stream_id)
            reply = dispatch.reply
        except Exception as exc:
            await self.close_session(connection_id=connection_id, source=source)
            code = exc.code if isinstance(exc, (StreamingError, SchedulerError)) else "stream_worker_failed"
            raise StreamingError(code) from exc
        ready = time.perf_counter()
        if self.sessions.get(key) is not session:
            raise StreamingError("stream_not_started")
        final_text = reply.get("text")
        if not isinstance(final_text, str):
            await self.close_session(connection_id=connection_id, source=source)
            raise StreamingError("invalid_worker_response")
        revision_rate, stability = token_revision_metrics(session.current_text, final_text)
        if final_text != session.current_text:
            session.revision += 1
        session.current_text = final_text
        language = reply.get("language")
        if isinstance(language, str) and language:
            session.current_language = language
        finish_decode_ms = max(0.0, float(reply.get("decode_wall_ms", 0.0)))
        finish_steps = dispatch.metrics.get("qwen_decode_steps_delta", 0)
        if finish_steps > 0:
            session.cumulative_decode_wall_ms += finish_decode_ms
            session.decode_wall_samples.append(finish_decode_ms)
            if session.first_decode_ready_ms is None:
                origin = dispatch.metrics.get("first_audio_at")
                ready_at = dispatch.metrics.get("job_ready_at")
                dispatch_at = dispatch.metrics.get("dispatch_at")
                if origin is not None and ready_at is not None and dispatch_at is not None:
                    session.first_decode_ready_ms = max(0.0, (ready_at - origin) * 1000.0)
                    session.first_scheduler_wait_ms = max(0.0, (dispatch_at - ready_at) * 1000.0)
                    session.first_decode_start_ms = max(0.0, (dispatch_at - origin) * 1000.0)
        cursor_ms = session.audio_samples * 1000.0 / 16_000
        stream_metrics = self.scheduler.stream_snapshot(connection_id, session.stream_id) or {}
        session.scheduler_metrics.append(dict(dispatch.metrics))
        event = {
            "event": "final_candidate",
            "schema_version": 2,
            "backend": "qwen3_asr",
            "transcript_mode": "STREAMING_PARTIALS",
            "streaming_class": STREAMING_CLASS,
            "text": final_text,
            "language": session.current_language,
            "final": True,
            "provisional": True,
            "candidate_only": True,
            "truth_status": "candidate_only",
            "replace": True,
            "stream_id": session.stream_id,
            "source": session.source,
            "speaker": "you" if session.source == "mic" else "them",
            "revision": session.revision,
            "audio_cursor_ms": round(cursor_ms, 2),
            "QWEN_NEW_AUDIO_MS": None,
            "EPOCH_AUDIO_ACCUMULATED_MS": round(cursor_ms, 3),
            "QWEN_AUDIO_ACCUM_MS": None,
            "request_id": request_id or session.request_id,
            "client_request_id": request_id,
            "AUDIO_DURATION_MS": round(cursor_ms, 2),
            "PARTIAL_COUNT": session.partial_count,
            "FIRST_PARTIAL_MS": round(
                (session.first_partial_at - session.first_audio_at) * 1000, 2
            ) if session.first_partial_at is not None and session.first_audio_at is not None else None,
            "PARTIAL_REVISION_RATE": round(revision_rate, 4) if revision_rate is not None else None,
            "PARTIAL_STABILITY": round(stability, 4) if stability is not None else None,
            "FINALIZATION_AFTER_SERVER_EOS_MS": round(max(0.0, (ready - eos_at) * 1000), 2),
            "SERVER_EOS_TO_FINAL_CANDIDATE_MS": round(max(0.0, (ready - eos_at) * 1000), 2),
            "QWEN_FINISH_DECODE_WALL_MS": round(finish_decode_ms, 2),
            "QWEN_CUMULATIVE_DECODE_WALL_MS": round(session.cumulative_decode_wall_ms, 2),
            "QWEN_STREAM_RTF": round(session.cumulative_decode_wall_ms / max(cursor_ms, 1.0), 4),
            "MODEL_DECODE_CHUNK_MS": session.chunk_size_ms,
            "EFFECTIVE_MAX_BACKLOG_MS": session.chunk_size_ms * self.scheduler.max_backlog_chunks,
            "QWEN_MAX_BACKLOG_CHUNKS": self.scheduler.max_backlog_chunks,
            "FIRST_STREAM_INIT_MS": round(session.stream_init_ms or 0.0, 3),
            "FIRST_STREAM_STATE_INIT_WALL_MS": round(session.stream_state_init_wall_ms, 3) if session.stream_state_init_wall_ms is not None else None,
            "FIRST_STREAM_INIT_RPC_OVERHEAD_MS": round(max(0.0, session.stream_init_ms - session.stream_state_init_wall_ms), 3) if session.stream_init_ms is not None and session.stream_state_init_wall_ms is not None else None,
            "FIRST_AUDIO_TO_FIRST_DECODE_READY_MS": round(session.first_decode_ready_ms, 3) if session.first_decode_ready_ms is not None else None,
            "FIRST_SCHEDULER_WAIT_MS": round(session.first_scheduler_wait_ms, 3) if session.first_scheduler_wait_ms is not None else None,
            "FIRST_AUDIO_TO_FIRST_DECODE_START_MS": round(session.first_decode_start_ms, 3) if session.first_decode_start_ms is not None else None,
            **_decode_ordinal_metrics(session),
            "QWEN_DECODE_SLO_TARGET_MS": 100,
            "QWEN_DECODE_SLO_VIOLATION": None if finish_steps <= 0 else bool(dispatch.metrics.get("qwen_decode_slo_violation", False)),
            "SERVER_FIRST_PARTIAL_MS": round((session.first_partial_at - session.first_audio_at) * 1000, 2) if session.first_partial_at is not None and session.first_audio_at is not None else None,
            "FIRST_PARTIAL_MS": round((session.first_partial_at - session.first_audio_at) * 1000, 2) if session.first_partial_at is not None and session.first_audio_at is not None else None,
            "AUDIO_PUSH_INTERVAL_MS": AUDIO_PUSH_INTERVAL_MS,
            "QWEN_SCHEDULER_WAIT_MS": dispatch.metrics.get("qwen_scheduler_wait_ms"),
            "QWEN_SCHEDULER_REVISION": dispatch.key.scheduler_revision,
            "QWEN_SCHEDULER_WAIT_P50_MS": stream_metrics.get("scheduler_wait_p50_ms"),
            "QWEN_SCHEDULER_WAIT_P95_MS": stream_metrics.get("scheduler_wait_p95_ms"),
            "QWEN_SCHEDULER_WAIT_MAX_MS": stream_metrics.get("scheduler_wait_max_ms"),
            "QWEN_SCHEDULER_WAIT_SAMPLE_COUNT": stream_metrics.get("scheduler_wait_sample_count"),
            "QWEN_DECODE_WALL_P50_MS": stream_metrics.get("decode_wall_p50_ms"),
            "QWEN_DECODE_WALL_P95_MS": stream_metrics.get("decode_wall_p95_ms"),
            "QWEN_DECODE_WALL_MAX_MS": stream_metrics.get("decode_wall_max_ms"),
            "QWEN_DECODE_WALL_SAMPLE_COUNT": stream_metrics.get("decode_wall_sample_count"),
            "QWEN_SCHEDULER_METRIC_HISTORY_LIMIT": stream_metrics.get("scheduler_metric_history_limit"),
            "QWEN_DECODE_STEPS_DELTA": dispatch.metrics.get("qwen_decode_steps_delta", 0),
            "QWEN_DECODE_STEPS_DELTA_TOTAL": stream_metrics.get("decode_steps_delta_total"),
            "QWEN_DECODE_BACKLOG_MS": dispatch.metrics.get("qwen_decode_backlog_ms"),
            "QWEN_DECODE_BACKLOG_MAX_MS": stream_metrics.get("scheduler_max_backlog_audio_ms"),
            "STREAM_LAG_MS": dispatch.metrics.get("stream_lag_ms"),
            "STREAM_LAG_MAX_MS": stream_metrics.get("stream_lag_max_ms"),
            "PENDING_DECODE_COUNT": dispatch.metrics.get("pending_decode_count"),
            "PENDING_DECODE_COUNT_MAX": stream_metrics.get("scheduler_pending_jobs_max"),
            "ACTIVE_STREAM_COUNT": self.scheduler.active_stream_count,
            "ACTIVE_STREAM_COUNT_MAX": stream_metrics.get("active_stream_count_max"),
            "DECODE_OVERRUN": bool(dispatch.metrics.get("decode_overrun", False)),
        }
        event = self._filter_candidate_result(session, event, "final_candidate")
        await self.close_session(connection_id=connection_id, source=source)
        return event

    @staticmethod
    def _filter_candidate_result(
        session: StreamSession,
        event: dict[str, Any],
        expected_event: str,
    ) -> dict[str, Any] | None:
        """Apply an optional canonical gate after local freshness succeeds."""
        result_filter = session.candidate_result_filter
        if result_filter is None:
            return event
        try:
            filtered = result_filter(event)
            if filtered is None or not isinstance(filtered, dict):
                return None
            return validate_candidate_event(filtered, expected_event)
        except Exception:
            # A configured result gate is fail-closed: its failure cannot
            # allow an unverified candidate to reach a sink or caller.
            return None

    async def close_session(self, *, connection_id: str, source: str) -> None:
        key = (connection_id, source)
        async with self._sessions_lock:
            session = self.sessions.pop(key, None)
            if session is not None:
                self.sessions_by_stream_id.pop((connection_id, session.stream_id), None)
        if session is not None:
            await self.scheduler.fence_stream(connection_id, session.stream_id)
            try:
                await self.worker.request("close", stream_id=session.stream_id)
            except Exception:
                pass

    async def receive_event(self, *, connection_id: str, source: str) -> dict[str, Any]:
        session = self.sessions.get((connection_id, source))
        if session is None:
            raise StreamingError("stream_not_started")
        return await session.event_queue.get()

    async def _scheduler_fault(
        self, connection_id: str, stream_id: str, code: str, details: dict[str, Any],
    ) -> None:
        session = self.sessions_by_stream_id.get((connection_id, stream_id))
        if session is None or code == "stream_scheduler_overrun":
            return
        payload = {"event": "error", "code": code, "DECODE_OVERRUN": False}
        try:
            if session.event_sink is not None:
                await session.event_sink(payload)
            else:
                session.event_queue.put_nowait(payload)
        except Exception:
            pass
        await self.close_session(connection_id=connection_id, source=session.source)

    @staticmethod
    def _observe_local_stale_reject(job: DecodeJob, key: RequestKey, reason: str) -> None:
        observer = job.local_stale_reject_observer
        if observer is None:
            return
        try:
            observer(key.stream_id, reason)
        except Exception:
            # Observability must not alter stale-result fencing or delivery.
            pass

    async def close_connection(self, connection_id: str) -> None:
        keys = [key for key in self.sessions if key[0] == connection_id]
        for _, source in keys:
            await self.close_session(connection_id=connection_id, source=source)

    async def _expire_locked(self, now: float) -> None:
        expired = [
            key for key, session in self.sessions.items()
            if now - session.last_activity_at >= self.session_idle_ttl_seconds
        ]
        for connection_id, source in expired:
            session = self.sessions.pop((connection_id, source), None)
            if session is not None:
                self.sessions_by_stream_id.pop((connection_id, session.stream_id), None)
                await self.scheduler.fence_stream(connection_id, session.stream_id)
                try:
                    await self.worker.request("close", stream_id=session.stream_id)
                except Exception:
                    pass

    async def _expire_loop(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(min(5.0, max(0.25, self.session_idle_ttl_seconds / 4)))
                async with self._sessions_lock:
                    await self._expire_locked(time.perf_counter())
        except asyncio.CancelledError:
            raise

    async def close(self) -> None:
        self._closing = True
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
        for connection_id, source in list(self.sessions):
            await self.close_session(connection_id=connection_id, source=source)
        await self.scheduler.close()
        await self.worker.close()


def safe_worker_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", value))
