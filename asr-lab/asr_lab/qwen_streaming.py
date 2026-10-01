"""Bounded, connection-owned Qwen streaming sessions over a private worker RPC."""
from __future__ import annotations

import asyncio
import base64
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

STREAMING_CLASS = "accumulated-audio-pseudostreaming"
ALLOWED_CHUNK_MS = (250, 500, 1000, 2000)
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


class StreamingError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


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
    first_audio_at: float | None = None
    audio_samples: int = 0
    revision: int = 0
    partial_count: int = 0
    current_text: str = ""
    current_language: str | None = None
    first_partial_at: float | None = None
    last_partial_at: float | None = None
    cumulative_decode_wall_ms: float = 0.0


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
        self._sessions_lock = asyncio.Lock()
        self._sweeper: asyncio.Task[None] | None = None
        self._closing = False
        self._worker_metrics: list[dict[str, Any]] = []

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
        return 0

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
        self._sweeper = asyncio.create_task(self._expire_loop(), name="qwen-stream-expiry")

    async def open_session(
        self,
        *,
        connection_id: str,
        source: str,
        language: str | None = None,
        context: str = "",
        chunk_size_ms: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.ready:
            raise StreamingError("backend_not_ready")
        language_value = self.default_language if language is None else language
        chunk_ms = self.default_chunk_ms if chunk_size_ms is None else chunk_size_ms
        if chunk_ms not in ALLOWED_CHUNK_MS:
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
            await self.worker.request(
                "init",
                stream_id=stream_id,
                chunk_size_sec=chunk_ms / 1000.0,
                language=QWEN_LANGUAGE_NAMES.get(language_value),
                context=context,
                unfixed_chunk_num=self.unfixed_chunk_num,
                unfixed_token_num=self.unfixed_token_num,
            )
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
            )
            self.sessions[key] = session
        return {
            "event": "stream_started",
            "backend": "qwen3_asr",
            "stream_id": stream_id,
            "request_id": request_id,
            "transcript_mode": "STREAMING_PARTIALS",
            "streaming_class": STREAMING_CLASS,
            "audio_push_interval_ms": 100,
            "model_decode_chunk_ms": chunk_ms,
            "language": language_value,
        }

    async def push_audio(
        self, *, connection_id: str, source: str, pcm16le: bytes,
    ) -> dict[str, Any] | None:
        key = (connection_id, source)
        session = self.sessions.get(key)
        if session is None or session.owner_connection_id != connection_id:
            raise StreamingError("stream_not_started")
        samples = len(pcm16le) // 2
        if session.audio_samples + samples > session.max_stream_seconds * 16_000:
            await self.close_session(connection_id=connection_id, source=source)
            raise StreamingError("stream_duration_limit")
        session.audio_samples += samples
        chunk_started = time.perf_counter()
        if session.first_audio_at is None:
            session.first_audio_at = chunk_started
        session.last_activity_at = chunk_started
        started = chunk_started
        try:
            reply = await self.worker.request(
                "push",
                stream_id=session.stream_id,
                pcm16le_base64=base64.b64encode(pcm16le).decode("ascii"),
            )
        except StreamingError as exc:
            if exc.code in {"stream_worker_unavailable", "stream_worker_timeout"}:
                await self.close_session(connection_id=connection_id, source=source)
                raise StreamingError("stream_worker_failed") from exc
            raise
        except Exception as exc:
            raise StreamingError("stream_worker_failed") from exc
        ready = time.perf_counter()
        if self.sessions.get(key) is not session:
            # Disconnect/TTL fencing: a reply for a removed owner is discarded.
            return None
        session.last_activity_at = ready
        decode_wall_ms = max(0.0, float(reply.get("decode_wall_ms", 0.0)))
        if reply.get("decoded") is True:
            session.cumulative_decode_wall_ms += decode_wall_ms
        text = reply.get("text")
        if not isinstance(text, str):
            raise StreamingError("invalid_worker_response")
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
        cursor_ms = session.audio_samples * 1000.0 / 16_000
        duration_ms = max(cursor_ms, 1.0)
        return {
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
            "request_id": session.request_id,
            "PARTIAL_COUNT": session.partial_count,
            "FIRST_PARTIAL_MS": round((ready - (session.first_audio_at or session.started_at)) * 1000, 2),
            "PARTIAL_UPDATE_INTERVAL_MS": round(interval_ms, 2) if interval_ms is not None else None,
            "PARTIAL_REVISION_RATE": round(revision_rate, 4) if revision_rate is not None else None,
            "PARTIAL_STABILITY": round(stability, 4) if stability is not None else None,
            "SERVER_CHUNK_TO_PARTIAL_MS": round((ready - started) * 1000, 2),
            "QWEN_DECODE_CALL_WALL_MS": round(decode_wall_ms, 2) if reply.get("decoded") else None,
            "QWEN_CUMULATIVE_DECODE_WALL_MS": round(session.cumulative_decode_wall_ms, 2),
            "QWEN_STREAM_RTF": round(session.cumulative_decode_wall_ms / duration_ms, 4),
            "AUDIO_DURATION_MS": round(cursor_ms, 2),
            "MODEL_DECODE_CHUNK_MS": session.chunk_size_ms,
            "AUDIO_PUSH_INTERVAL_MS": 100,
        }

    async def finish(
        self, *, connection_id: str, source: str, server_eos_at: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        key = (connection_id, source)
        session = self.sessions.get(key)
        if session is None or session.owner_connection_id != connection_id:
            raise StreamingError("stream_not_started")
        eos_at = time.perf_counter() if server_eos_at is None else server_eos_at
        try:
            reply = await self.worker.request("finish", stream_id=session.stream_id)
        except Exception as exc:
            await self.close_session(connection_id=connection_id, source=source)
            raise StreamingError("stream_worker_failed") from exc
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
        session.cumulative_decode_wall_ms += finish_decode_ms
        cursor_ms = session.audio_samples * 1000.0 / 16_000
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
            "AUDIO_PUSH_INTERVAL_MS": 100,
        }
        await self.close_session(connection_id=connection_id, source=source)
        return event

    async def close_session(self, *, connection_id: str, source: str) -> None:
        key = (connection_id, source)
        async with self._sessions_lock:
            session = self.sessions.pop(key, None)
            if session is not None:
                try:
                    await self.worker.request("close", stream_id=session.stream_id)
                except Exception:
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
        await self.worker.close()


def safe_worker_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", value))
