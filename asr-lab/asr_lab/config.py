"""Validated runtime configuration with no implicit backend winner."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

MODEL_IDS = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "faster_whisper": "deepdml/faster-whisper-large-v3-turbo-ct2",
}
MODEL_REVISIONS = {
    "faster_whisper": "4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
}
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_STREAM_CHUNK_MS_VALUES = (250, 500, 1000, 2000)
QWEN_LANGUAGE_CODES = frozenset({
    "auto", "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
    "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da",
    "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
})
ALLOWED_SOURCES = {"mic", "system"}
ALLOWED_SPEAKERS = {"you", "them", "system"}


@dataclass(frozen=True)
class Settings:
    backend: str | None
    api_token: str | None
    workers: int
    max_message_bytes: int
    max_chunk_seconds: float
    max_buffer_seconds: float
    max_queue_size: int
    max_connections: int
    max_pending_per_connection: int
    error: str | None = None
    test_no_model: bool = False
    qwen_stream_chunk_ms: int = 1000
    qwen_unfixed_chunk_num: int = 2
    qwen_unfixed_token_num: int = 5
    qwen_language: str = "auto"
    qwen_gpu_memory_utilization: float = 0.65
    qwen_max_active_sessions: int = 6
    qwen_max_pending_jobs: int = 24
    qwen_max_backlog_chunks: int = 4
    qwen_max_stream_seconds: float = 60.0
    qwen_session_idle_ttl_seconds: float = 120.0
    qwen_max_context_chars: int = 512
    qwen_worker_python: str = "/opt/qwen-asr-venv/bin/python"
    qwen_streaming_enabled: bool = False
    qwen_streaming_runtime_available: bool = False

    @property
    def model_id(self) -> str | None:
        if self.backend is None:
            return None
        return MODEL_IDS[self.backend]

    @property
    def model_revision(self) -> str | None:
        if self.backend is None:
            return None
        return MODEL_REVISIONS.get(self.backend)

    @property
    def transcript_mode(self) -> str:
        return "STREAMING_PARTIALS" if self.qwen_streaming_enabled else "FINAL_SEGMENT"

    @property
    def active_backend(self) -> str | None:
        return "qwen3_asr" if self.qwen_streaming_enabled else self.backend

    @property
    def active_model_id(self) -> str | None:
        return QWEN_MODEL_ID if self.qwen_streaming_enabled else self.model_id

    @property
    def active_model_revision(self) -> str | None:
        return QWEN_MODEL_REVISION if self.qwen_streaming_enabled else self.model_revision

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        errors: list[str] = []

        raw_backend = env.get("ASR_BACKEND", "").strip()
        backend = raw_backend if raw_backend in MODEL_IDS else None
        if backend is None:
            errors.append("missing_or_invalid_backend")

        raw_qwen_enabled = env.get("QWEN_STREAMING_ENABLED", "0").strip().lower()
        if raw_qwen_enabled not in {"0", "1", "false", "true"}:
            errors.append("invalid_qwen_streaming_enabled")
        qwen_streaming_enabled = raw_qwen_enabled in {"1", "true"}
        qwen_streaming_runtime_available = env.get("QWEN_STREAMING_RUNTIME_AVAILABLE", "0") == "1"
        if qwen_streaming_enabled:
            if not qwen_streaming_runtime_available:
                errors.append("qwen_streaming_runtime_not_installed")
        else:
            # An inactive experiment must not invalidate the production service
            # because of stale or malformed Qwen-only configuration.
            qwen_streaming_runtime_available = False

        token = env.get("ASR_API_TOKEN", "")
        if len(token.encode("utf-8")) < 24:
            token = None
            errors.append("missing_or_weak_api_token")

        def integer(name: str, default: int, low: int, high: int | None) -> int:
            raw = env.get(name, str(default))
            try:
                value = int(raw)
            except (TypeError, ValueError):
                errors.append("invalid_" + name.lower())
                return default
            if value < low or (high is not None and value > high):
                errors.append("invalid_" + name.lower())
                return default
            return value

        def number(name: str, default: float, low: float, high: float) -> float:
            raw = env.get(name, str(default))
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append("invalid_" + name.lower())
                return default
            if not low <= value <= high:
                errors.append("invalid_" + name.lower())
                return default
            return value

        qwen_chunk_ms = 1000
        qwen_language = "auto"
        qwen_worker_python = "/opt/qwen-asr-venv/bin/python"
        qwen_unfixed_chunk_num = 2
        qwen_unfixed_token_num = 5
        qwen_gpu_memory_utilization = 0.65
        qwen_max_active_sessions = 6
        qwen_max_pending_jobs = 24
        qwen_max_backlog_chunks = 4
        qwen_max_stream_seconds = 60.0
        qwen_session_idle_ttl_seconds = 120.0
        qwen_max_context_chars = 512
        if qwen_streaming_enabled:
            raw_qwen_chunk = env.get("QWEN_STREAM_CHUNK_MS", "1000")
            try:
                qwen_chunk_ms = int(raw_qwen_chunk)
                if qwen_chunk_ms not in QWEN_STREAM_CHUNK_MS_VALUES:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("invalid_qwen_stream_chunk_ms")

            qwen_language = env.get("QWEN_LANGUAGE", "auto").strip().lower()
            if qwen_language not in QWEN_LANGUAGE_CODES:
                errors.append("invalid_qwen_language")
                qwen_language = "auto"

            qwen_worker_python = env.get("QWEN_ASR_PYTHON", qwen_worker_python).strip()
            if not qwen_worker_python or "\n" in qwen_worker_python or "\r" in qwen_worker_python:
                errors.append("invalid_qwen_asr_python")
                qwen_worker_python = "/opt/qwen-asr-venv/bin/python"
            qwen_unfixed_chunk_num = integer("QWEN_UNFIXED_CHUNK_NUM", 2, 0, 64)
            qwen_unfixed_token_num = integer("QWEN_UNFIXED_TOKEN_NUM", 5, 0, 128)
            qwen_gpu_memory_utilization = number("QWEN_GPU_MEMORY_UTILIZATION", 0.65, 0.1, 0.89)
            qwen_max_active_sessions = integer("QWEN_MAX_ACTIVE_STREAMS", 6, 1, 6)
            qwen_max_pending_jobs = integer("QWEN_MAX_PENDING_JOBS", 24, 6, 192)
            qwen_max_backlog_chunks = integer("QWEN_MAX_BACKLOG_CHUNKS", 4, 1, 32)
            qwen_max_stream_seconds = number("QWEN_MAX_STREAM_SECONDS", 60.0, 5.0, 300.0)
            qwen_session_idle_ttl_seconds = number("QWEN_SESSION_IDLE_TTL_SECONDS", 120.0, 15.0, 3600.0)
            qwen_max_context_chars = integer("QWEN_MAX_CONTEXT_CHARS", 512, 0, 2048)

        # Each worker owns one full model instance. There is deliberately no
        # artificial ceiling; VRAM and throughput must be measured on the GPU.
        workers = integer("ASR_WORKERS", 1, 1, None)
        test_no_model = env.get("ASR_TEST_NO_MODEL", "") == "1"
        if test_no_model and env.get("CI", "").lower() != "true":
            errors.append("test_mode_not_allowed")

        return cls(
            backend=backend,
            api_token=token,
            workers=workers,
            max_message_bytes=integer("ASR_MAX_MESSAGE_BYTES", 1_048_576, 1024, 8_388_608),
            max_chunk_seconds=number("ASR_MAX_CHUNK_SECONDS", 5.0, 0.1, 30.0),
            max_buffer_seconds=number("ASR_MAX_BUFFER_SECONDS", 30.0, 0.5, 120.0),
            max_queue_size=integer("ASR_MAX_QUEUE_SIZE", 8, 1, 256),
            max_connections=integer("ASR_MAX_CONNECTIONS", 8, 1, 128),
            max_pending_per_connection=integer("ASR_MAX_PENDING_PER_CONNECTION", 2, 1, 4),
            error=errors[0] if errors else None,
            test_no_model=test_no_model and env.get("CI", "").lower() == "true",
            qwen_stream_chunk_ms=qwen_chunk_ms,
            qwen_unfixed_chunk_num=qwen_unfixed_chunk_num,
            qwen_unfixed_token_num=qwen_unfixed_token_num,
            qwen_language=qwen_language,
            qwen_gpu_memory_utilization=qwen_gpu_memory_utilization,
            qwen_max_active_sessions=qwen_max_active_sessions,
            qwen_max_pending_jobs=qwen_max_pending_jobs,
            qwen_max_backlog_chunks=qwen_max_backlog_chunks,
            qwen_max_stream_seconds=qwen_max_stream_seconds,
            qwen_session_idle_ttl_seconds=qwen_session_idle_ttl_seconds,
            qwen_max_context_chars=qwen_max_context_chars,
            qwen_worker_python=qwen_worker_python,
            qwen_streaming_enabled=qwen_streaming_enabled,
            qwen_streaming_runtime_available=qwen_streaming_runtime_available,
        )
