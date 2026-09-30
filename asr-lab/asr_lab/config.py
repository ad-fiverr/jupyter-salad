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

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        errors: list[str] = []

        raw_backend = env.get("ASR_BACKEND", "").strip()
        backend = raw_backend if raw_backend in MODEL_IDS else None
        if backend is None:
            errors.append("missing_or_invalid_backend")

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
        )
