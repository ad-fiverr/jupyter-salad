"""Backend protocol and PCM conversion shared by selected adapters."""
from __future__ import annotations

from typing import Protocol


class ASRBackend(Protocol):
    name: str
    model_id: str

    def load(self, model_id: str, model_revision: str) -> None: ...
    def warmup(self) -> None: ...
    def transcribe(self, pcm16le: bytes) -> str: ...
    def close(self) -> None: ...


def pcm16le_to_float32(pcm16le: bytes):
    import numpy as np

    return np.frombuffer(pcm16le, dtype="<i2").astype(np.float32) / 32768.0


def create_backend(name: str) -> ASRBackend:
    """Import exactly one backend module based on the validated selector."""
    if name == "parakeet":
        from .parakeet import ParakeetBackend

        return ParakeetBackend()
    if name == "faster_whisper":
        from .faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend()
    raise ValueError("ASR_BACKEND must be parakeet or faster_whisper.")
