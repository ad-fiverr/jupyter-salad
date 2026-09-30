"""CTranslate2 adapter for deepdml/faster-whisper-large-v3-turbo-ct2."""
from __future__ import annotations

import os

from .base import pcm16le_to_float32


class FasterWhisperBackend:
    name = "faster_whisper"

    def __init__(self):
        self.model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
        self._model = None

    def load(self, model_id: str | None = None, model_revision: str | None = None) -> None:
        self.model_id = model_id or self.model_id
        from huggingface_hub import snapshot_download
        from faster_whisper import WhisperModel

        revision = model_revision or "4df90f75321148c3a29a9e2351b7ddf8f5b115a8"
        model_path = snapshot_download(
            repo_id=self.model_id,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or None,
        )
        self._model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16",
        )

    def warmup(self) -> None:
        import numpy as np

        if self._model is None:
            raise RuntimeError("MODEL_NOT_LOADED")
        segments, _ = self._model.transcribe(
            np.zeros(8_000, dtype=np.float32),
            language="es",
            beam_size=1,
            vad_filter=False,
            temperature=0,
            condition_on_previous_text=False,
        )
        list(segments)

    def transcribe(self, pcm16le: bytes) -> str:
        if self._model is None:
            raise RuntimeError("MODEL_NOT_LOADED")
        waveform = pcm16le_to_float32(pcm16le)
        segments, _ = self._model.transcribe(
            waveform,
            language="es",
            beam_size=1,
            vad_filter=False,
            temperature=0,
            condition_on_previous_text=False,
        )
        # Faster-Whisper returns a lazy generator; iterate inside the timed call.
        return "".join(segment.text for segment in segments).strip()

    def close(self) -> None:
        self._model = None
