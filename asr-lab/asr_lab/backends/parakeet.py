"""Historical NVIDIA NeMo adapter for Parakeet TDT v3."""
from __future__ import annotations

from .base import pcm16le_to_float32

PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
HALLUCINATION_BLACKLIST = {
    "you", "thank you", "oh", "bye", "subtitles by",
    "gracias por ver el video", "gracias por ver el vídeo",
    "suscríbete al canal", "yeah", "let's go", "mm-hmm",
}


def normalize_historical_transcript(text: str) -> str:
    """Use the exact pre-blacklist normalization from the historical server."""
    return text.lower().strip(" .,!¡¿?")


def should_discard_historical_transcript(text: str) -> bool:
    return not text or len(text) < 2 or normalize_historical_transcript(text) in HALLUCINATION_BLACKLIST


class ParakeetBackend:
    name = "parakeet"

    def __init__(self):
        self.model_id = PARAKEET_MODEL
        self._model = None

    def load(self, model_id: str | None = None, model_revision: str | None = None) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")

        # NeMo's pretrained resolver owns revision selection for the historical
        # model ID; model_revision belongs only to the Faster-Whisper adapter.
        self.model_id = model_id or PARAKEET_MODEL
        self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)
        self._model = self._model.cuda()
        self._model.eval()

    def warmup(self) -> None:
        import numpy as np

        if self._model is None:
            raise RuntimeError("MODEL_NOT_LOADED")
        self._model.transcribe([np.zeros(16_000, dtype=np.float32)])

    def transcribe(self, pcm16le: bytes) -> str:
        if self._model is None:
            raise RuntimeError("MODEL_NOT_LOADED")
        waveform = pcm16le_to_float32(pcm16le)
        result = self._model.transcribe([waveform])
        return result[0].text.strip() if result else ""

    def close(self) -> None:
        self._model = None
