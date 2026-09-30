"""Historical Parakeet RMS segmentation with the lab's per-connection buffers."""
from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import AudioChunk, pcm_rms_normalized

RMS_SPEECH_THRESHOLD = 0.015
RMS_SILENCE_THRESHOLD = 0.008
MIN_SPEECH_SAMPLES = 8_000
MAX_BUFFER_SAMPLES = 48_000
SILENCE_CHUNKS_TO_FLUSH = 4
INACTIVITY_FLUSH_SECONDS = 0.6


@dataclass
class AudioBuffer:
    source: str
    speaker: str
    started_at: float
    last_audio_at: float
    pcm: bytearray = field(default_factory=bytearray)
    last_voice_at: float = 0.0
    has_voice: bool = False
    is_speaking: bool = False
    silence_streak: int = 0

    def append(
        self,
        chunk: AudioChunk,
        now: float,
        *,
        speech_threshold: float = RMS_SPEECH_THRESHOLD,
        silence_threshold: float = RMS_SILENCE_THRESHOLD,
        silence_chunks_to_flush: int = SILENCE_CHUNKS_TO_FLUSH,
    ) -> bool:
        """Append one protocol chunk and report whether the legacy gate closes."""
        if chunk.source != self.source:
            raise ValueError("source_mismatch")
        self.last_audio_at = now
        self.speaker = chunk.speaker
        rms = pcm_rms_normalized(chunk.pcm16le)

        if rms >= speech_threshold:
            self.is_speaking = True
            self.silence_streak = 0
            self.pcm.extend(chunk.pcm16le)
            self.last_voice_at = now
            self.has_voice = True
        elif rms < silence_threshold:
            if self.is_speaking:
                self.silence_streak += 1
                self.pcm.extend(chunk.pcm16le)
        elif self.is_speaking:
            # Historical hysteresis: the middle band is appended but does not
            # reset the number of already observed quiet chunks.
            self.pcm.extend(chunk.pcm16le)

        should_flush = (
            (self.is_speaking and self.silence_streak >= silence_chunks_to_flush)
            or self.sample_count >= MAX_BUFFER_SAMPLES
        )
        if should_flush and not self.can_flush:
            self.reset()
        return should_flush

    @property
    def sample_count(self) -> int:
        return len(self.pcm) // 2

    @property
    def can_flush(self) -> bool:
        return self.is_speaking and self.sample_count >= MIN_SPEECH_SAMPLES

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / 16_000.0

    def take(self) -> bytes:
        payload = bytes(self.pcm)
        self.reset()
        return payload

    def force_flush(self) -> bytes | None:
        """Match the historical inactivity path: only voiced >=500 ms flushes."""
        payload = bytes(self.pcm) if self.can_flush else None
        self.reset()
        return payload

    def reset(self) -> None:
        self.pcm.clear()
        self.last_voice_at = 0.0
        self.has_voice = False
        self.is_speaking = False
        self.silence_streak = 0
