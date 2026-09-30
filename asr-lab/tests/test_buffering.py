from __future__ import annotations

import base64
import json
import struct
import unittest

from asr_lab.buffering import (
    INACTIVITY_FLUSH_SECONDS,
    MAX_BUFFER_SAMPLES,
    MIN_SPEECH_SAMPLES,
    RMS_SILENCE_THRESHOLD,
    RMS_SPEECH_THRESHOLD,
    SILENCE_CHUNKS_TO_FLUSH,
    AudioBuffer,
)
from asr_lab.config import MODEL_REVISIONS, Settings
from asr_lab.protocol import AudioChunk, parse_message


class BufferingTests(unittest.TestCase):
    def chunk(self, value: int, speaker: str = "you", source: str = "mic", samples: int = 1600) -> AudioChunk:
        pcm = struct.pack("<h", value) * samples
        message = {
            "source": source,
            "speaker": speaker,
            "encoding": "pcm_int16",
            "sample_rate": 16000,
            "audio": base64.b64encode(pcm).decode("ascii"),
        }
        result = parse_message(json.dumps(message), max_message_bytes=1_000_000, max_chunk_seconds=5)
        assert isinstance(result, AudioChunk)
        return result

    def test_historical_thresholds_are_restored(self):
        self.assertEqual(RMS_SPEECH_THRESHOLD, 0.015)
        self.assertEqual(RMS_SILENCE_THRESHOLD, 0.008)
        self.assertEqual(MIN_SPEECH_SAMPLES, 8000)
        self.assertEqual(MAX_BUFFER_SAMPLES, 48000)
        self.assertEqual(SILENCE_CHUNKS_TO_FLUSH, 4)
        self.assertEqual(INACTIVITY_FLUSH_SECONDS, 0.6)

    def test_hysteresis_and_four_chunk_flush_match_historical_path(self):
        buffer = AudioBuffer("mic", "you", 1.0, 1.0)
        for index in range(5):
            self.assertFalse(buffer.append(self.chunk(8000), 1.1 + index * 0.1))
        self.assertEqual(buffer.sample_count, MIN_SPEECH_SAMPLES)
        self.assertTrue(buffer.is_speaking)
        self.assertAlmostEqual(buffer.last_voice_at, 1.5)

        for index in range(3):
            self.assertFalse(buffer.append(self.chunk(0), 1.6 + index * 0.1))
        # The middle RMS band is buffered but does not reset the quiet streak.
        self.assertFalse(buffer.append(self.chunk(400), 1.9))
        self.assertEqual(buffer.silence_streak, 3)
        self.assertTrue(buffer.append(self.chunk(0), 2.0))
        self.assertTrue(buffer.can_flush)
        self.assertAlmostEqual(buffer.duration_seconds, 1.0)

    def test_silence_only_is_ignored_and_does_not_flush(self):
        buffer = AudioBuffer("system", "them", 0.0, 0.0)
        for index in range(8):
            self.assertFalse(buffer.append(self.chunk(0, "them", "system"), index * 0.1))
        self.assertFalse(buffer.has_voice)
        self.assertEqual(buffer.sample_count, 0)

    def test_max_buffer_flushes_at_48000_samples(self):
        buffer = AudioBuffer("mic", "you", 0.0, 0.0)
        for index in range(29):
            self.assertFalse(buffer.append(self.chunk(8000), index * 0.1))
        self.assertTrue(buffer.append(self.chunk(8000), 2.9))
        self.assertEqual(buffer.sample_count, MAX_BUFFER_SAMPLES)
        self.assertTrue(buffer.can_flush)

    def test_inactivity_flush_requires_500ms_of_speech(self):
        buffer = AudioBuffer("mic", "you", 0.0, 0.0)
        buffer.append(self.chunk(8000, samples=MIN_SPEECH_SAMPLES - 1), 0.1)
        self.assertIsNone(buffer.force_flush())
        buffer = AudioBuffer("mic", "you", 0.0, 0.0)
        buffer.append(self.chunk(8000, samples=MIN_SPEECH_SAMPLES), 0.1)
        self.assertEqual(len(buffer.force_flush() or b""), MIN_SPEECH_SAMPLES * 2)

    def test_backend_revisions_are_pinned_where_backend_uses_them(self):
        self.assertNotIn("parakeet", MODEL_REVISIONS)
        self.assertEqual(len(MODEL_REVISIONS["faster_whisper"]), 40)
        settings = Settings.from_env({"ASR_BACKEND": "faster_whisper", "ASR_API_TOKEN": "x" * 32})
        self.assertEqual(settings.model_revision, MODEL_REVISIONS["faster_whisper"])
        self.assertEqual(settings.max_pending_per_connection, 2)


if __name__ == "__main__":
    unittest.main()
