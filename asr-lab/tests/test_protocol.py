from __future__ import annotations

import base64
import unittest

from asr_lab.config import Settings
from asr_lab.protocol import AudioChunk, BenchmarkPing, FlushRequest, ProtocolError, parse_message
from asr_lab.security import token_matches


def audio_message(**overrides):
    message = {
        "source": "mic",
        "speaker": "you",
        "encoding": "pcm_int16",
        "sample_rate": 16000,
        "audio": base64.b64encode(b"\x01\x00" * 1600).decode("ascii"),
    }
    message.update(overrides)
    return message


class ProtocolTests(unittest.TestCase):
    def test_valid_legacy_pcm_shape(self):
        import json
        parsed = parse_message(json.dumps(audio_message()), max_message_bytes=1_000_000, max_chunk_seconds=5)
        self.assertIsInstance(parsed, AudioChunk)
        self.assertEqual(parsed.source, "mic")
        self.assertEqual(parsed.sample_rate, 16000)
        self.assertEqual(parsed.duration_seconds, 0.1)

    def test_flush_control(self):
        import json
        parsed = parse_message(json.dumps({"event": "flush", "source": "system"}), max_message_bytes=1000, max_chunk_seconds=5)
        self.assertEqual(parsed, FlushRequest("system"))

    def test_benchmark_ping_is_strict_and_requires_a_bounded_nonempty_id(self):
        import json
        parsed = parse_message(
            json.dumps({"event": "benchmark_ping", "request_id": "ping-1"}),
            max_message_bytes=1000,
            max_chunk_seconds=5,
        )
        self.assertEqual(parsed, BenchmarkPing("ping-1"))
        for payload in (
            {"event": "benchmark_ping"},
            {"event": "benchmark_ping", "request_id": ""},
            {"event": "benchmark_ping", "request_id": "x" * 129},
            {"event": "benchmark_ping", "request_id": "ping-1", "source": "mic"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                parse_message(json.dumps(payload), max_message_bytes=1000, max_chunk_seconds=5)

    def test_invalid_payloads_fail_closed(self):
        import json
        cases = [
            audio_message(sample_rate=48000),
            audio_message(encoding="webm"),
            audio_message(audio="not base64!"),
            audio_message(source="other"),
            audio_message(extra="unsupported"),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ProtocolError):
                    parse_message(json.dumps(case), max_message_bytes=1_000_000, max_chunk_seconds=5)

    def test_message_and_chunk_limits(self):
        import json
        message = json.dumps(audio_message())
        with self.assertRaises(ProtocolError):
            parse_message(message, max_message_bytes=20, max_chunk_seconds=5)
        with self.assertRaises(ProtocolError):
            parse_message(json.dumps(audio_message(audio=base64.b64encode(b"\0\0" * 40000).decode())), max_message_bytes=1_000_000, max_chunk_seconds=1)

    def test_odd_pcm_byte_count_is_rejected(self):
        import json
        with self.assertRaises(ProtocolError) as raised:
            parse_message(
                json.dumps(audio_message(audio=base64.b64encode(b"\x01\x00\x02").decode())),
                max_message_bytes=1_000_000,
                max_chunk_seconds=5,
            )
        self.assertEqual(raised.exception.code, "invalid_pcm_length")

    def test_settings_need_selector_and_strong_token(self):
        base = {"ASR_BACKEND": "parakeet", "ASR_API_TOKEN": "x" * 32}
        self.assertIsNone(Settings.from_env(base).error)
        self.assertEqual(Settings.from_env(base).workers, 1)
        self.assertEqual(Settings.from_env(base).max_buffer_seconds, 30.0)
        for worker_count in (1, 2, 4, 8, 16, 100):
            self.assertEqual(Settings.from_env({**base, "ASR_WORKERS": str(worker_count)}).workers, worker_count)
            self.assertIsNone(Settings.from_env({**base, "ASR_WORKERS": str(worker_count)}).error)
        self.assertIsNotNone(Settings.from_env({**base, "ASR_WORKERS": "0"}).error)
        self.assertIsNotNone(Settings.from_env({**base, "ASR_WORKERS": "many"}).error)
        self.assertIsNotNone(Settings.from_env({**base, "ASR_BACKEND": "both"}).error)
        self.assertIsNotNone(Settings.from_env({**base, "ASR_API_TOKEN": "weak"}).error)

    def test_ci_model_bypass_cannot_be_enabled_outside_ci(self):
        base = {"ASR_BACKEND": "parakeet", "ASR_API_TOKEN": "x" * 32, "ASR_TEST_NO_MODEL": "1"}
        self.assertIsNotNone(Settings.from_env(base).error)
        self.assertTrue(Settings.from_env({**base, "CI": "true"}).test_no_model)

    def test_token_comparison(self):
        self.assertTrue(token_matches("secret" * 5, "secret" * 5))
        self.assertFalse(token_matches("wrong" * 5, "secret" * 5))


if __name__ == "__main__":
    unittest.main()
