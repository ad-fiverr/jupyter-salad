from __future__ import annotations

import base64
import unittest

from asr_lab.config import Settings
from asr_lab.protocol import AudioChunk, BenchmarkPing, FlushRequest, ProtocolError, StreamStart, parse_message
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

    def test_qwen_stream_start_accepts_configured_chunk_sizes_and_normalizes_context(self):
        import json
        for chunk_ms in (250, 500, 1000, 2000):
            parsed = parse_message(
                json.dumps({
                    "event": "stream_start", "source": "mic", "language": "auto",
                    "context": "  OpenAI   Salad  ", "chunk_size_ms": chunk_ms,
                    "request_id": "stream-1",
                }),
                max_message_bytes=1000,
                max_chunk_seconds=5,
            )
            self.assertEqual(
                parsed,
                StreamStart("mic", "auto", "OpenAI Salad", chunk_ms, "stream-1"),
            )

    def test_qwen_stream_start_rejects_unbounded_or_ambiguous_controls(self):
        import json
        cases = (
            {"event": "stream_start", "source": "mic", "chunk_size_ms": 251},
            {"event": "stream_start", "source": "mic", "chunk_size_ms": 250.0},
            {"event": "stream_start", "source": "mic", "language": "auto", "extra": "x"},
            {"event": "stream_start", "source": "mic", "context": "x" * 9},
            {"event": "stream_start", "source": "mic", "context": "safe\\nunsafe"},
        )
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                parse_message(
                    json.dumps(payload),
                    max_message_bytes=1000,
                    max_chunk_seconds=5,
                    max_context_chars=8,
                )

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

    def test_qwen_settings_are_opt_in_and_validate_experimental_values(self):
        base = {
            "ASR_BACKEND": "parakeet",
            "ASR_API_TOKEN": "x" * 32,
            "QWEN_STREAMING_ENABLED": "1",
            "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        }
        self.assertIsNone(Settings.from_env(base).error)
        self.assertEqual(Settings.from_env(base).backend, "parakeet")
        self.assertEqual(Settings.from_env(base).active_backend, "qwen3_asr")
        self.assertEqual(Settings.from_env(base).active_model_id, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(Settings.from_env(base).transcript_mode, "STREAMING_PARTIALS")
        self.assertEqual(Settings.from_env(base).qwen_stream_chunk_ms, 1000)
        self.assertEqual(Settings.from_env(base).qwen_max_active_sessions, 6)
        self.assertEqual(Settings.from_env(base).qwen_max_pending_jobs, 24)
        self.assertEqual(Settings.from_env(base).qwen_max_backlog_chunks, 4)
        self.assertEqual(Settings.from_env(base).qwen_language, "auto")
        for chunk_ms in (250, 500, 1000, 2000):
            configured = Settings.from_env({**base, "QWEN_STREAM_CHUNK_MS": str(chunk_ms)})
            self.assertIsNone(configured.error)
            self.assertEqual(configured.qwen_stream_chunk_ms, chunk_ms)
        for stream_count in range(1, 7):
            configured = Settings.from_env({**base, "QWEN_MAX_ACTIVE_STREAMS": str(stream_count)})
            self.assertIsNone(configured.error)
            self.assertEqual(configured.qwen_max_active_sessions, stream_count)
        self.assertIsNotNone(Settings.from_env({**base, "QWEN_MAX_ACTIVE_STREAMS": "7"}).error)
        self.assertIsNotNone(Settings.from_env({**base, "QWEN_STREAM_CHUNK_MS": "300"}).error)
        self.assertIsNone(Settings.from_env({**base, "QWEN_LANGUAGE": "es"}).error)
        self.assertIsNotNone(Settings.from_env({**base, "QWEN_LANGUAGE": "unknown"}).error)
        self.assertIsNotNone(Settings.from_env({"ASR_BACKEND": "qwen3_asr", "ASR_API_TOKEN": "x" * 32}).error)
        self.assertIsNotNone(Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": "x" * 32,
            "QWEN_STREAMING_ENABLED": "1",
        }).error)
        self.assertIsNotNone(Settings.from_env({**base, "QWEN_STREAMING_RUNTIME_AVAILABLE": "0"}).error)
        disabled = Settings.from_env({"ASR_BACKEND": "faster_whisper", "ASR_API_TOKEN": "x" * 32})
        self.assertIsNone(disabled.error)
        self.assertEqual(disabled.transcript_mode, "FINAL_SEGMENT")
        self.assertEqual(disabled.active_backend, "faster_whisper")

    def test_invalid_inactive_qwen_settings_do_not_invalidate_production_asr(self):
        production = {"ASR_BACKEND": "parakeet", "ASR_API_TOKEN": "x" * 32}
        stale_qwen_values = {
            "QWEN_STREAMING_ENABLED": "0",
            "QWEN_STREAMING_RUNTIME_AVAILABLE": "invalid",
            "QWEN_STREAM_CHUNK_MS": "300",
            "QWEN_LANGUAGE": "unknown",
            "QWEN_ASR_PYTHON": "\ninvalid",
            "QWEN_UNFIXED_CHUNK_NUM": "-1",
            "QWEN_UNFIXED_TOKEN_NUM": "999",
            "QWEN_GPU_MEMORY_UTILIZATION": "9",
            "QWEN_MAX_ACTIVE_STREAMS": "0",
            "QWEN_MAX_STREAM_SECONDS": "not-a-number",
            "QWEN_SESSION_IDLE_TTL_SECONDS": "0",
            "QWEN_MAX_CONTEXT_CHARS": "99999",
        }
        disabled = Settings.from_env({**production, **stale_qwen_values})
        self.assertIsNone(disabled.error)
        self.assertFalse(disabled.qwen_streaming_enabled)
        self.assertEqual(disabled.backend, "parakeet")

        enabled = Settings.from_env({
            **production,
            **stale_qwen_values,
            "QWEN_STREAMING_ENABLED": "1",
            "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        })
        self.assertIsNotNone(enabled.error)

    def test_ci_model_bypass_cannot_be_enabled_outside_ci(self):
        base = {"ASR_BACKEND": "parakeet", "ASR_API_TOKEN": "x" * 32, "ASR_TEST_NO_MODEL": "1"}
        self.assertIsNotNone(Settings.from_env(base).error)
        self.assertTrue(Settings.from_env({**base, "CI": "true"}).test_no_model)

    def test_token_comparison(self):
        self.assertTrue(token_matches("secret" * 5, "secret" * 5))
        self.assertFalse(token_matches("wrong" * 5, "secret" * 5))


    def test_qwen_model_chunk_aliases_and_conflicts_are_strict(self):
        base = {
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": "x" * 32,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        }
        for value in (50, 100, 150, 200, 250, 500, 1000, 2000):
            with self.subTest(value=value):
                canonical = Settings.from_env({**base, "QWEN_MODEL_CHUNK_MS": str(value)})
                legacy = Settings.from_env({**base, "QWEN_STREAM_CHUNK_MS": str(value)})
                equal = Settings.from_env({**base, "QWEN_MODEL_CHUNK_MS": str(value), "QWEN_STREAM_CHUNK_MS": str(value)})
                self.assertIsNone(canonical.error)
                self.assertIsNone(legacy.error)
                self.assertIsNone(equal.error)
                self.assertEqual(canonical.qwen_model_chunk_ms, value)
                self.assertEqual(legacy.qwen_model_chunk_ms, value)
                self.assertEqual(equal.qwen_stream_chunk_ms, value)
        for raw in ("0", "-1", "75", "50.0", "nope"):
            config = Settings.from_env({**base, "QWEN_MODEL_CHUNK_MS": raw})
            self.assertEqual(config.error, "invalid_qwen_model_chunk_ms")
        conflict = Settings.from_env({**base, "QWEN_MODEL_CHUNK_MS": "50", "QWEN_STREAM_CHUNK_MS": "100"})
        self.assertEqual(conflict.error, "conflicting_qwen_model_chunk_ms")
        inactive = Settings.from_env({**base, "QWEN_STREAMING_ENABLED": "0", "QWEN_MODEL_CHUNK_MS": "75", "QWEN_STREAM_CHUNK_MS": "50.0"})
        self.assertIsNone(inactive.error)

    def test_qwen_stream_start_canonical_and_legacy_chunk_names(self):
        import json
        for value in (50, 100, 150, 200, 250, 500, 1000, 2000):
            for field in ("model_chunk_ms", "chunk_size_ms"):
                parsed = parse_message(json.dumps({"event": "stream_start", "source": "mic", field: value}), max_message_bytes=1000, max_chunk_seconds=5)
                self.assertEqual(parsed.model_chunk_ms, value)
                self.assertEqual(parsed.chunk_size_ms, value)
            both = parse_message(json.dumps({"event": "stream_start", "source": "mic", "model_chunk_ms": value, "chunk_size_ms": value}), max_message_bytes=1000, max_chunk_seconds=5)
            self.assertEqual(both.model_chunk_ms, value)
        for raw in (True, 50.0, 0, -1, 75):
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                parse_message(json.dumps({"event": "stream_start", "source": "mic", "model_chunk_ms": raw}), max_message_bytes=1000, max_chunk_seconds=5)
        for fields in (
            {"model_chunk_ms": 50, "chunk_size_ms": 50.0},
            {"model_chunk_ms": 50, "chunk_size_ms": None},
            {"model_chunk_ms": None, "chunk_size_ms": 50},
            {"model_chunk_ms": None},
            {"chunk_size_ms": None},
        ):
            with self.subTest(fields=fields), self.assertRaises(ProtocolError) as raised:
                parse_message(json.dumps({"event": "stream_start", "source": "mic", **fields}), max_message_bytes=1000, max_chunk_seconds=5)
            self.assertEqual(raised.exception.code, "invalid_model_chunk")
        with self.assertRaises(ProtocolError) as raised:
            parse_message(json.dumps({"event": "stream_start", "source": "mic", "model_chunk_ms": 50, "chunk_size_ms": 100}), max_message_bytes=1000, max_chunk_seconds=5)
        self.assertEqual(raised.exception.code, "conflicting_model_chunk")


if __name__ == "__main__":
    unittest.main()
