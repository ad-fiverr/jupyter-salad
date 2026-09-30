from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from benchmark import benchmark_ws


class FakeWebSocket:
    def __init__(self, expected_chunks, transcript_payloads=None):
        self.expected_chunks = expected_chunks
        self.transcript_payloads = transcript_payloads if transcript_payloads is not None else [
            {
                "event": "transcript", "type": "transcript", "text": "prueba",
                "backend": "parakeet", "MODEL_INFERENCE_MS": 25.0,
                "SEGMENT_WAIT_MS": 10.0, "SERVER_TO_TRANSCRIPT_MS": 30.0,
                "SERVER_RECEIVE_TO_TRANSCRIPT_MS": 40.0,
                "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": 70.0, "audio_duration_ms": 200.0,
            },
            {
                "event": "transcript", "type": "transcript", "text": "benchmark",
                "backend": "parakeet", "MODEL_INFERENCE_MS": 5.0,
                "SEGMENT_WAIT_MS": 20.0, "SERVER_TO_TRANSCRIPT_MS": 50.0,
                "SERVER_RECEIVE_TO_TRANSCRIPT_MS": 60.0,
                "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": 90.0, "audio_duration_ms": 200.0,
            },
        ]
        self.sent = []
        self.all_audio_sent = asyncio.Event()
        self.flush_sent = asyncio.Event()
        self.responses = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def ping(self):
        pong = asyncio.get_running_loop().create_future()
        pong.set_result(None)
        return pong

    async def send(self, raw):
        payload = json.loads(raw)
        self.sent.append(payload)
        if len(self.sent) == self.expected_chunks:
            self.all_audio_sent.set()
        if payload.get("event") == "flush":
            self.flush_sent.set()

    async def recv(self):
        await self.all_audio_sent.wait()
        await self.flush_sent.wait()
        self.responses += 1
        if self.responses <= len(self.transcript_payloads):
            return json.dumps(self.transcript_payloads[self.responses - 1])
        flush = next(item for item in reversed(self.sent) if item.get("event") == "flush")
        return json.dumps({"event": "flush_complete", "request_id": flush["request_id"]})


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


class BenchmarkContractTests(unittest.IsolatedAsyncioTestCase):
    def test_total_audio_latency_uses_last_transcript_not_flush_barrier(self):
        transcripts = [
            {"client_received_at": 10.125},
            {"client_received_at": 10.240},
        ]
        self.assertAlmostEqual(benchmark_ws.audio_end_to_last_transcript_ms(10.0, transcripts), 240.0)
        self.assertIsNone(benchmark_ws.audio_end_to_last_transcript_ms(10.0, []))

    def test_url_builder_targets_same_asr_websocket_and_never_changes_path(self):
        url = benchmark_ws.build_ws_url("wss://salad.example/asr/ws?mode=lab&token=old", "x" * 32)
        self.assertEqual(urlsplit_path(url), "/asr/ws")
        self.assertIn("token=" + "x" * 32, url)
        self.assertNotIn("old", url)

    async def test_benchmark_uses_ws_contract_exact_metric_names_and_unavailable_vram(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "synthetic.wav"
            with wave.open(str(fixture), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes((10_000).to_bytes(2, "little", signed=True) * 6_400)
            fixture.with_suffix(".txt").write_text("prueba benchmark", encoding="utf-8")

            expected_chunks = 4
            socket = FakeWebSocket(expected_chunks)
            fake_websockets = types.ModuleType("websockets")
            passed_urls = []

            def connect(url, **kwargs):
                passed_urls.append((url, kwargs))
                return socket

            fake_websockets.connect = connect
            health = {
                "backend": "parakeet",
                "model_id": "nvidia/parakeet-tdt-0.6b-v3",
                "model_revision": "fixed-revision",
                "worker_metrics": [{
                    "model_load_ms": 100.0,
                    "vram_before": {"used_global_mib": 900.0},
                    "vram_after_load": {"used_global_mib": 2_000.0},
                    "vram_after_warmup": {"device": "GPU test", "process_allocated_mib": 500.0},
                }],
            }

            def open_health(*_args, **_kwargs):
                return FakeHTTPResponse(json.dumps(health).encode("utf-8"))

            with patch.dict(sys.modules, {"websockets": fake_websockets}), patch.object(
                benchmark_ws.urllib.request, "urlopen", side_effect=open_health
            ):
                result = await benchmark_ws.run_one(
                    "wss://salad.example/asr/ws", "s" * 32, fixture
                )

        self.assertEqual(len(passed_urls), 1)
        self.assertEqual(urlsplit_path(passed_urls[0][0]), "/asr/ws")
        self.assertIn("token=" + "s" * 32, passed_urls[0][0])
        audio_chunks = [chunk for chunk in socket.sent if chunk.get("type", "audio") == "audio" and "audio" in chunk]
        self.assertTrue(all(chunk["encoding"] == "pcm_int16" and chunk["sample_rate"] == 16_000 for chunk in audio_chunks))
        self.assertEqual(result["MODEL_INFERENCE_MS"], 30.0)
        self.assertEqual(result["SEGMENT_WAIT_MS"], 30.0)
        self.assertEqual(result["SERVER_TO_TRANSCRIPT_MS"], 50.0)
        self.assertEqual(result["SERVER_RECEIVE_TO_TRANSCRIPT_MS"], 60.0)
        self.assertEqual(result["SERVER_AUDIO_END_TO_TRANSCRIPT_MS"], 90.0)
        self.assertEqual(result["segments"][0]["SERVER_AUDIO_END_TO_TRANSCRIPT_MS"], 70.0)
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["transcript"], "prueba benchmark")
        self.assertIsInstance(result["NETWORK_RTT_MS"], float)
        self.assertIsInstance(result["TOTAL_AUDIO_END_TO_TRANSCRIPT_MS"], float)
        self.assertEqual(result["VRAM_MEASUREMENT"], "UNAVAILABLE")
        self.assertIn("DEVICE_GLOBAL_VRAM_OBSERVATIONS", result)
        self.assertNotIn("s" * 32, json.dumps(result))

    def test_unmeasurable_server_metrics_remain_null_in_summary(self):
        summary = benchmark_ws.summarize([{
            "MODEL_INFERENCE_MS": None,
            "SEGMENT_WAIT_MS": None,
            "SERVER_TO_TRANSCRIPT_MS": None,
            "SERVER_RECEIVE_TO_TRANSCRIPT_MS": None,
            "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": None,
            "NETWORK_RTT_MS": None,
            "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS": None,
            "MODEL_INFERENCE_RTF": None,
            "END_TO_END_RTF": None,
            "errors": [],
        }])
        self.assertIsNone(summary["MODEL_INFERENCE_MS_P50"])
        self.assertIsNone(summary["SEGMENT_WAIT_MS_P95"])
        self.assertIsNone(summary["SERVER_AUDIO_END_TO_TRANSCRIPT_MS_P50"])
        self.assertIsNone(summary["NETWORK_RTT_MS_P50"])

    async def test_no_transcript_keeps_audio_latency_and_end_to_end_rtf_null(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "silent.wav"
            with wave.open(str(fixture), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes((0).to_bytes(2, "little", signed=True) * 1_600)

            socket = FakeWebSocket(1, transcript_payloads=[])
            fake_websockets = types.ModuleType("websockets")
            fake_websockets.connect = lambda *_args, **_kwargs: socket

            def open_health(*_args, **_kwargs):
                return FakeHTTPResponse(json.dumps({"backend": "parakeet"}).encode("utf-8"))

            with patch.dict(sys.modules, {"websockets": fake_websockets}), patch.object(
                benchmark_ws.urllib.request, "urlopen", side_effect=open_health
            ):
                result = await benchmark_ws.run_one("wss://salad.example/asr/ws", "s" * 32, fixture)

        self.assertEqual(result["segment_count"], 0)
        self.assertIsNone(result["TOTAL_AUDIO_END_TO_TRANSCRIPT_MS"])
        self.assertIsNone(result["END_TO_END_RTF"])


def urlsplit_path(url: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(url).path


if __name__ == "__main__":
    unittest.main()
