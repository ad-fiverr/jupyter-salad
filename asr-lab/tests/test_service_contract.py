from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_lab.config import Settings


TOKEN = "t" * 32


class FakeJSONResponse:
    def __init__(self, content, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class FakeFileResponse:
    def __init__(self, path, headers=None):
        self.path = path
        self.headers = headers or {}


class FakeFastAPI:
    def __init__(self, **kwargs):
        self.title = kwargs.get("title")
        self.routes = {}

    def _register(self, method, path):
        def decorator(function):
            self.routes[(method, path)] = function
            return function
        return decorator

    def get(self, path, **kwargs):
        return self._register("GET", path)

    def websocket(self, path):
        return self._register("WEBSOCKET", path)


def load_service_module():
    sys.modules.pop("asr_lab.service", None)
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.Query = lambda default=None: default
    fastapi.Header = lambda default=None: default
    fastapi.WebSocket = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = FakeJSONResponse
    responses.FileResponse = FakeFileResponse
    fastapi.responses = responses
    with patch.dict(sys.modules, {"fastapi": fastapi, "fastapi.responses": responses}):
        return importlib.import_module("asr_lab.service")


class FakeBroker:
    def __init__(self, ready=True, block=False, connection_owned_text=False):
        self.ready = ready
        self.connection_owned_text = connection_owned_text
        self.worker_metrics = []
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not block:
            self.release.set()

    async def transcribe(self, **kwargs):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return {
            "type": "transcript",
            "event": "transcript",
            "text": f"resultado-{kwargs['connection_id']}" if self.connection_owned_text else "prueba de audio",
            "speaker": kwargs["speaker"],
            "source": kwargs["source"],
            "MODEL_INFERENCE_MS": 2.0,
            "SEGMENT_WAIT_MS": kwargs["segment_wait_ms"],
            "SERVER_TO_TRANSCRIPT_MS": 3.0,
        }


class FakeWebSocket:
    def __init__(self, frames=()):
        self.frames = list(frames)
        self.closed = []
        self.accepted = False
        self.sent = []
        self.received = 0

    async def close(self, code, reason):
        self.closed.append((code, reason))

    async def accept(self):
        self.accepted = True

    async def receive(self):
        await asyncio.sleep(0)
        self.received += 1
        if self.frames:
            return self.frames.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)


def audio_frame(value=8000, source="mic", request_id=None):
    import base64
    import struct

    pcm = struct.pack("<h", value) * 9_600  # 600 ms, above historical 500 ms minimum
    payload = {
        "source": source,
        "speaker": "you" if source == "mic" else "them",
        "encoding": "pcm_int16",
        "sample_rate": 16000,
        "audio": base64.b64encode(pcm).decode("ascii"),
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return {"type": "websocket.receive", "text": json.dumps(payload)}


def flush_frame(request_id=None):
    payload = {"event": "flush", "source": "mic"}
    if request_id is not None:
        payload["request_id"] = request_id
    return {"type": "websocket.receive", "text": json.dumps(payload)}


class ServiceContractTests(unittest.IsolatedAsyncioTestCase):
    def setup_service(self, *, max_connections=8, max_pending=2, ready=True, block=False):
        service = load_service_module()
        service.settings = Settings.from_env({"ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN})
        service.config_valid = True
        service.connection_slots = asyncio.Semaphore(max_connections)
        service.broker = FakeBroker(ready=ready, block=block)
        return service

    async def test_missing_and_wrong_tokens_close_before_acceptance(self):
        service = self.setup_service()
        for supplied in (None, "wrong-token" * 4):
            socket = FakeWebSocket([audio_frame()])
            await service.websocket_asr(socket, token=supplied)
            self.assertEqual(socket.closed[0][0], 4401)
            self.assertFalse(socket.accepted)
            self.assertEqual(socket.received, 0)

    async def test_valid_token_receives_transcript_using_service_handler(self):
        service = self.setup_service()
        socket = FakeWebSocket([audio_frame(request_id="segment-1"), flush_frame(), {"type": "websocket.disconnect"}])
        await service.websocket_asr(socket, token=TOKEN)
        transcripts = [item for item in socket.sent if item.get("event") == "transcript"]
        self.assertEqual(len(transcripts), 1)
        self.assertEqual(transcripts[0]["text"], "prueba de audio")
        self.assertIn("MODEL_INFERENCE_MS", transcripts[0])
        self.assertIn("SERVER_TO_TRANSCRIPT_MS", transcripts[0])
        self.assertIn("SERVER_RECEIVE_TO_TRANSCRIPT_MS", transcripts[0])
        self.assertIn("SERVER_AUDIO_END_TO_TRANSCRIPT_MS", transcripts[0])
        self.assertEqual(transcripts[0]["client_request_id"], "segment-1")
        self.assertFalse(socket.closed)

    async def test_server_audio_end_metric_uses_server_monotonic_clock(self):
        service = self.setup_service()
        service.broker = FakeBroker()

        class Audio:
            can_flush = True
            duration_seconds = 0.5
            has_voice = True
            speaker = "you"
            started_at = 10.0
            last_voice_at = 10.2
            request_id = "browser-segment-7"

            def take(self):
                return b"\x01\x00" * 8_000

        socket = FakeWebSocket()
        with patch("asr_lab.service.time.perf_counter", return_value=10.7):
            await service._flush(socket, "client-a", "mic", Audio(), 10.4, 10.0, socket.send_json)
        self.assertAlmostEqual(socket.sent[0]["SERVER_AUDIO_END_TO_TRANSCRIPT_MS"], 500.0, places=2)
        self.assertEqual(socket.sent[0]["client_request_id"], "browser-segment-7")

    async def test_benchmark_route_serves_allowlisted_page_assets_with_security_headers(self):
        service = self.setup_service()
        page = await service.benchmark_page()
        self.assertEqual(Path(page.path).name, "index.html")
        self.assertTrue(Path(page.path).is_file())
        self.assertEqual(page.headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", page.headers["Content-Security-Policy"])

        for name in ("app.mjs", "audio-worklet.mjs", "core.mjs", "style.css"):
            asset = await service.benchmark_asset(name)
            self.assertTrue(Path(asset.path).is_file())
        denied = await service.benchmark_asset("../service.py")
        self.assertEqual(denied.status_code, 404)

    async def test_benchmark_static_assets_do_not_persist_or_embed_token(self):
        service = self.setup_service()
        page = Path(service.BENCHMARK_WEB_ROOT, "index.html").read_text(encoding="utf-8")
        app = Path(service.BENCHMARK_WEB_ROOT, "app.mjs").read_text(encoding="utf-8")
        worklet = Path(service.BENCHMARK_WEB_ROOT, "audio-worklet.mjs").read_text(encoding="utf-8")
        self.assertIn('id="api-token" type="password"', page)
        self.assertIn("new AudioWorkletNode", app)
        self.assertIn("sample_rate: 16_000", app)
        self.assertIn("CHUNK_SAMPLES = 1_600", Path(service.BENCHMARK_WEB_ROOT, "core.mjs").read_text(encoding="utf-8"))
        self.assertIn('this.port.postMessage({ type: "flushed" })', worklet)
        for forbidden in ("localStorage", "sessionStorage", "document.cookie"):
            self.assertNotIn(forbidden, app)
        self.assertNotIn("state.token", app[app.index("function safeResult"):app.index("function download")])
        self.assertNotIn(TOKEN, app + page)
        stop_run = app[app.index("async function stopRun"):app.index("async function cleanup")]
        self.assertLess(stop_run.index("state.isStopping = true"), stop_run.index("await waitForWorkletFlush()"))
        self.assertIn("!state.isRecording && !state.isStopping", app)

    async def test_telemetry_endpoint_requires_bearer_and_returns_only_allowlisted_snapshot(self):
        service = self.setup_service()
        denied = await service.telemetry(authorization=None)
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(denied.headers["Cache-Control"], "no-store")

        snapshot = {
            "schema_version": 1, "backend": "parakeet", "model_id": "nvidia/parakeet-tdt-0.6b-v3",
            "model_loaded": True, "ready": True, "workers": 1, "queue_depth": 0,
            "process_rss_mib": 123.4, "system_ram": {"total_mib": 1000.0, "used_mib": 500.0},
            "gpu": {"available": False},
        }
        with patch.object(service, "collect_telemetry", return_value=snapshot):
            allowed = await service.telemetry(authorization=f"Bearer {TOKEN}")
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["Cache-Control"], "no-store")
        self.assertEqual(set(allowed.content), set(snapshot))
        self.assertNotIn(TOKEN, json.dumps(allowed.content))

    async def test_fixture_flush_barrier_follows_every_transcript(self):
        service = self.setup_service()
        socket = FakeWebSocket([
            audio_frame(),
            flush_frame(request_id="fixture-42"),
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        events = [item.get("event") for item in socket.sent]
        self.assertEqual(events, ["transcript", "flush_complete"])
        self.assertEqual(socket.sent[-1]["request_id"], "fixture-42")

    async def test_each_audio_source_has_its_own_600ms_inactivity_flush(self):
        service = self.setup_service()

        class InterleavedWebSocket(FakeWebSocket):
            async def receive(self):
                if self.received:
                    await asyncio.sleep(0.1)
                return await super().receive()

        socket = InterleavedWebSocket([
            audio_frame(source="mic"),
            *[audio_frame(source="system") for _ in range(7)],
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        mic_transcripts = [
            item for item in socket.sent
            if item.get("event") == "transcript" and item.get("source") == "mic"
        ]
        self.assertEqual(len(mic_transcripts), 1)

    async def test_service_sends_each_broker_result_only_to_its_own_websocket(self):
        service = self.setup_service()
        service.broker = FakeBroker(connection_owned_text=True)

        class Audio:
            duration_seconds = 0.3
            has_voice = True
            can_flush = True
            speaker = "you"
            started_at = 10.0
            last_voice_at = 10.3

            def take(self):
                return b"\x01\x00" * 4_800

        socket_a = FakeWebSocket()
        socket_b = FakeWebSocket()
        await asyncio.gather(
            service._flush(socket_a, "client-a", "mic", Audio(), 10.4, 10.0, socket_a.send_json),
            service._flush(socket_b, "client-b", "system", Audio(), 10.4, 10.0, socket_b.send_json),
        )

        self.assertEqual([item["text"] for item in socket_a.sent], ["resultado-client-a"])
        self.assertEqual([item["text"] for item in socket_b.sent], ["resultado-client-b"])
        self.assertNotIn("resultado-client-b", json.dumps(socket_a.sent))
        self.assertNotIn("resultado-client-a", json.dumps(socket_b.sent))

    async def test_historical_parakeet_blacklist_output_is_not_sent_to_client(self):
        service = self.setup_service()

        class BlacklistBroker(FakeBroker):
            async def transcribe(self, **kwargs):
                return {
                    "type": "transcript",
                    "event": "transcript",
                    "text": "You!",
                    "_discarded_historical_output": True,
                }

        service.broker = BlacklistBroker()

        class Audio:
            can_flush = True
            duration_seconds = 0.5
            has_voice = True
            speaker = "you"
            started_at = 10.0
            last_voice_at = 10.3

            def take(self):
                return b"\x01\x00" * 8_000

        socket = FakeWebSocket()
        await service._flush(socket, "client-a", "mic", Audio(), 10.4, 10.0, socket.send_json)
        self.assertEqual(socket.sent, [])

    async def test_connection_limit_rejects_excess_connection(self):
        service = self.setup_service(max_connections=0)
        socket = FakeWebSocket([audio_frame()])
        await service.websocket_asr(socket, token=TOKEN)
        self.assertEqual(socket.closed[0][0], 1013)
        self.assertEqual(socket.closed[0][1], "connection_limit")
        self.assertFalse(socket.accepted)

    async def test_pending_job_limit_is_enforced_per_connection(self):
        service = self.setup_service(block=True)
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet",
            "ASR_API_TOKEN": TOKEN,
            "ASR_MAX_PENDING_PER_CONNECTION": "1",
        })
        socket = FakeWebSocket([
            audio_frame(), flush_frame(),
            audio_frame(), flush_frame(),
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        self.assertEqual(service.broker.calls, 1)
        self.assertIn("pending_limit", [item.get("code") for item in socket.sent])

    async def test_health_and_readiness_have_distinct_fake_model_states(self):
        service = self.setup_service(ready=False)
        live = await service.health()
        unready = await service.readiness()
        self.assertEqual(live["status"], "ok")
        self.assertFalse(live["model_loaded"])
        self.assertEqual(unready.status_code, 503)
        self.assertFalse(unready.content["ready"])

        service.broker.ready = True
        ready = await service.readiness()
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.content["ready"])

    async def test_control_heartbeat_json_is_not_treated_as_audio(self):
        service = self.setup_service()
        socket = FakeWebSocket([
            {"type": "websocket.receive", "text": json.dumps({"event": "ping"})},
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        self.assertEqual(service.broker.calls, 0)
        self.assertIn("unsupported_fields", [item.get("code") for item in socket.sent])

    async def test_authentication_does_not_log_the_query_token(self):
        service = self.setup_service()
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Capture()
        service.logger.addHandler(handler)
        try:
            await service.websocket_asr(FakeWebSocket(), token=TOKEN)
        finally:
            service.logger.removeHandler(handler)
        self.assertTrue(all(TOKEN not in message for message in records))


if __name__ == "__main__":
    unittest.main()
