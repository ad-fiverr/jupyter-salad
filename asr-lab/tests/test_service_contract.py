from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import types
import unittest
from unittest.mock import patch

from asr_lab.config import Settings


TOKEN = "t" * 32


class FakeJSONResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


class FakeFastAPI:
    def __init__(self, **kwargs):
        self.title = kwargs.get("title")
        self.routes = {}

    def _register(self, method, path):
        def decorator(function):
            self.routes[(method, path)] = function
            return function
        return decorator

    def get(self, path):
        return self._register("GET", path)

    def websocket(self, path):
        return self._register("WEBSOCKET", path)


def load_service_module():
    sys.modules.pop("asr_lab.service", None)
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.Query = lambda default=None: default
    fastapi.WebSocket = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = FakeJSONResponse
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


def audio_frame(value=8000, source="mic"):
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
        socket = FakeWebSocket([audio_frame(), flush_frame(), {"type": "websocket.disconnect"}])
        await service.websocket_asr(socket, token=TOKEN)
        transcripts = [item for item in socket.sent if item.get("event") == "transcript"]
        self.assertEqual(len(transcripts), 1)
        self.assertEqual(transcripts[0]["text"], "prueba de audio")
        self.assertIn("MODEL_INFERENCE_MS", transcripts[0])
        self.assertIn("SERVER_TO_TRANSCRIPT_MS", transcripts[0])
        self.assertIn("SERVER_RECEIVE_TO_TRANSCRIPT_MS", transcripts[0])
        self.assertFalse(socket.closed)

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
