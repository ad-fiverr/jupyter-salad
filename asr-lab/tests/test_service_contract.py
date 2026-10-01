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
    starlette = types.ModuleType("starlette")
    starlette_responses = types.ModuleType("starlette.responses")
    starlette_responses.Response = object
    starlette.responses = starlette_responses
    with patch.dict(sys.modules, {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "starlette": starlette,
        "starlette.responses": starlette_responses,
    }):
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
            "SERVER_MODEL_INFERENCE_MS": 2.0,
            "SERVER_ENDPOINTING_MS": kwargs["segment_wait_ms"],
            "SERVER_QUEUE_WAIT_MS": 0.02,
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
        self.assertIn("SERVER_EOS_TO_TRANSCRIPT_MS", transcripts[0])
        self.assertIn("SERVER_POSTPROCESS_MS", transcripts[0])
        self.assertNotIn("_internal_timing", transcripts[0])
        self.assertEqual(transcripts[0]["client_request_id"], "segment-1")
        self.assertFalse(socket.closed)

    async def test_benchmark_ping_pong_is_authenticated_and_does_not_interfere_with_audio_flush(self):
        service = self.setup_service()
        ping = {"event": "benchmark_ping", "request_id": "rtt-sample-1"}
        socket = FakeWebSocket([
            {"type": "websocket.receive", "text": json.dumps(ping)},
            audio_frame(request_id="segment-after-ping"),
            flush_frame("flush-after-ping"),
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        self.assertEqual(socket.sent[0], {"event": "benchmark_pong", "request_id": "rtt-sample-1"})
        self.assertEqual(service.broker.calls, 1)
        transcript = next(item for item in socket.sent if item.get("event") == "transcript")
        self.assertEqual(transcript["client_request_id"], "segment-after-ping")
        self.assertIn({"event": "flush_complete", "request_id": "flush-after-ping"}, socket.sent)

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

    async def test_server_metric_components_reconcile_without_audio_duration_or_client_clock(self):
        service = self.setup_service()

        class TimingBroker:
            ready = True

            async def transcribe(self, **_kwargs):
                return {
                    "event": "transcript", "type": "transcript", "text": "segmento siete",
                    "AUDIO_DURATION_MS": 2_000.0, "audio_duration_ms": 2_000.0,
                    "SERVER_ENDPOINTING_MS": 400.0,
                    "SERVER_QUEUE_WAIT_MS": 0.02,
                    "queue_wait_ms": 0.02,
                    "SERVER_MODEL_INFERENCE_MS": 112.0,
                    "MODEL_INFERENCE_MS": 112.0,
                    "_internal_timing": {"model_finished_at": 10.712},
                }

        service.broker = TimingBroker()

        class Audio:
            can_flush = True
            duration_seconds = 2.0
            has_voice = True
            speaker = "you"
            started_at = 8.2
            last_voice_at = 10.2
            request_id = "browser-segment-7"

            def take(self):
                return b"\x01\x00" * 16_000

        socket = FakeWebSocket()
        with patch("asr_lab.service.time.perf_counter", return_value=10.713):
            await service._flush(socket, "client-a", "mic", Audio(), 10.6, 0.0, socket.send_json)

        transcript = socket.sent[0]
        self.assertEqual(transcript["SERVER_ENDPOINTING_MS"], 400.0)
        self.assertEqual(transcript["SERVER_QUEUE_WAIT_MS"], 0.02)
        self.assertEqual(transcript["SERVER_MODEL_INFERENCE_MS"], 112.0)
        self.assertEqual(transcript["SERVER_POSTPROCESS_MS"], 1.0)
        self.assertEqual(transcript["SERVER_EOS_TO_TRANSCRIPT_MS"], 513.0)
        self.assertAlmostEqual(
            transcript["SERVER_EOS_TO_TRANSCRIPT_MS"],
            transcript["SERVER_ENDPOINTING_MS"] + transcript["SERVER_QUEUE_WAIT_MS"]
            + transcript["SERVER_MODEL_INFERENCE_MS"] + transcript["SERVER_POSTPROCESS_MS"],
            delta=0.03,
        )
        self.assertEqual(transcript["SERVER_RECEIVE_TO_TRANSCRIPT_MS"], 2_513.0)
        self.assertEqual(transcript["AUDIO_DURATION_MS"], 2_000.0)
        self.assertEqual(transcript["MODEL_INFERENCE_MS"], 112.0)
        self.assertNotIn("_internal_timing", transcript)

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
        core = Path(service.BENCHMARK_WEB_ROOT, "core.mjs").read_text(encoding="utf-8")
        worklet = Path(service.BENCHMARK_WEB_ROOT, "audio-worklet.mjs").read_text(encoding="utf-8")
        self.assertIn('id="api-token" type="password"', page)
        self.assertIn("new AudioWorkletNode", app)
        self.assertIn("sample_rate: 16_000", app)
        self.assertIn("CHUNK_SAMPLES = 1_600", Path(service.BENCHMARK_WEB_ROOT, "core.mjs").read_text(encoding="utf-8"))
        self.assertIn('this.port.postMessage({ type: "flushed" })', worklet)
        self.assertIn("Latencia del servidor / modelo", page)
        self.assertIn("Client / proxy path", page)
        self.assertIn("SERVER_RECEIVE_TO_TRANSCRIPT_MS", page)
        self.assertNotIn("GPU no disponible", app)
        self.assertIn("gpuStatusLabel", app + core)
        self.assertIn("record_type", app)
        self.assertIn("PROXY_WS_RTT_MS", app + core)
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

    async def test_qwen_websocket_pumps_owned_partial_then_final_before_flush_barrier(self):
        service = self.setup_service()
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        })

        class FakeQwenRuntime:
            ready = True

            async def open_session(self, *, connection_id, source, event_sink, **kwargs):
                self.connection_id = connection_id
                self.source = source
                self.event_sink = event_sink
                self.stream_id = "owned-stream"
                return {"event": "stream_started", "stream_id": self.stream_id, "request_id": kwargs.get("request_id")}

            async def push_audio(self, *, connection_id, source, pcm16le):
                self.assert_owner(connection_id, source, pcm16le)
                await self.event_sink({
                    "event": "partial_candidate", "stream_id": self.stream_id,
                    "text": "provisional", "final": False, "provisional": True,
                    "replace": True, "truth_status": "candidate_only", "revision": 1,
                })

            def assert_owner(self, connection_id, source, pcm16le):
                assert connection_id == self.connection_id and source == self.source and pcm16le

            async def finish(self, *, connection_id, source, request_id, **kwargs):
                self.assert_owner(connection_id, source, b"pcm")
                return {
                    "event": "final_candidate", "stream_id": self.stream_id,
                    "text": "final candidate", "final": True, "provisional": True,
                    "candidate_only": True, "truth_status": "candidate_only",
                    "replace": True, "revision": 2, "request_id": request_id,
                }

            async def close_connection(self, _connection_id):
                return None

        service.qwen_runtime = FakeQwenRuntime()
        start = {"event": "stream_start", "source": "mic", "request_id": "start-1", "language": "auto", "chunk_size_ms": 250}
        socket = FakeWebSocket([
            {"type": "websocket.receive", "text": json.dumps(start)},
            audio_frame(),
            flush_frame("flush-1"),
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        events = [message.get("event") for message in socket.sent]
        self.assertEqual(events, ["stream_started", "partial_candidate", "final_candidate", "flush_complete"])
        self.assertEqual(socket.sent[1]["stream_id"], "owned-stream")
        self.assertEqual(socket.sent[2]["request_id"], "flush-1")

    async def test_six_qwen_websocket_clients_are_isolated_and_overrun_requires_explicit_restart(self):
        service = self.setup_service(max_connections=8)
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
            "QWEN_MAX_ACTIVE_STREAMS": "6",
        })

        class FakeQwenRuntime:
            ready = True

            def __init__(self):
                self.sessions = {}
                self.opened = []
                self.overrun_raised = False
                self.closed = []

            async def open_session(self, *, connection_id, source, event_sink, context="", request_id=None, **kwargs):
                session_number = sum(item["connection_id"] == connection_id for item in self.opened) + 1
                stream_id = f"{connection_id[:8]}-{session_number}"
                session = {
                    "connection_id": connection_id, "source": source, "stream_id": stream_id,
                    "context": context, "event_sink": event_sink,
                }
                self.sessions[(connection_id, source)] = session
                self.opened.append(session)
                return {"event": "stream_started", "stream_id": stream_id, "request_id": request_id}

            async def push_audio(self, *, connection_id, source, pcm16le):
                session = self.sessions[(connection_id, source)]
                assert pcm16le
                if session["context"] == "cliente-0-original" and not self.overrun_raised:
                    self.overrun_raised = True
                    raise service.StreamingError("stream_scheduler_overrun", {
                        "reason": "per_stream_backlog_limit", "accepted_audio_ms": 250, "rejected_audio_ms": 600,
                        "terminal_metrics": {
                            "scheduler_wait_p50_ms": 12.5,
                            "scheduler_wait_sample_count": 3,
                            "decode_wall_p95_ms": 95.0,
                            "pending_jobs": 4,
                            "backlog_audio_ms": 500.0,
                            "accepted_audio_total_ms": 750.0,
                            "dispatched_audio_total_ms": 250.0,
                            "overrun_reason": "per_stream_backlog_limit",
                            "overrun_limit_kind": "backlog_ms",
                            "overrun_limit_value": 1000.0,
                            "connection_id": "must-not-leak",
                            "transcript": "must-not-leak",
                            "secret": "must-not-leak",
                        },
                    })
                await session["event_sink"]({
                    "event": "partial_candidate", "stream_id": session["stream_id"],
                    "text": session["context"], "final": False, "provisional": True,
                    "replace": True, "truth_status": "candidate_only", "revision": 1,
                })

            async def finish(self, *, connection_id, source, request_id, **kwargs):
                session = self.sessions[(connection_id, source)]
                return {
                    "event": "final_candidate", "stream_id": session["stream_id"],
                    "text": session["context"] + " final", "final": True,
                    "provisional": True, "candidate_only": True,
                    "truth_status": "candidate_only", "replace": True,
                    "revision": 2, "request_id": request_id,
                }

            async def close_connection(self, connection_id):
                self.closed.append(connection_id)
                for key in [key for key in self.sessions if key[0] == connection_id]:
                    self.sessions.pop(key, None)

        runtime = FakeQwenRuntime()
        service.qwen_runtime = runtime
        sockets = []
        for index in range(6):
            first_context = f"cliente-{index}-original"
            start = {"event": "stream_start", "source": "mic", "request_id": f"start-{index}",
                     "language": "auto", "context": first_context, "chunk_size_ms": 250}
            frames = [
                {"type": "websocket.receive", "text": json.dumps(start)},
                audio_frame(),
            ]
            if index == 0:
                frames.extend([
                    audio_frame(),
                    flush_frame("failed-stream-flush"),
                    {"type": "websocket.receive", "text": json.dumps({
                        "event": "stream_start", "source": "mic", "request_id": "restart-0",
                        "language": "auto", "context": "cliente-0-reinicio", "chunk_size_ms": 250,
                    })},
                    audio_frame(),
                    flush_frame("restart-flush-0"),
                ])
            else:
                frames.append(flush_frame(f"flush-{index}"))
            frames.append({"type": "websocket.disconnect"})
            sockets.append(FakeWebSocket(frames))

        await asyncio.gather(*(service.websocket_asr(socket, token=TOKEN) for socket in sockets))

        self.assertEqual(len(runtime.opened), 7)
        self.assertEqual(len({item["connection_id"] for item in runtime.opened}), 6)
        self.assertEqual(sum(item["context"] == "cliente-0-original" for item in runtime.opened), 1)
        self.assertEqual(sum(item["context"] == "cliente-0-reinicio" for item in runtime.opened), 1)
        for index, socket in enumerate(sockets):
            candidates = [item for item in socket.sent if item.get("event") in {"partial_candidate", "final_candidate"}]
            payload = json.dumps(candidates, ensure_ascii=False)
            for other in range(6):
                if index != other:
                    self.assertNotIn(f"cliente-{other}-", payload)
            if index == 0:
                self.assertIn("stream_scheduler_overrun", [item.get("code") for item in socket.sent])
                self.assertIn("stream_terminal", [item.get("code") for item in socket.sent])
                overrun = next(item for item in socket.sent if item.get("code") == "stream_scheduler_overrun")
                terminal_metrics = overrun["scheduler"]["terminal_metrics"]
                self.assertEqual(terminal_metrics["scheduler_wait_sample_count"], 3)
                self.assertEqual(terminal_metrics["decode_wall_p95_ms"], 95.0)
                self.assertEqual(terminal_metrics["accepted_audio_total_ms"], 750.0)
                self.assertFalse({"connection_id", "transcript", "secret"} & terminal_metrics.keys())
                self.assertEqual([item["text"] for item in candidates], [
                    "cliente-0-reinicio", "cliente-0-reinicio final",
                ])
                self.assertEqual(len([item for item in socket.sent if item.get("event") == "stream_started"]), 2)
                final_position = next(pos for pos, item in enumerate(socket.sent) if item.get("event") == "final_candidate")
                barrier_position = max(pos for pos, item in enumerate(socket.sent) if item.get("event") == "flush_complete")
                self.assertLess(final_position, barrier_position)
            else:
                events = [item.get("event") for item in socket.sent]
                self.assertEqual(events, ["stream_started", "partial_candidate", "final_candidate", "flush_complete"])
                self.assertEqual(candidates[0]["text"], f"cliente-{index}-original")
                self.assertEqual(candidates[1]["text"], f"cliente-{index}-original final")
                self.assertLess(events.index("final_candidate"), events.index("flush_complete"))

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
