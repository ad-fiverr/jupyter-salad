from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import types
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_lab.config import Settings
from asr_lab.qwen_streaming import QwenStreamingRuntime


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


def qwen_audio_frame(duration_ms=100, source="mic"):
    import base64
    import struct

    samples = int(16_000 * duration_ms / 1000)
    pcm = struct.pack("<h", 8000) * samples
    payload = {
        "source": source,
        "speaker": "you" if source == "mic" else "them",
        "encoding": "pcm_int16",
        "sample_rate": 16000,
        "audio": base64.b64encode(pcm).decode("ascii"),
    }
    return {"type": "websocket.receive", "text": json.dumps(payload)}


class FlushHandoffWorker:
    """Deterministically hold one worker RPC while the real lifecycle rolls over."""

    def __init__(self):
        self.ready = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block_at_push = 46
        self.push_count = 0
        self.requests = []

    async def start(self, **_options):
        self.ready = True
        return {"model_load_ms": 1.0, "vram_after_warmup": {"available": False}}

    async def request(self, operation, **payload):
        self.requests.append((operation, dict(payload)))
        stream_id = payload.get("stream_id")
        if operation == "init":
            return {"stream_id": stream_id, "stream_state_init_wall_ms": 1.0}
        if operation == "close":
            return {"closed": True}
        if operation in {"push", "finish"}:
            if operation == "push":
                self.push_count += 1
                if self.push_count == self.block_at_push:
                    self.started.set()
                    await self.release.wait()
            return {
                "scheduler_key": payload.get("scheduler_key"),
                "decoded": True,
                "decode_wall_ms": 2.0,
                "decode_steps_delta": 1,
                "text": "candidate-" + str(stream_id),
                "language": "Spanish",
            }
        raise AssertionError(operation)

    async def close(self):
        self.ready = False


class CapacityWaitWebSocket(FakeWebSocket):
    def __init__(self, worker, registry_instances, *, release_delay=0.05):
        super().__init__()
        self.worker = worker
        self.registry_instances = registry_instances
        self.release_delay = release_delay
        self.step = 0
        self.blocked_reason = None
        self.handoff_state = None
        self.flush_received_before_capacity_release = False
        self.flush_received_at = None
        self.service_completed_at = None
        self.lifecycle = None
        self.error_before_capacity_release = False

    async def wait_until(self, predicate, message):
        for _ in range(2_000):
            if predicate():
                return
            await asyncio.sleep(0.001)
        raise AssertionError(message)

    async def receive(self):
        await asyncio.sleep(0)
        self.step += 1
        if self.step == 1:
            return {"type": "websocket.receive", "text": json.dumps({
                "event": "stream_start", "source": "mic", "language": "auto",
                "model_chunk_ms": 100, "request_id": "start-c4",
            })}
        audio_index = self.step - 1
        if 1 <= audio_index <= 51:
            registry = self.registry_instances[-1]
            lifecycle = registry._sources["mic"].lifecycle
            if 2 <= audio_index <= 46:
                await self.wait_until(
                    lambda: lifecycle._observed_local_processed_cursor >= (audio_index - 1) * 1_600,
                    "controlled worker did not drain the preceding 100 ms audio window",
                )
            elif audio_index == 47:
                await asyncio.wait_for(self.worker.started.wait(), timeout=2)
            return qwen_audio_frame(100)
        if self.step == 53:
            registry = self.registry_instances[-1]
            self.lifecycle = registry._sources["mic"].lifecycle
            observability = self.lifecycle.observability_snapshot()
            transition = observability["last_transition"]
            self.handoff_state = transition["state"]
            self.blocked_reason = transition["failure_reason"]
            self.flush_received_before_capacity_release = not self.worker.release.is_set()
            self.flush_received_at = time.monotonic()
            if self.release_delay is not None:
                asyncio.get_running_loop().call_later(self.release_delay, self.worker.release.set)
            return flush_frame("flush-c4")
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        if payload.get("code") == "stream_handoff_incomplete":
            self.error_before_capacity_release = not self.worker.release.is_set()
        await super().send_json(payload)


class ServiceContractTests(unittest.IsolatedAsyncioTestCase):
    def setup_service(self, *, max_connections=8, max_pending=2, ready=True, block=False):
        service = load_service_module()
        service.settings = Settings.from_env({"ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN})
        service.config_valid = True
        service.connection_slots = asyncio.Semaphore(max_connections)
        service.broker = FakeBroker(ready=ready, block=block)
        return service

    async def run_capacity_wait_service_case(self, *, release_delay, deadline_seconds=None):
        service = self.setup_service()
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        })
        worker = FlushHandoffWorker()
        runtime = QwenStreamingRuntime(
            worker=worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="fixed",
            gpu_memory_utilization=0.65,
            max_active_sessions=6,
            max_pending_jobs=24,
            max_backlog_chunks=4,
            max_stream_seconds=5.0,
            session_idle_ttl_seconds=120.0,
            max_context_chars=512,
            default_chunk_ms=100,
            default_language="auto",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        await runtime.start()
        service.qwen_runtime = runtime
        registry_instances = []
        real_registry = service.QwenServiceLifecycleRegistry

        class CapturingRegistry(real_registry):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                registry_instances.append(self)

        service.QwenServiceLifecycleRegistry = CapturingRegistry
        socket = CapacityWaitWebSocket(worker, registry_instances, release_delay=release_delay)
        started_at = time.monotonic()
        try:
            if deadline_seconds is None:
                await asyncio.wait_for(service.websocket_asr(socket, token=TOKEN), timeout=8)
            else:
                with patch.dict(real_registry.finish.__globals__, {
                    "_HANDOFF_FLUSH_DRAIN_DEADLINE_SECONDS": deadline_seconds,
                }):
                    await asyncio.wait_for(service.websocket_asr(socket, token=TOKEN), timeout=8)
            socket.service_completed_at = time.monotonic()
            elapsed = time.monotonic() - started_at
            return socket, worker, elapsed
        finally:
            worker.release.set()
            await runtime.close()

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

        class FakeQwenRuntime(service.QwenStreamingRuntime):
            ready = True
            max_stream_seconds = 60.0

            def __init__(self):
                self.active = None

            async def open_session(self, *, connection_id, source, event_sink, **kwargs):
                self.connection_id = connection_id
                self.source = source
                self.event_sink = event_sink
                self.candidate_result_filter = kwargs.get("candidate_result_filter")
                self.stream_id = "owned-stream"
                self.active = True
                return {"event": "stream_started", "stream_id": self.stream_id, "request_id": kwargs.get("request_id")}

            async def push_audio(self, *, connection_id, source, pcm16le, allow_capacity_wait=False):
                self.assert_owner(connection_id, source, pcm16le)
                candidate = {
                    "event": "partial_candidate", "stream_id": self.stream_id,
                    "text": "provisional", "final": False, "provisional": True,
                    "replace": True, "truth_status": "candidate_only", "revision": 1,
                }
                candidate = self.candidate_result_filter(candidate)
                if candidate is not None:
                    await self.event_sink(candidate)

            def assert_owner(self, connection_id, source, pcm16le):
                assert connection_id == self.connection_id and source == self.source and pcm16le

            async def finish(self, *, connection_id, source, request_id, **kwargs):
                self.assert_owner(connection_id, source, b"pcm")
                candidate = {
                    "event": "final_candidate", "stream_id": self.stream_id,
                    "text": "final candidate", "final": True, "provisional": True,
                    "candidate_only": True, "truth_status": "candidate_only",
                    "replace": True, "revision": 2, "request_id": request_id,
                }
                candidate = self.candidate_result_filter(candidate)
                self.active = False
                return candidate

            async def close_session(self, *, connection_id, source):
                self.assert_owner(connection_id, source, b"pcm")
                self.active = False

            async def close_connection(self, _connection_id):
                return None

        service.qwen_runtime = FakeQwenRuntime()
        socket = FakeWebSocket([
            audio_frame(),
            flush_frame("flush-1"),
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        events = [message.get("event") for message in socket.sent]
        self.assertEqual(events, ["stream_started", "partial_candidate", "final_candidate", "flush_complete"])
        self.assertEqual(socket.sent[1]["stream_id"], "owned-stream")
        self.assertEqual(socket.sent[2]["request_id"], "flush-1")

    async def test_qwen_websocket_flush_waits_for_capacity_during_pending_handoff(self):
        socket, worker, _elapsed = await self.run_capacity_wait_service_case(release_delay=0.05)
        self.assertEqual(socket.handoff_state, "HANDOFF_PENDING")
        self.assertEqual(socket.blocked_reason, "stream_scheduler_capacity_wait")
        self.assertTrue(socket.flush_received_before_capacity_release)
        self.assertFalse(socket.error_before_capacity_release)

        events = [item.get("event") for item in socket.sent]
        self.assertEqual(events.count("final_candidate"), 1)
        self.assertNotIn("stream_handoff_incomplete", [item.get("code") for item in socket.sent])
        self.assertLess(events.index("final_candidate"), events.index("flush_complete"))
        final = next(item for item in socket.sent if item.get("event") == "final_candidate")
        self.assertTrue(final["final"])
        self.assertEqual(final["truth_status"], "candidate_only")
        self.assertEqual(sum(operation == "finish" for operation, _ in worker.requests), 1)

        pcm = socket.lifecycle.pcm_handoff.snapshot()
        self.assertEqual(pcm.received_samples, 81_600)
        self.assertEqual(pcm.unique_primary_admitted_samples, 81_600)
        self.assertEqual(pcm.replay_admitted_samples, 8_000)
        # Known scheduler capacity rejections are retained for explicit drain;
        # only the admitted replay total proves whether any PCM was duplicated.
        self.assertGreater(pcm.downstream_admission_rejected_samples, 0)
        self.assertFalse(pcm.handoff_in_progress)

    async def test_qwen_websocket_flush_capacity_wait_times_out_fail_closed(self):
        socket, worker, _elapsed = await self.run_capacity_wait_service_case(
            release_delay=None,
            deadline_seconds=0.05,
        )
        self.assertEqual(socket.handoff_state, "HANDOFF_PENDING")
        self.assertEqual(socket.blocked_reason, "stream_scheduler_capacity_wait")
        self.assertTrue(socket.error_before_capacity_release)
        flush_wait = socket.service_completed_at - socket.flush_received_at
        self.assertGreaterEqual(flush_wait, 0.04)
        self.assertLessEqual(flush_wait, 0.5)
        events = [item.get("event") for item in socket.sent]
        codes = [item.get("code") for item in socket.sent]
        self.assertIn("stream_handoff_incomplete", codes)
        self.assertNotIn("final_candidate", events)
        self.assertLess(events.index("error"), events.index("flush_complete"))
        self.assertEqual(sum(operation == "finish" for operation, _ in worker.requests), 0)

    async def test_qwen_terminal_error_keeps_lifecycle_snapshot_before_cleanup(self):
        service = self.setup_service()
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        })

        class Runtime:
            ready = True

            async def close_connection(self, _connection_id):
                return None

        diagnostics = {
            "qwen_public_stream_id": "public-stream",
            "qwen_local_stream_id": "qwen-local-epoch-2",
            "qwen_epoch_observability": {
                "current_epoch": {"epoch_id": "epoch-2", "epoch_seq": 2},
                "last_transition": {
                    "state": "HANDOFF_PENDING", "stage": "CATCHUP_DRAINING",
                    "failure_stage": "catchup",
                },
                "logical_cumulative": {"EPOCH_ROLLOVER_COUNT": 1},
                "epoch_history": [], "transition_history": [],
            },
        }
        order = []

        class Registry:
            def __init__(self, **_kwargs):
                self.active = False

            async def open_source(self, **_kwargs):
                self.active = True
                return {"event": "stream_started", "stream_id": "public-stream"}

            def has_active_source(self, _source):
                return self.active

            async def submit_pcm(self, **_kwargs):
                raise service.StreamingError("transition_capacity_exceeded")

            def source_observability_snapshot(self, _source):
                order.append("snapshot")
                return diagnostics

            async def abort_source(self, _source):
                order.append("cleanup")
                self.active = False

            async def close_connection(self):
                self.active = False

        service.qwen_runtime = Runtime()
        socket = FakeWebSocket([
            {"type": "websocket.receive", "text": json.dumps({
                "event": "stream_start", "source": "mic", "request_id": "start-1",
                "language": "auto", "context": "", "model_chunk_ms": 250,
            })},
            audio_frame(),
            {"type": "websocket.disconnect"},
        ])
        with patch.object(service, "QwenServiceLifecycleRegistry", Registry):
            await service.websocket_asr(socket, token=TOKEN)

        error = next(item for item in socket.sent if item.get("event") == "error")
        self.assertEqual(order, ["snapshot", "cleanup"])
        self.assertEqual(error["qwen_public_stream_id"], "public-stream")
        self.assertEqual(error["qwen_local_stream_id"], "qwen-local-epoch-2")
        self.assertEqual(error["qwen_epoch_observability"]["last_transition"]["stage"], "CATCHUP_DRAINING")

    async def test_six_qwen_websocket_clients_are_isolated_and_overrun_rolls_epoch_in_place(self):
        service = self.setup_service(max_connections=8)
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
            "QWEN_MAX_ACTIVE_STREAMS": "6",
        })

        class FakeQwenRuntime(service.QwenStreamingRuntime):
            ready = True
            max_stream_seconds = 60.0

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
                    "candidate_result_filter": kwargs.get("candidate_result_filter"),
                }
                self.sessions[(connection_id, source)] = session
                self.opened.append(session)
                return {"event": "stream_started", "stream_id": stream_id, "request_id": request_id}

            async def push_audio(self, *, connection_id, source, pcm16le, allow_capacity_wait=False):
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
                candidate = {
                    "event": "partial_candidate", "stream_id": session["stream_id"],
                    "text": session["context"], "final": False, "provisional": True,
                    "replace": True, "truth_status": "candidate_only", "revision": 1,
                }
                candidate = session["candidate_result_filter"](candidate)
                if candidate is not None:
                    await session["event_sink"](candidate)

            async def finish(self, *, connection_id, source, request_id, **kwargs):
                session = self.sessions[(connection_id, source)]
                candidate = {
                    "event": "final_candidate", "stream_id": session["stream_id"],
                    "text": session["context"] + " final", "final": True,
                    "provisional": True, "candidate_only": True,
                    "truth_status": "candidate_only", "replace": True,
                    "revision": 2, "request_id": request_id,
                }
                candidate = session["candidate_result_filter"](candidate)
                self.sessions.pop((connection_id, source), None)
                return candidate

            async def close_session(self, *, connection_id, source):
                self.sessions.pop((connection_id, source), None)

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
                    audio_frame(),
                    flush_frame("flush-0"),
                ])
            else:
                frames.append(flush_frame(f"flush-{index}"))
            frames.append({"type": "websocket.disconnect"})
            sockets.append(FakeWebSocket(frames))

        await asyncio.gather(*(service.websocket_asr(socket, token=TOKEN) for socket in sockets))

        self.assertEqual(len(runtime.opened), 7)
        self.assertEqual(len({item["connection_id"] for item in runtime.opened}), 6)
        self.assertEqual(sum(item["context"] == "cliente-0-original" for item in runtime.opened), 2)
        for index, socket in enumerate(sockets):
            candidates = [item for item in socket.sent if item.get("event") in {"partial_candidate", "final_candidate"}]
            payload = json.dumps(candidates, ensure_ascii=False)
            for other in range(6):
                if index != other:
                    self.assertNotIn(f"cliente-{other}-", payload)
            if index == 0:
                self.assertFalse({"stream_scheduler_overrun", "stream_terminal"} & {item.get("code") for item in socket.sent})
                self.assertEqual([item["text"] for item in candidates], [
                    "cliente-0-original", "cliente-0-original final",
                ])
                started = next(item for item in socket.sent if item.get("event") == "stream_started")
                self.assertEqual(len([item for item in socket.sent if item.get("event") == "stream_started"]), 1)
                self.assertEqual(len({item["stream_id"] for item in candidates}), 1)
                self.assertEqual(len({item["qwen_local_stream_id"] for item in candidates}), 1)
                self.assertNotEqual(candidates[-1]["qwen_local_stream_id"], started["qwen_local_stream_id"])
                self.assertFalse(any("must-not-leak" in json.dumps(item) for item in socket.sent))
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


    async def test_conflicting_model_chunk_fields_fail_before_runtime_open(self):
        service = self.setup_service()
        service.settings = Settings.from_env({
            "ASR_BACKEND": "parakeet", "ASR_API_TOKEN": TOKEN,
            "QWEN_STREAMING_ENABLED": "1", "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
        })
        class Runtime:
            ready = True
            opened = 0
            async def open_session(self, **_kwargs):
                self.opened += 1
                return {"event": "stream_started"}
            async def close_connection(self, _connection_id):
                return None
        runtime = Runtime()
        service.qwen_runtime = runtime
        socket = FakeWebSocket([
            {"type": "websocket.receive", "text": json.dumps({
                "event": "stream_start", "source": "mic", "model_chunk_ms": 50,
                "chunk_size_ms": 100,
            })},
            {"type": "websocket.disconnect"},
        ])
        await service.websocket_asr(socket, token=TOKEN)
        errors = [item for item in socket.sent if item.get("event") == "error"]
        self.assertEqual(errors[0]["code"], "conflicting_model_chunk")
        self.assertEqual(runtime.opened, 0)


if __name__ == "__main__":
    unittest.main()
