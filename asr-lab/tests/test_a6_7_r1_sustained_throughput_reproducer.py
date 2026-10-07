from __future__ import annotations

import asyncio
import base64
import importlib
import json
import struct
import sys
import types
import unittest
from unittest.mock import patch

from asr_lab.asr_service_lifecycle import QwenServiceLifecycleRegistry
from asr_lab.config import Settings
from asr_lab.qwen_streaming import QwenStreamingRuntime


_TOKEN = "r1-diagnostic-token-not-a-credential"
_FRAME_SAMPLES = 1_600
_TRANSITION_SAMPLES = 80_000
_SCALED_TICK_MS = 5
_TICKS_PER_FRAME = 20


class _ScaledClock:
    """Deterministic virtual milliseconds advanced by continuous ingress."""

    def __init__(self) -> None:
        self.now_ms = 0.0
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    async def sleep_ms(self, delay_ms: float) -> None:
        if delay_ms <= 0:
            return
        future = asyncio.get_running_loop().create_future()
        self._waiters.append((self.now_ms + delay_ms, future))
        await future

    def advance(self, delta_ms: float) -> None:
        self.now_ms += delta_ms
        pending: list[tuple[float, asyncio.Future[None]]] = []
        for deadline, future in self._waiters:
            if deadline <= self.now_ms:
                if not future.done():
                    future.set_result(None)
            elif not future.done():
                pending.append((deadline, future))
        self._waiters = pending


class _ControlledWorker:
    """Worker fake with exact virtual decode duration and real RPC ordering."""

    def __init__(self, *, decode_ms: float, clock: _ScaledClock) -> None:
        self.decode_ms = decode_ms
        self.clock = clock
        self.ready = False
        self.push_count = 0
        self.init_count = 0

    async def start(self, **_options: object) -> dict[str, object]:
        self.ready = True
        return {"model_load_ms": 0.0}

    async def request(self, operation: str, **payload: object) -> dict[str, object]:
        if operation == "init":
            self.init_count += 1
            stream_id = payload["stream_id"]
            return {"stream_id": stream_id, "stream_state_init_wall_ms": 0.0}
        if operation == "push":
            self.push_count += 1
            await self.clock.sleep_ms(self.decode_ms)
            return {
                "scheduler_key": payload.get("scheduler_key"),
                "decode_wall_ms": self.decode_ms,
                "decode_steps_delta": 1,
                "text": f"candidate-{self.push_count}",
                "language": "English",
            }
        if operation == "close":
            return {"closed": True}
        raise AssertionError(f"unexpected worker operation: {operation}")

    async def close(self) -> None:
        self.ready = False


class _FakeWebSocket:
    """Continuous 100 ms PCM ingress over a bounded in-memory transport."""

    def __init__(self, *, frame_count: int, clock: _ScaledClock) -> None:
        self.frame_count = frame_count
        self.clock = clock
        self.sent: list[dict[str, object]] = []
        self._frame_index = 0
        pcm16le = struct.pack("<h", 1_000) * _FRAME_SAMPLES
        self._audio = base64.b64encode(pcm16le).decode("ascii")
        self.audio_frames_sent = 0

    async def accept(self) -> None:
        return None

    async def close(self, **_kwargs: object) -> None:
        return None

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    async def receive(self) -> dict[str, object]:
        if any(item.get("code") == "stream_terminal" for item in self.sent):
            return {"type": "websocket.disconnect"}

        if self._frame_index == 0:
            self._frame_index += 1
            return {
                "type": "websocket.receive",
                "text": json.dumps({
                    "event": "stream_start",
                    "source": "mic",
                    "language": "en",
                    "model_chunk_ms": 100,
                }),
            }

        if self.audio_frames_sent >= self.frame_count:
            return {"type": "websocket.disconnect"}

        # First PCM frame is the t=0 boundary; every later frame advances the
        # virtual clock by exactly 100 ms before reaching the service.
        if self.audio_frames_sent:
            for _ in range(_TICKS_PER_FRAME):
                self.clock.advance(_SCALED_TICK_MS)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

        self.audio_frames_sent += 1
        self._frame_index += 1
        return {
            "type": "websocket.receive",
            "text": json.dumps({
                "source": "mic",
                "speaker": "you",
                "encoding": "pcm_int16",
                "sample_rate": 16_000,
                "audio": self._audio,
            }),
        }


class _FakeFastAPI:
    def __init__(self, **_kwargs: object) -> None:
        self.routes: dict[tuple[str, str], object] = {}

    def _register(self, method: str, path: str):
        def decorator(function):
            self.routes[(method, path)] = function
            return function
        return decorator

    def get(self, path: str, **_kwargs: object):
        return self._register("GET", path)

    def websocket(self, path: str):
        return self._register("WEBSOCKET", path)


class _FakeResponse:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None


def _load_service_module():
    """Load the real service handler with import-only FastAPI stubs."""
    sys.modules.pop("asr_lab.service", None)
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FakeFastAPI
    fastapi.Query = lambda default=None: default
    fastapi.Header = lambda default=None: default
    fastapi.WebSocket = object
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _FakeResponse
    responses.FileResponse = _FakeResponse
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


class SustainedThroughputReproducerTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self) -> Settings:
        return Settings.from_env({
            "ASR_BACKEND": "faster_whisper",
            "ASR_API_TOKEN": _TOKEN,
            "ASR_MAX_CHUNK_SECONDS": "5",
            "QWEN_STREAMING_ENABLED": "1",
            "QWEN_STREAMING_RUNTIME_AVAILABLE": "1",
            "QWEN_MODEL_CHUNK_MS": "100",
            "QWEN_MAX_PENDING_JOBS": "24",
            "QWEN_MAX_BACKLOG_CHUNKS": "4",
            "QWEN_MAX_STREAM_SECONDS": "60",
        })

    async def _run(self, *, decode_ms: float, frame_count: int) -> dict[str, object]:
        clock = _ScaledClock()
        worker = _ControlledWorker(decode_ms=decode_ms, clock=clock)
        runtime = QwenStreamingRuntime(
            worker=worker,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="diagnostic-fixed-revision",
            gpu_memory_utilization=0.65,
            max_active_sessions=6,
            max_stream_seconds=60.0,
            session_idle_ttl_seconds=120.0,
            max_context_chars=512,
            default_chunk_ms=100,
            default_language="auto",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            max_pending_jobs=24,
            max_backlog_chunks=4,
        )
        await runtime.start()
        service = _load_service_module()
        service.settings = self._settings()
        service.qwen_runtime = runtime
        self.assertIs(service.QwenServiceLifecycleRegistry, QwenServiceLifecycleRegistry)
        websocket = _FakeWebSocket(frame_count=frame_count, clock=clock)
        try:
            await service._websocket_qwen(websocket, f"r1-{decode_ms}")
            return {
                "audio_frames_sent": websocket.audio_frames_sent,
                "decode_pushes_started": worker.push_count,
                "init_count": worker.init_count,
                "scheduler_overruns": runtime.scheduler.snapshot()["qwen_scheduler_overrun_total"],
                "errors": [item for item in websocket.sent if item.get("event") == "error"],
                "clock_ms": clock.now_ms,
            }
        finally:
            await runtime.close()

    async def test_slow_145ms_reproduces_transition_capacity_then_stream_terminal(self) -> None:
        runs = [await self._run(decode_ms=145.0, frame_count=240) for _ in range(3)]
        summaries = []
        for result in runs:
            errors = result["errors"]
            codes = [item.get("code") for item in errors]
            self.assertEqual(codes, ["transition_capacity_exceeded", "stream_terminal"])
            capacity_error = errors[0]
            observability = capacity_error["qwen_epoch_observability"]
            transition = observability["last_transition"]
            pcm = observability["current_epoch"]["pcm"]
            self.assertEqual(transition["failure_stage"], "transition_capacity")
            self.assertEqual(transition["failure_reason"], "transition_capacity_exceeded")
            self.assertEqual(transition["last_completed_stage"], "REPLAY_DRAINED")
            self.assertIsNotNone(transition["replay_completed_at_monotonic"])
            self.assertEqual(transition["stage"], "CATCHUP_DRAINING")
            self.assertIsNone(transition["catchup_completed_at_monotonic"])
            self.assertIsNone(transition["handoff_completed_at_monotonic"])
            self.assertEqual(pcm["max_transition_queued_samples"], _TRANSITION_SAMPLES)
            self.assertEqual(pcm["transition_queued_samples"], _TRANSITION_SAMPLES)
            self.assertEqual(pcm["explicit_source_rejected_samples"], _FRAME_SAMPLES)
            self.assertGreater(pcm["downstream_admission_rejected_samples"], 0)
            self.assertGreater(result["scheduler_overruns"], 0)
            self.assertGreater(result["decode_pushes_started"], 0)
            self.assertEqual(
                pcm["source_head_cursor"],
                pcm["unique_primary_admitted_cursor"]
                + pcm["transition_queued_samples"]
                + pcm["explicit_source_rejected_samples"],
            )
            self.assertEqual(
                pcm["retained_source_samples"],
                pcm["replay_retained_samples"] + pcm["transition_queued_samples"],
            )
            summaries.append((
                result["audio_frames_sent"],
                result["decode_pushes_started"],
                result["scheduler_overruns"],
                transition["last_completed_stage"],
                pcm["transition_queued_samples"],
                pcm["explicit_source_rejected_samples"],
            ))
        self.assertEqual(summaries, [summaries[0]] * len(summaries))

    async def test_fast_92ms_control_does_not_roll_over_or_terminalize(self) -> None:
        runs = [await self._run(decode_ms=92.0, frame_count=240) for _ in range(3)]
        for result in runs:
            codes = [item.get("code") for item in result["errors"]]
            self.assertNotIn("transition_capacity_exceeded", codes)
            self.assertNotIn("stream_terminal", codes)
            self.assertEqual(result["scheduler_overruns"], 0)
            self.assertEqual(result["init_count"], 1)
            self.assertEqual(result["audio_frames_sent"], 240)


if __name__ == "__main__":
    unittest.main()
