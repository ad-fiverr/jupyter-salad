"""FastAPI ASR WebSocket service. Audio and credentials are never logged."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Header, Query, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import Response

from .broker import InferenceBroker
from .buffering import AudioBuffer, INACTIVITY_FLUSH_SECONDS
from .config import Settings
from .protocol import AudioChunk, FlushRequest, ProtocolError, parse_message
from .security import token_matches
from .telemetry import collect_telemetry

settings = Settings.from_env()
logger = logging.getLogger("asr_lab.service")
broker: InferenceBroker | None = None
config_valid = settings.error is None
connection_slots = asyncio.Semaphore(settings.max_connections)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global broker
    loader: asyncio.Task[None] | None = None
    if config_valid and not settings.test_no_model and settings.backend and settings.model_id:
        broker = InferenceBroker(
            backend_name=settings.backend,
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            workers=settings.workers,
            queue_size=settings.max_queue_size,
        )
        loader = asyncio.create_task(_load_backend(broker), name="asr-model-loader")
    try:
        yield
    finally:
        if loader is not None and not loader.done():
            loader.cancel()
            await asyncio.gather(loader, return_exceptions=True)
        if broker is not None:
            await broker.close()
            broker = None


async def _load_backend(target: InferenceBroker) -> None:
    try:
        await target.start()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("ASR startup failed exception_type=%s", type(exc).__name__)


app = FastAPI(title="Salad ASR Lab", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
BENCHMARK_WEB_ROOT = Path(__file__).with_name("benchmark_web")
BENCHMARK_ASSETS = {"app.mjs", "audio-worklet.mjs", "core.mjs", "style.css"}
BENCHMARK_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self' ws: wss:; worker-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "Permissions-Policy": "microphone=(self)",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


@app.get("/asr/benchmark", include_in_schema=False)
async def benchmark_page() -> FileResponse:
    return FileResponse(BENCHMARK_WEB_ROOT / "index.html", headers=BENCHMARK_HEADERS)


@app.get("/asr/benchmark/{asset_name}", include_in_schema=False, response_model=None)
async def benchmark_asset(asset_name: str) -> Response:
    if asset_name not in BENCHMARK_ASSETS:
        return JSONResponse({"detail": "Not found"}, status_code=404, headers=BENCHMARK_HEADERS)
    return FileResponse(BENCHMARK_WEB_ROOT / asset_name, headers=BENCHMARK_HEADERS)


@app.get("/asr/telemetry", include_in_schema=False)
async def telemetry(authorization: str | None = Header(default=None)) -> JSONResponse:
    scheme, _, bearer = (authorization or "").partition(" ")
    if scheme.casefold() != "bearer" or not settings.api_token or not token_matches(bearer, settings.api_token):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401, headers={"Cache-Control": "no-store"})
    snapshot = await asyncio.to_thread(collect_telemetry, broker, settings)
    return JSONResponse(snapshot, headers={"Cache-Control": "no-store"})


def _worker_metrics() -> list[dict[str, Any]]:
    return [] if broker is None else broker.worker_metrics


@app.get("/asr/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": settings.backend,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "model_loaded": bool(broker and broker.ready),
        "cuda": any(worker.get("vram_after_warmup", {}).get("available") for worker in _worker_metrics()),
        "workers": settings.workers,
        "worker_metrics": _worker_metrics(),
    }


@app.get("/asr/readiness")
async def readiness() -> JSONResponse:
    ready = config_valid and not settings.test_no_model and broker is not None and broker.ready
    return JSONResponse(
        {"ready": bool(ready), "backend": settings.backend, "model_loaded": bool(broker and broker.ready)},
        status_code=200 if ready else 503,
    )


@app.websocket("/asr/ws")
async def websocket_asr(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
    if not settings.api_token or not token_matches(token or "", settings.api_token):
        await websocket.close(code=4401, reason="unauthorized")
        return
    if not config_valid:
        await websocket.close(code=1013, reason="service_not_configured")
        return
    if settings.test_no_model or broker is None or not broker.ready:
        if settings.test_no_model and settings.api_token:
            await websocket.accept()
            await websocket.send_json({"event": "error", "code": "model_not_ready", "message": "Transcription model is unavailable in CI smoke mode."})
            await websocket.close(code=1013, reason="model_not_ready")
            return
        await websocket.close(code=1013, reason="model_not_ready")
        return
    try:
        await asyncio.wait_for(connection_slots.acquire(), timeout=0.1)
    except TimeoutError:
        await websocket.close(code=1013, reason="connection_limit")
        return

    connection_id = uuid.uuid4().hex
    connection_started = time.perf_counter()
    buffers: dict[str, AudioBuffer] = {}
    inactivity_timers: dict[str, asyncio.Task[None]] = {}
    pending: set[asyncio.Task[None]] = set()
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def queue_segment(source: str, audio: AudioBuffer, now: float) -> None:
        if not audio.can_flush:
            return
        if len(pending) >= settings.max_pending_per_connection:
            await send_json({"event": "error", "code": "pending_limit", "message": "This connection already has the maximum number of pending transcriptions."})
            return
        task = asyncio.create_task(
            _flush(websocket, connection_id, source, audio, now, connection_started, send_json),
            name=f"asr-job-{connection_id}-{source}",
        )
        pending.add(task)
        task.add_done_callback(pending.discard)

    def cancel_inactivity_timer(source: str) -> None:
        timer = inactivity_timers.pop(source, None)
        if timer is not None and not timer.done():
            timer.cancel()

    async def inactivity_flush(source: str, expected_audio: AudioBuffer) -> None:
        try:
            await asyncio.sleep(INACTIVITY_FLUSH_SECONDS)
            audio = buffers.get(source)
            if audio is expected_audio and time.perf_counter() - audio.last_audio_at >= INACTIVITY_FLUSH_SECONDS:
                await queue_segment(source, audio, time.perf_counter())
                buffers.pop(source, None)
        except asyncio.CancelledError:
            raise
        finally:
            current = asyncio.current_task()
            if inactivity_timers.get(source) is current:
                inactivity_timers.pop(source, None)

    def reset_inactivity_timer(source: str, audio: AudioBuffer) -> None:
        cancel_inactivity_timer(source)
        if audio.pcm:
            inactivity_timers[source] = asyncio.create_task(
                inactivity_flush(source, audio),
                name=f"asr-inactivity-{connection_id}-{source}",
            )

    try:
        await websocket.accept()
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                break
            raw: str | bytes | None = frame.get("text")
            if raw is None:
                raw = frame.get("bytes", b"")
            try:
                message = parse_message(
                    raw or b"",
                    max_message_bytes=settings.max_message_bytes,
                    max_chunk_seconds=settings.max_chunk_seconds,
                )
            except ProtocolError as exc:
                await send_json({"event": "error", "code": exc.code, "message": exc.message})
                continue
            now = time.perf_counter()
            if isinstance(message, FlushRequest):
                audio = buffers.pop(message.source, None)
                cancel_inactivity_timer(message.source)
                if audio is not None:
                    await queue_segment(message.source, audio, now)
                if message.request_id is not None:
                    # The benchmark uses this barrier to know that every
                    # transcript produced by one fixture has been delivered.
                    completed_jobs = tuple(pending)
                    if completed_jobs:
                        await asyncio.gather(*completed_jobs, return_exceptions=True)
                    await send_json({
                        "event": "flush_complete",
                        "request_id": message.request_id,
                    })
                continue
            assert isinstance(message, AudioChunk)
            audio = buffers.get(message.source)
            if audio is None:
                audio = AudioBuffer(message.source, message.speaker, now, now)
                buffers[message.source] = audio
            if audio.duration_seconds + message.duration_seconds > settings.max_buffer_seconds:
                cancel_inactivity_timer(message.source)
                await send_json({"event": "error", "code": "buffer_too_large", "message": "Audio buffer exceeds the configured limit."})
                buffers.pop(message.source, None)
                continue
            should_flush = audio.append(message, now)
            if should_flush:
                cancel_inactivity_timer(message.source)
                await queue_segment(message.source, audio, now)
                buffers.pop(message.source, None)
            else:
                reset_inactivity_timer(message.source, audio)
    finally:
        timers = tuple(inactivity_timers.values())
        inactivity_timers.clear()
        for timer in timers:
            timer.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        buffers.clear()
        connection_slots.release()


async def _flush(
    websocket: WebSocket, connection_id: str, source: str, audio: AudioBuffer,
    now: float, connection_started: float,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    if not audio.can_flush:
        return
    segment_started_at = audio.started_at
    last_voice_at = audio.last_voice_at
    client_request_id = getattr(audio, "request_id", None)
    speaker = audio.speaker
    pcm = audio.take()
    try:
        if broker is None:
            raise RuntimeError("backend_not_ready")
        result: dict[str, Any] = await broker.transcribe(
            connection_id=connection_id,
            source=source,
            speaker=speaker,
            pcm16le=pcm,
            segment_start_s=max(0.0, segment_started_at - connection_started),
            segment_end_s=max(0.0, last_voice_at - connection_started),
            segment_wait_ms=max(0.0, (now - last_voice_at) * 1000),
        )
        if result.pop("_discarded_historical_output", False):
            return
        transcript_ready_at = time.perf_counter()
        result["SERVER_AUDIO_END_TO_TRANSCRIPT_MS"] = round(
            max(0.0, transcript_ready_at - last_voice_at) * 1000, 2
        )
        result["SERVER_RECEIVE_TO_TRANSCRIPT_MS"] = round(
            max(0.0, transcript_ready_at - segment_started_at) * 1000, 2
        )
        if client_request_id is not None:
            result["client_request_id"] = client_request_id
        await send_json(result)
    except RuntimeError as exc:
        code = str(exc) if str(exc) in {"backend_not_ready", "inference_queue_full", "inference_failed"} else "inference_failed"
        with contextlib.suppress(Exception):
            await send_json({"event": "error", "code": code, "message": "Transcription is temporarily unavailable."})
    except Exception as exc:
        logger.error("ASR response delivery failed exception_type=%s", type(exc).__name__)
