"""FastAPI ASR WebSocket service. Audio and credentials are never logged."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
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
from .config import QWEN_LANGUAGE_CODES, Settings
from .protocol import AudioChunk, BenchmarkPing, FlushRequest, ProtocolError, StreamStart, parse_message
from .qwen_process import QwenWorkerProcess
from .qwen_streaming import QwenStreamingRuntime, StreamingError, validate_candidate_event
from .security import token_matches
from .telemetry import collect_telemetry, gpu_compute_state

settings = Settings.from_env()
logger = logging.getLogger("asr_lab.service")
broker: InferenceBroker | None = None
qwen_runtime: QwenStreamingRuntime | None = None
config_valid = settings.error is None
connection_slots = asyncio.Semaphore(settings.max_connections)

_SAFE_SCHEDULER_REASONS = {
    "per_stream_backlog_limit",
    "global_pending_job_limit",
    "global_pending_job_limit_at_eos",
}
_SAFE_SCHEDULER_LIMIT_KINDS = {"backlog_ms", "pending_jobs"}
_TERMINAL_SCHEDULER_NUMERIC_FIELDS = {
    "scheduler_wait_p50_ms", "scheduler_wait_p95_ms", "scheduler_wait_max_ms",
    "scheduler_wait_sample_count", "decode_wall_p50_ms", "decode_wall_p95_ms",
    "decode_wall_max_ms", "decode_wall_sample_count", "pending_jobs",
    "stream_pending_jobs", "active_stream_count", "backlog_audio_ms",
    "stream_lag_ms", "max_backlog_audio_ms", "max_stream_lag_ms",
    "accepted_audio_total_ms", "dispatched_audio_total_ms", "overrun_limit_value",
    "effective_max_backlog_ms", "qwen_max_backlog_chunks", "model_chunk_ms",
    "scheduler_metric_history_limit",
}


def _safe_scheduler_diagnostics(details: dict[str, Any]) -> dict[str, Any]:
    """Copy only bounded scheduler numbers and known enum labels to the client."""
    result: dict[str, Any] = {}
    for key in ("accepted_audio_ms", "rejected_audio_ms", "backlog_limit_ms", "pending_jobs", "pending_job_limit"):
        value = details.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            result[key] = value
    reason = details.get("reason")
    if reason in _SAFE_SCHEDULER_REASONS:
        result["reason"] = reason

    terminal = details.get("terminal_metrics")
    if isinstance(terminal, dict):
        safe_terminal: dict[str, Any] = {}
        for key in _TERMINAL_SCHEDULER_NUMERIC_FIELDS:
            value = terminal.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                safe_terminal[key] = value
        terminal_reason = terminal.get("overrun_reason")
        if terminal_reason in _SAFE_SCHEDULER_REASONS:
            safe_terminal["overrun_reason"] = terminal_reason
        limit_kind = terminal.get("overrun_limit_kind")
        if limit_kind in _SAFE_SCHEDULER_LIMIT_KINDS:
            safe_terminal["overrun_limit_kind"] = limit_kind
        if safe_terminal:
            result["terminal_metrics"] = safe_terminal
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global broker, qwen_runtime
    loader: asyncio.Task[None] | None = None
    if config_valid and not settings.test_no_model and settings.backend and settings.model_id:
        if settings.qwen_streaming_enabled:
            worker = QwenWorkerProcess(
                python=settings.qwen_worker_python,
                script=Path(__file__).with_name("qwen_worker.py"),
            )
            qwen_runtime = QwenStreamingRuntime(
                worker=worker,
                model_id=settings.active_model_id or "",
                model_revision=settings.active_model_revision or "",
                gpu_memory_utilization=settings.qwen_gpu_memory_utilization,
                max_active_sessions=settings.qwen_max_active_sessions,
                max_stream_seconds=settings.qwen_max_stream_seconds,
                session_idle_ttl_seconds=settings.qwen_session_idle_ttl_seconds,
                max_context_chars=settings.qwen_max_context_chars,
                default_chunk_ms=settings.qwen_model_chunk_ms,
                default_language=settings.qwen_language,
                unfixed_chunk_num=settings.qwen_unfixed_chunk_num,
                unfixed_token_num=settings.qwen_unfixed_token_num,
                max_pending_jobs=settings.qwen_max_pending_jobs,
                max_backlog_chunks=settings.qwen_max_backlog_chunks,
            )
            loader = asyncio.create_task(_load_qwen(qwen_runtime), name="qwen-model-loader")
        else:
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
        if qwen_runtime is not None:
            await qwen_runtime.close()
            qwen_runtime = None


async def _load_backend(target: InferenceBroker) -> None:
    try:
        await target.start()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("ASR startup failed exception_type=%s", type(exc).__name__)


async def _load_qwen(target: QwenStreamingRuntime) -> None:
    try:
        await target.start()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Qwen streaming startup failed exception_type=%s", type(exc).__name__)


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
    active_runtime = qwen_runtime if settings.qwen_streaming_enabled else broker
    qwen_scheduler = qwen_runtime.scheduler.snapshot() if qwen_runtime is not None and settings.qwen_streaming_enabled else None
    snapshot = await asyncio.to_thread(collect_telemetry, active_runtime, settings, qwen_scheduler)
    snapshot["transcript_mode"] = settings.transcript_mode
    snapshot["active_streams"] = qwen_runtime.active_sessions if qwen_runtime is not None else None
    snapshot["max_active_streams"] = settings.qwen_max_active_sessions if settings.qwen_streaming_enabled else None
    if settings.qwen_streaming_enabled:
        snapshot["workers"] = 1
        snapshot["runtime_provenance"] = qwen_runtime.runtime_provenance if qwen_runtime is not None else None
        snapshot["qwen_scheduler"] = qwen_scheduler
        snapshot["active_streams"] = qwen_scheduler["active_stream_count"] if qwen_scheduler else 0
        snapshot["pending_decode_count"] = qwen_scheduler["pending_decode_count"] if qwen_scheduler else 0
        snapshot["active_decode_count"] = qwen_scheduler["active_decode_count"] if qwen_scheduler else 0
        snapshot["qwen_max_pending_jobs"] = settings.qwen_max_pending_jobs
        snapshot["qwen_max_backlog_chunks"] = settings.qwen_max_backlog_chunks
    return JSONResponse(snapshot, headers={"Cache-Control": "no-store"})


def _worker_metrics() -> list[dict[str, Any]]:
    return [] if broker is None else broker.worker_metrics


@app.get("/asr/health")
async def health() -> dict[str, Any]:
    active_runtime = qwen_runtime if settings.qwen_streaming_enabled else broker
    compute = gpu_compute_state(active_runtime, settings)
    model_loaded = bool(active_runtime and active_runtime.ready)
    return {
        "status": "ok",
        "backend": settings.active_backend,
        "production_backend": settings.backend,
        "model_id": settings.active_model_id,
        "model_revision": settings.active_model_revision,
        "model_loaded": model_loaded,
        "transcript_mode": settings.transcript_mode,
        "streaming_class": "accumulated-audio-pseudostreaming" if settings.qwen_streaming_enabled else None,
        "cuda": compute["cuda"],
        "gpu_compute": compute,
        "workers": 1 if settings.qwen_streaming_enabled else settings.workers,
        "worker_metrics": active_runtime.worker_metrics if active_runtime is not None else _worker_metrics(),
        "active_streams": qwen_runtime.active_sessions if qwen_runtime is not None else None,
        "max_active_streams": settings.qwen_max_active_sessions if settings.qwen_streaming_enabled else None,
        "runtime_provenance": qwen_runtime.runtime_provenance if settings.qwen_streaming_enabled and qwen_runtime is not None else None,
    }


@app.get("/asr/readiness")
async def readiness() -> JSONResponse:
    active_runtime = qwen_runtime if settings.qwen_streaming_enabled else broker
    ready = config_valid and not settings.test_no_model and active_runtime is not None and active_runtime.ready
    return JSONResponse(
        {"ready": bool(ready), "backend": settings.active_backend, "production_backend": settings.backend, "model_loaded": bool(active_runtime and active_runtime.ready)},
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
    active_runtime = qwen_runtime if settings.qwen_streaming_enabled else broker
    if settings.test_no_model or active_runtime is None or not active_runtime.ready:
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
    if settings.qwen_streaming_enabled:
        try:
            await _websocket_qwen(websocket, connection_id)
        finally:
            if qwen_runtime is not None:
                await qwen_runtime.close_connection(connection_id)
            connection_slots.release()
        return
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
                    max_context_chars=settings.qwen_max_context_chars,
                )
            except ProtocolError as exc:
                await send_json({"event": "error", "code": exc.code, "message": exc.message})
                continue
            now = time.perf_counter()
            if isinstance(message, BenchmarkPing):
                await send_json({"event": "benchmark_pong", "request_id": message.request_id})
                continue
            if isinstance(message, StreamStart):
                await send_json({
                    "event": "error",
                    "code": "streaming_backend_required",
                    "message": "This ASR backend accepts complete segments only.",
                })
                continue
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


async def _websocket_qwen(websocket: WebSocket, connection_id: str) -> None:
    runtime = qwen_runtime
    if runtime is None or not runtime.ready:
        await websocket.close(code=1013, reason="model_not_ready")
        return
    send_lock = asyncio.Lock()
    opened_sources: set[str] = set()
    terminal_sources: set[str] = set()
    terminal_error_codes = {
        "stream_duration_limit", "stream_not_started", "stream_worker_failed",
        "stream_worker_timeout", "stream_scheduler_overrun", "stream_result_queue_full",
        "invalid_worker_response", "stream_fenced",
    }
    outbound: asyncio.Queue[tuple[dict[str, Any], asyncio.Future[None] | None] | None] = asyncio.Queue(maxsize=128)

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def enqueue_result(payload: dict[str, Any]) -> None:
        try:
            outbound.put_nowait((payload, None))
        except asyncio.QueueFull as exc:
            raise StreamingError("stream_result_queue_full") from exc

    async def enqueue_final(payload: dict[str, Any]) -> None:
        delivered = asyncio.get_running_loop().create_future()
        await outbound.put((payload, delivered))
        await delivered

    async def result_pump() -> None:
        while True:
            item = await outbound.get()
            if item is None:
                return
            payload, delivered = item
            try:
                event = payload.get("event")
                if event in {"partial_candidate", "final_candidate"}:
                    expected = "final_candidate" if event == "final_candidate" else "partial_candidate"
                    payload = validate_candidate_event(payload, expected)
                await send_json(payload)
                if delivered is not None and not delivered.done():
                    delivered.set_result(None)
            except Exception as exc:
                if delivered is not None and not delivered.done():
                    delivered.set_exception(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise

    pump_task = asyncio.create_task(result_pump(), name=f"qwen-result-pump-{connection_id}")

    async def open_default_session(source: str) -> None:
        payload = await runtime.open_session(
            connection_id=connection_id, source=source, event_sink=enqueue_result,
        )
        opened_sources.add(source)
        await send_json(payload)

    await websocket.accept()
    try:
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                return
            raw: str | bytes | None = frame.get("text")
            if raw is None:
                raw = frame.get("bytes", b"")
            try:
                message = parse_message(
                    raw or b"",
                    max_message_bytes=settings.max_message_bytes,
                    max_chunk_seconds=settings.max_chunk_seconds,
                    max_context_chars=settings.qwen_max_context_chars,
                )
            except ProtocolError as exc:
                await send_json({"event": "error", "code": exc.code, "message": exc.message})
                continue

            if isinstance(message, BenchmarkPing):
                await send_json({"event": "benchmark_pong", "request_id": message.request_id})
                continue

            if isinstance(message, StreamStart):
                if message.language not in QWEN_LANGUAGE_CODES:
                    await send_json({
                        "event": "error",
                        "code": "invalid_language",
                        "request_id": message.request_id,
                        "message": "Qwen language must be auto or a supported language code.",
                    })
                    continue
                try:
                    result = await runtime.open_session(
                        connection_id=connection_id,
                        source=message.source,
                        language=message.language,
                        context=message.context,
                        model_chunk_ms=message.model_chunk_ms,
                        request_id=message.request_id,
                        event_sink=enqueue_result,
                    )
                except StreamingError as exc:
                    await send_json({
                        "event": "error",
                        "code": exc.code,
                        "request_id": message.request_id,
                        "message": "Streaming session could not be started.",
                    })
                    continue
                terminal_sources.discard(message.source)
                opened_sources.add(message.source)
                await send_json(result)
                continue

            if isinstance(message, FlushRequest):
                if message.source not in opened_sources:
                    await send_json({
                        "event": "flush_complete",
                        "request_id": message.request_id,
                    })
                    continue
                try:
                    result = await runtime.finish(
                        connection_id=connection_id,
                        source=message.source,
                        server_eos_at=time.perf_counter(),
                        request_id=message.request_id,
                    )
                    await enqueue_final(validate_candidate_event(result, "final_candidate"))
                    opened_sources.discard(message.source)
                except StreamingError as exc:
                    opened_sources.discard(message.source)
                    if exc.code in terminal_error_codes:
                        terminal_sources.add(message.source)
                    error_payload: dict[str, Any] = {
                        "event": "error", "code": exc.code,
                        "message": "Streaming finalization failed.",
                    }
                    if exc.code == "stream_scheduler_overrun":
                        error_payload["DECODE_OVERRUN"] = True
                        scheduler_diagnostics = _safe_scheduler_diagnostics(exc.details)
                        if scheduler_diagnostics:
                            error_payload["scheduler"] = scheduler_diagnostics
                    await send_json(error_payload)
                if message.request_id is not None:
                    await send_json({
                        "event": "flush_complete",
                        "request_id": message.request_id,
                    })
                continue

            assert isinstance(message, AudioChunk)
            if message.source in terminal_sources:
                await send_json({
                    "event": "error",
                    "code": "stream_terminal",
                    "message": "This stream ended; send an explicit stream_start before sending more audio.",
                })
                continue
            if message.source not in opened_sources:
                try:
                    await open_default_session(message.source)
                except StreamingError as exc:
                    await send_json({"event": "error", "code": exc.code, "message": "Streaming session capacity is unavailable."})
                    continue
            try:
                result = await runtime.push_audio(
                    connection_id=connection_id,
                    source=message.source,
                    pcm16le=message.pcm16le,
                )
            except StreamingError as exc:
                if exc.code in terminal_error_codes:
                    opened_sources.discard(message.source)
                    terminal_sources.add(message.source)
                error_payload = {
                    "event": "error", "code": exc.code,
                    "DECODE_OVERRUN": exc.code == "stream_scheduler_overrun",
                    "message": "Streaming audio could not be processed.",
                }
                if exc.code == "stream_scheduler_overrun":
                    scheduler_diagnostics = _safe_scheduler_diagnostics(exc.details)
                    if scheduler_diagnostics:
                        error_payload["scheduler"] = scheduler_diagnostics
                await send_json(error_payload)
    finally:
        await runtime.close_connection(connection_id)
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)


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
            server_eos_at=last_voice_at,
        )
        if result.pop("_discarded_historical_output", False):
            return
        internal_timing = result.pop("_internal_timing", {})
        if client_request_id is not None:
            result["client_request_id"] = client_request_id
        transcript_ready_at = time.perf_counter()
        model_finished_at = internal_timing.get("model_finished_at")
        if not isinstance(model_finished_at, (int, float)):
            model_finished_at = transcript_ready_at
        postprocess_ms = max(0.0, transcript_ready_at - float(model_finished_at)) * 1000
        server_eos_ms = max(0.0, transcript_ready_at - last_voice_at) * 1000
        result["SERVER_POSTPROCESS_MS"] = round(postprocess_ms, 2)
        result["SERVER_EOS_TO_TRANSCRIPT_MS"] = round(server_eos_ms, 2)
        # Backwards-compatible aliases. SEGMENT_WAIT_MS remains the legacy
        # last-voice-to-close decision interval, before broker submission.
        result["SERVER_AUDIO_END_TO_TRANSCRIPT_MS"] = round(server_eos_ms, 2)
        result["SERVER_RECEIVE_TO_TRANSCRIPT_MS"] = round(
            max(0.0, transcript_ready_at - segment_started_at) * 1000, 2
        )
        await send_json(result)
    except RuntimeError as exc:
        code = str(exc) if str(exc) in {"backend_not_ready", "inference_queue_full", "inference_failed"} else "inference_failed"
        with contextlib.suppress(Exception):
            await send_json({"event": "error", "code": code, "message": "Transcription is temporarily unavailable."})
    except Exception as exc:
        logger.error("ASR response delivery failed exception_type=%s", type(exc).__name__)
