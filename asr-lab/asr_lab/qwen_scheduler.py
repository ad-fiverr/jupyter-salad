"""Single-engine, bounded round-robin admission for Qwen decode work.

The scheduler owns audio windows and request fencing, not Qwen model state.
It deliberately runs one worker RPC at a time; asyncio/WebSocket concurrency
must not be confused with model or GPU concurrency.
"""
from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class RequestKey:
    connection_id: str
    stream_id: str
    scheduler_revision: int


LocalStaleRejectObserver = Callable[[str, str], None]


@dataclass
class DecodeJob:
    kind: str
    pcm16le: bytes
    cursor_end_samples: int
    ready_at: float
    future: asyncio.Future["DispatchResult"] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    local_stale_reject_observer: LocalStaleRejectObserver | None = None


@dataclass(frozen=True)
class DispatchResult:
    key: RequestKey
    reply: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class StreamQueue:
    connection_id: str
    stream_id: str
    chunk_ms: int
    pending: deque[DecodeJob] = field(default_factory=deque)
    tail: bytearray = field(default_factory=bytearray)
    accepted_samples: int = 0
    first_audio_at: float | None = None
    decoded_samples: int = 0
    next_revision: int = 0
    active_key: RequestKey | None = None
    fenced: bool = False
    accepting: bool = True
    waits_ms: list[float] = field(default_factory=list)
    decode_wall_ms: list[float] = field(default_factory=list)
    decode_steps: int = 0
    accepted_total_samples: int = 0
    dispatched_total_samples: int = 0
    max_backlog_samples: int = 0
    max_stream_lag_samples: int = 0
    max_pending_jobs: int = 0
    max_active_streams: int = 0
    local_stale_reject_observer: LocalStaleRejectObserver | None = None

    @property
    def identity(self) -> tuple[str, str]:
        return self.connection_id, self.stream_id

    @property
    def queued_samples(self) -> int:
        return sum(len(job.pcm16le) // 2 for job in self.pending) + len(self.tail) // 2


class SchedulerError(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


ExecuteJob = Callable[[RequestKey, DecodeJob], Awaitable[dict[str, Any]]]
ResultHandler = Callable[[RequestKey, DecodeJob, dict[str, Any], dict[str, Any]], Awaitable[None]]
FaultHandler = Callable[[str, str, str, dict[str, Any]], Awaitable[None]]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.9999)))
    return round(ordered[index], 3)


class QwenDecodeScheduler:
    """Bounded per-stream PCM queues serviced in global round-robin order."""

    def __init__(
        self,
        *,
        execute: ExecuteJob,
        on_result: ResultHandler,
        on_fault: FaultHandler,
        max_pending_jobs: int = 24,
        max_backlog_chunks: int = 4,
        max_active_streams: int = 6,
        history_limit: int = 256,
    ) -> None:
        if min(max_pending_jobs, max_backlog_chunks, max_active_streams, history_limit) < 1:
            raise ValueError("scheduler_limits_must_be_positive")
        self.execute = execute
        self.on_result = on_result
        self.on_fault = on_fault
        self.max_pending_jobs = max_pending_jobs
        self.max_backlog_chunks = max_backlog_chunks
        self.max_active_streams = max_active_streams
        self.history_limit = history_limit
        self.streams: dict[tuple[str, str], StreamQueue] = {}
        self.ready: deque[tuple[str, str]] = deque()
        self._ready_set: set[tuple[str, str]] = set()
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._active_job: tuple[StreamQueue, RequestKey, DecodeJob] | None = None
        self._waits_ms: deque[float] = deque(maxlen=history_limit)
        self._decode_wall_ms: deque[float] = deque(maxlen=history_limit)
        self._decode_steps = 0
        self._overrun_total = 0
        self._decode_budget_overrun_total = 0
        self._decode_slo_violation_total = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._dispatch_loop(), name="qwen-fair-decode-scheduler")

    async def register(
        self,
        connection_id: str,
        stream_id: str,
        chunk_ms: int,
        local_stale_reject_observer: LocalStaleRejectObserver | None = None,
    ) -> None:
        identity = (connection_id, stream_id)
        async with self._condition:
            if identity in self.streams:
                raise SchedulerError("stream_already_registered")
            active = sum(not stream.fenced for stream in self.streams.values())
            if active >= self.max_active_streams:
                raise SchedulerError("stream_capacity_exceeded")
            self.streams[identity] = StreamQueue(
                connection_id,
                stream_id,
                chunk_ms,
                local_stale_reject_observer=local_stale_reject_observer,
            )
            self._observe_global_high_water_locked()

    async def append_pcm(self, connection_id: str, stream_id: str, pcm16le: bytes, received_at: float | None = None) -> dict[str, Any]:
        if not pcm16le or len(pcm16le) % 2:
            raise SchedulerError("invalid_audio")
        identity = (connection_id, stream_id)
        fault: tuple[str, dict[str, Any]] | None = None
        async with self._condition:
            stream = self.streams.get(identity)
            if stream is None or stream.fenced or not stream.accepting:
                raise SchedulerError("stream_not_started")
            incoming_samples = len(pcm16le) // 2
            window_samples = stream.chunk_ms * 16
            combined = bytes(stream.tail) + pcm16le
            window_bytes = window_samples * 2
            window_count = len(combined) // window_bytes
            residual = combined[window_count * window_bytes:]
            first_audio = stream.first_audio_at is None
            scheduled_job_count = window_count + int(first_audio and window_count == 0)
            backlog_samples = stream.queued_samples + incoming_samples
            per_stream_limit = window_samples * self.max_backlog_chunks
            if backlog_samples > per_stream_limit:
                fault = ("stream_scheduler_overrun", {
                    "reason": "per_stream_backlog_limit",
                    "accepted_audio_ms": stream.queued_samples * 1000.0 / 16_000,
                    "rejected_audio_ms": incoming_samples * 1000.0 / 16_000,
                    "backlog_limit_ms": per_stream_limit * 1000.0 / 16_000,
                })
            elif self.pending_jobs + scheduled_job_count > self.max_pending_jobs:
                fault = ("stream_scheduler_overrun", {
                    "reason": "global_pending_job_limit",
                    "accepted_audio_ms": stream.queued_samples * 1000.0 / 16_000,
                    "rejected_audio_ms": incoming_samples * 1000.0 / 16_000,
                    "pending_jobs": self.pending_jobs,
                    "pending_job_limit": self.max_pending_jobs,
                })
            if fault is not None:
                fault[1]["terminal_metrics"] = self._terminal_metrics_locked(stream, fault[1])
                self._fence_locked(stream)
                self._overrun_total += 1
            else:
                cursor = stream.accepted_samples - (len(stream.tail) // 2)
                now = time.perf_counter() if received_at is None else received_at
                if stream.first_audio_at is None:
                    stream.first_audio_at = now
                for index in range(window_count):
                    pcm = combined[index * window_bytes:(index + 1) * window_bytes]
                    cursor += window_samples
                    stream.pending.append(DecodeJob(
                        "push", pcm, cursor, now,
                        local_stale_reject_observer=stream.local_stale_reject_observer,
                    ))
                if first_audio and window_count == 0:
                    # Start the ASR stream from speech arrival, even when this
                    # first transport payload is shorter than a model window.
                    # This one-time unpadded priming push is consumed exactly
                    # once; later pushes still follow configured model windows.
                    priming_samples = len(combined) // 2
                    stream.pending.append(DecodeJob(
                        "push", combined, stream.accepted_samples + priming_samples, now,
                        local_stale_reject_observer=stream.local_stale_reject_observer,
                    ))
                    residual = b""
                stream.tail = bytearray(residual)
                stream.accepted_samples += incoming_samples
                stream.accepted_total_samples += incoming_samples
                stream.max_backlog_samples = max(stream.max_backlog_samples, stream.queued_samples)
                stream.max_stream_lag_samples = max(
                    stream.max_stream_lag_samples,
                    max(0, stream.accepted_samples - stream.decoded_samples),
                )
                self._observe_global_high_water_locked()
                self._mark_ready_locked(stream)
                self._condition.notify_all()
                return {
                    "accepted_samples": incoming_samples,
                    "backlog_audio_ms": stream.queued_samples * 1000.0 / 16_000,
                    "pending_jobs": self.pending_jobs,
                }
        if fault is not None:
            await self.on_fault(connection_id, stream_id, fault[0], fault[1])
            raise SchedulerError(fault[0], fault[1])
        raise SchedulerError("scheduler_admission_failed")

    async def finish_stream(self, connection_id: str, stream_id: str) -> DispatchResult:
        identity = (connection_id, stream_id)
        fault: dict[str, Any] | None = None
        future: asyncio.Future[DispatchResult]
        async with self._condition:
            stream = self.streams.get(identity)
            if stream is None or stream.fenced or not stream.accepting:
                raise SchedulerError("stream_not_started")
            residual_samples = len(stream.tail) // 2
            needed = int(residual_samples > 0) + 1
            if self.pending_jobs + needed > self.max_pending_jobs:
                fault = {
                    "reason": "global_pending_job_limit_at_eos",
                    "accepted_audio_ms": stream.queued_samples * 1000.0 / 16_000,
                    "pending_jobs": self.pending_jobs,
                    "pending_job_limit": self.max_pending_jobs,
                }
                fault["terminal_metrics"] = self._terminal_metrics_locked(stream, fault)
                self._fence_locked(stream)
                self._overrun_total += 1
            else:
                stream.accepting = False
                now = time.perf_counter()
                if residual_samples:
                    stream.pending.append(DecodeJob(
                        "push", bytes(stream.tail), stream.accepted_samples, now,
                        local_stale_reject_observer=stream.local_stale_reject_observer,
                    ))
                    stream.tail.clear()
                future = asyncio.get_running_loop().create_future()
                stream.pending.append(DecodeJob(
                    "finish", b"", stream.accepted_samples, now, future=future,
                    local_stale_reject_observer=stream.local_stale_reject_observer,
                ))
                self._observe_global_high_water_locked()
                self._mark_ready_locked(stream)
                self._condition.notify_all()
        if fault is not None:
            await self.on_fault(connection_id, stream_id, "stream_scheduler_overrun", fault)
            raise SchedulerError("stream_scheduler_overrun", fault)
        return await future

    async def fence_stream(self, connection_id: str, stream_id: str, reason: str = "stream_closed") -> None:
        async with self._condition:
            stream = self.streams.get((connection_id, stream_id))
            if stream is not None:
                self._fence_locked(stream)
                self._condition.notify_all()

    def is_current(self, key: RequestKey) -> bool:
        stream = self.streams.get((key.connection_id, key.stream_id))
        return bool(stream and not stream.fenced and stream.active_key == key)

    @property
    def pending_jobs(self) -> int:
        return sum(len(stream.pending) for stream in self.streams.values() if not stream.fenced)

    @property
    def active_jobs(self) -> int:
        return int(self._active_job is not None)

    @property
    def active_stream_count(self) -> int:
        return sum(not stream.fenced for stream in self.streams.values())

    def stream_snapshot(self, connection_id: str, stream_id: str) -> dict[str, Any] | None:
        stream = self.streams.get((connection_id, stream_id))
        if stream is None or stream.fenced:
            return None
        wait = stream.waits_ms[-self.history_limit:]
        wall = stream.decode_wall_ms[-self.history_limit:]
        backlog_samples = stream.queued_samples
        return {
            "connection_id": connection_id,
            "stream_id": stream_id,
            "scheduler_pending_jobs": len(stream.pending),
            "scheduler_pending_jobs_max": stream.max_pending_jobs,
            "effective_max_backlog_ms": stream.chunk_ms * self.max_backlog_chunks,
            "qwen_max_backlog_chunks": self.max_backlog_chunks,
            "model_chunk_ms": stream.chunk_ms,
            "active_stream_count_max": stream.max_active_streams,
            "scheduler_backlog_audio_ms": round(backlog_samples * 1000.0 / 16_000, 3),
            "stream_lag_ms": round(max(0, stream.accepted_samples - stream.decoded_samples) * 1000.0 / 16_000, 3),
            "scheduler_max_backlog_audio_ms": round(stream.max_backlog_samples * 1000.0 / 16_000, 3),
            "stream_lag_max_ms": round(stream.max_stream_lag_samples * 1000.0 / 16_000, 3),
            "scheduler_wait_p50_ms": _percentile(wait, 0.50),
            "scheduler_wait_p95_ms": _percentile(wait, 0.95),
            "scheduler_wait_max_ms": max(wait) if wait else None,
            "scheduler_wait_sample_count": len(wait),
            "decode_wall_p50_ms": _percentile(wall, 0.50),
            "decode_wall_p95_ms": _percentile(wall, 0.95),
            "decode_wall_max_ms": round(max(wall), 3) if wall else None,
            "decode_wall_sample_count": len(wall),
            "scheduler_metric_history_limit": self.history_limit,
            "decode_steps_delta_total": stream.decode_steps,
            "accepted_audio_ms": round(stream.accepted_total_samples * 1000.0 / 16_000, 3),
            "dispatched_audio_ms": round(stream.dispatched_total_samples * 1000.0 / 16_000, 3),
        }

    def _terminal_metrics_locked(self, stream: StreamQueue, fault: dict[str, Any]) -> dict[str, Any]:
        """Capture bounded numeric diagnostics before an overrun fences a stream.

        This allowlisted snapshot intentionally contains no stream/connection IDs,
        transcript text, PCM, or request metadata. It is safe to carry in the
        terminal WebSocket error and benchmark row.
        """
        waits = stream.waits_ms[-self.history_limit:]
        walls = stream.decode_wall_ms[-self.history_limit:]
        backlog_samples = stream.queued_samples
        stream_lag_samples = max(0, stream.accepted_samples - stream.decoded_samples)
        if "backlog_limit_ms" in fault:
            limit_kind = "backlog_ms"
            limit_value = fault.get("backlog_limit_ms")
        else:
            limit_kind = "pending_jobs"
            limit_value = fault.get("pending_job_limit")
        return {
            "scheduler_wait_p50_ms": _percentile(waits, 0.50),
            "scheduler_wait_p95_ms": _percentile(waits, 0.95),
            "scheduler_wait_max_ms": round(max(waits), 3) if waits else None,
            "scheduler_wait_sample_count": len(waits),
            "decode_wall_p50_ms": _percentile(walls, 0.50),
            "decode_wall_p95_ms": _percentile(walls, 0.95),
            "decode_wall_max_ms": round(max(walls), 3) if walls else None,
            "decode_wall_sample_count": len(walls),
            "pending_jobs": self.pending_jobs,
            "stream_pending_jobs": len(stream.pending),
            "active_stream_count": self.active_stream_count,
            "backlog_audio_ms": round(backlog_samples * 1000.0 / 16_000, 3),
            "stream_lag_ms": round(stream_lag_samples * 1000.0 / 16_000, 3),
            "max_backlog_audio_ms": round(stream.max_backlog_samples * 1000.0 / 16_000, 3),
            "max_stream_lag_ms": round(stream.max_stream_lag_samples * 1000.0 / 16_000, 3),
            "accepted_audio_total_ms": round(stream.accepted_total_samples * 1000.0 / 16_000, 3),
            "dispatched_audio_total_ms": round(stream.dispatched_total_samples * 1000.0 / 16_000, 3),
            "scheduler_metric_history_limit": self.history_limit,
            "overrun_reason": fault.get("reason"),
            "overrun_limit_kind": limit_kind,
            "overrun_limit_value": limit_value,
            "effective_max_backlog_ms": stream.chunk_ms * self.max_backlog_chunks,
            "qwen_max_backlog_chunks": self.max_backlog_chunks,
            "model_chunk_ms": stream.chunk_ms,
        }

    def snapshot(self) -> dict[str, Any]:
        active = [stream for stream in self.streams.values() if not stream.fenced]
        backlog_samples = sum(stream.queued_samples for stream in active)
        stream_backlogs = [stream.queued_samples for stream in active]
        stream_lags = [max(0, stream.accepted_samples - stream.decoded_samples) for stream in active]
        return {
            "active_stream_count": len(active),
            "max_active_streams": self.max_active_streams,
            "pending_decode_count": self.pending_jobs,
            "active_decode_count": self.active_jobs,
            "ready_stream_count": len(self._ready_set),
            "qwen_scheduler_backlog_ms": round(backlog_samples * 1000.0 / 16_000, 3),
            "qwen_scheduler_max_stream_backlog_ms": round(max(stream_backlogs, default=0) * 1000.0 / 16_000, 3),
            "qwen_scheduler_stream_lag_ms": round(sum(stream_lags) * 1000.0 / 16_000, 3),
            "qwen_scheduler_max_stream_lag_ms": round(max(stream_lags, default=0) * 1000.0 / 16_000, 3),
            "qwen_scheduler_wait_p50_ms": _percentile(list(self._waits_ms), 0.50),
            "qwen_scheduler_wait_p95_ms": _percentile(list(self._waits_ms), 0.95),
            "qwen_scheduler_wait_max_ms": max(self._waits_ms) if self._waits_ms else None,
            "qwen_decode_wall_p50_ms": _percentile(list(self._decode_wall_ms), 0.50),
            "qwen_decode_wall_p95_ms": _percentile(list(self._decode_wall_ms), 0.95),
            "qwen_decode_steps_delta_total": self._decode_steps,
            "qwen_scheduler_overrun_total": self._overrun_total,
            "qwen_decode_budget_overrun_total": self._decode_budget_overrun_total,
            "qwen_decode_slo_violation_total": self._decode_slo_violation_total,
        }

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def _mark_ready_locked(self, stream: StreamQueue) -> None:
        identity = stream.identity
        if stream.pending and not stream.fenced and identity not in self._ready_set:
            self.ready.append(identity)
            self._ready_set.add(identity)

    def _observe_global_high_water_locked(self) -> None:
        active = [stream for stream in self.streams.values() if not stream.fenced]
        active_count = len(active)
        pending_count = self.pending_jobs
        for stream in active:
            stream.max_pending_jobs = max(stream.max_pending_jobs, pending_count)
            stream.max_active_streams = max(stream.max_active_streams, active_count)

    def _fence_locked(self, stream: StreamQueue) -> None:
        identity = stream.identity
        stream.fenced = True
        stream.accepting = False
        stream.active_key = None
        for job in stream.pending:
            if job.future is not None and not job.future.done():
                job.future.set_exception(SchedulerError("stream_fenced"))
        stream.pending.clear()
        stream.tail.clear()
        self.ready = deque(item for item in self.ready if item != identity)
        self._ready_set.discard(identity)
        self.streams.pop(identity, None)

    async def _dispatch_loop(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closing or bool(self.ready))
                if self._closing:
                    return
                identity = self.ready.popleft()
                self._ready_set.discard(identity)
                stream = self.streams.get(identity)
                if stream is None or stream.fenced or not stream.pending:
                    continue
                job = stream.pending.popleft()
                stream.next_revision += 1
                key = RequestKey(stream.connection_id, stream.stream_id, stream.next_revision)
                stream.active_key = key
                if stream.pending:
                    self._mark_ready_locked(stream)
                dispatch_at = time.perf_counter()
                wait_ms = max(0.0, (dispatch_at - job.ready_at) * 1000.0)
                backlog_ms = stream.queued_samples * 1000.0 / 16_000
                lag_ms = max(0, stream.accepted_samples - stream.decoded_samples) * 1000.0 / 16_000
                metrics = {
                    "qwen_scheduler_wait_ms": round(wait_ms, 3),
                    "dispatch_at": dispatch_at,
                    "job_ready_at": job.ready_at,
                    "first_audio_at": stream.first_audio_at,
                    "qwen_scheduler_revision": key.scheduler_revision,
                    "qwen_decode_backlog_ms": round(backlog_ms, 3),
                    "stream_lag_ms": round(lag_ms, 3),
                    "pending_decode_count": self.pending_jobs,
                    "active_decode_count": 1,
                    "ready_stream_count": len(self._ready_set),
                    "decode_kind": job.kind,
                    "audio_cursor_ms": round(job.cursor_end_samples * 1000.0 / 16_000, 3),
                }
                job.metrics = metrics
                self._active_job = (stream, key, job)

            try:
                reply = await self.execute(key, job)
                if not isinstance(reply, dict):
                    raise SchedulerError("invalid_worker_response")
                error: Exception | None = None
            except Exception as exc:
                reply = {}
                error = exc

            async with self._condition:
                current = (
                    self.streams.get(identity) is stream
                    and not stream.fenced
                    and stream.active_key == key
                )
                self._active_job = None
                if error is None and current:
                    raw_wall = reply.get("decode_wall_ms", 0.0)
                    raw_steps = reply.get("decode_steps_delta", 0)
                    if (isinstance(raw_wall, bool) or not isinstance(raw_wall, (int, float))
                            or not math.isfinite(float(raw_wall)) or raw_wall < 0 or isinstance(raw_steps, bool)
                            or not isinstance(raw_steps, int) or raw_steps not in (0, 1)):
                        error = SchedulerError("invalid_worker_response")
                        self._fence_locked(stream)
                    else:
                        wall_ms = float(raw_wall)
                        steps = raw_steps
                        decode_overrun = steps > 0 and wall_ms > stream.chunk_ms
                        wait_ms = float(metrics["qwen_scheduler_wait_ms"])
                        stream.waits_ms.append(wait_ms)
                        stream.waits_ms = stream.waits_ms[-self.history_limit:]
                        stream.dispatched_total_samples += len(job.pcm16le) // 2
                        stream.decoded_samples = max(stream.decoded_samples, job.cursor_end_samples)
                        stream.max_backlog_samples = max(stream.max_backlog_samples, stream.queued_samples)
                        stream.max_stream_lag_samples = max(
                            stream.max_stream_lag_samples,
                            max(0, stream.accepted_samples - stream.decoded_samples),
                        )
                        self._waits_ms.append(wait_ms)
                        if steps > 0:
                            stream.decode_wall_ms.append(wall_ms)
                            stream.decode_wall_ms = stream.decode_wall_ms[-self.history_limit:]
                            stream.decode_steps += steps
                            self._decode_wall_ms.append(wall_ms)
                            self._decode_steps += steps
                            if decode_overrun:
                                self._decode_budget_overrun_total += 1
                            if wall_ms >= 100:
                                self._decode_slo_violation_total += 1
                        metrics.update({
                            "qwen_decode_call_wall_ms": round(wall_ms, 3),
                            "qwen_decode_steps_delta": steps,
                            "stream_lag_ms": round(max(0, stream.accepted_samples - stream.decoded_samples) * 1000.0 / 16_000, 3),
                            "accepted_audio_ms": round(stream.accepted_total_samples * 1000.0 / 16_000, 3),
                            "dispatched_audio_ms": round(stream.dispatched_total_samples * 1000.0 / 16_000, 3),
                            "decode_overrun": decode_overrun,
                            "qwen_decode_slo_target_ms": 100,
                            "qwen_decode_slo_violation": bool(steps > 0 and wall_ms >= 100),
                        })
                elif error is not None and current:
                    self._fence_locked(stream)
                self._condition.notify_all()

            if error is not None:
                failure_code = error.code if isinstance(error, SchedulerError) else "stream_worker_failed"
                if job.future is not None and not job.future.done():
                    job.future.set_exception(SchedulerError(failure_code))
                if current:
                    await self.on_fault(stream.connection_id, stream.stream_id, failure_code, {})
                continue
            if not current:
                if job.local_stale_reject_observer is not None:
                    try:
                        job.local_stale_reject_observer(
                            key.stream_id,
                            "scheduler_request_key_stale",
                        )
                    except Exception:
                        pass
                if job.future is not None and not job.future.done():
                    job.future.set_exception(SchedulerError("stream_fenced"))
                continue

            result = DispatchResult(key, reply, dict(metrics))
            if job.future is not None:
                async with self._condition:
                    if stream.active_key == key:
                        stream.active_key = None
                if not job.future.done():
                    job.future.set_result(result)
            elif job.kind == "push":
                try:
                    await self.on_result(key, job, reply, metrics)
                except Exception:
                    await self.fence_stream(stream.connection_id, stream.stream_id, "result_delivery_failed")
                    await self.on_fault(
                        stream.connection_id, stream.stream_id,
                        "stream_result_delivery_failed", {},
                    )
                finally:
                    async with self._condition:
                        if stream.active_key == key:
                            stream.active_key = None
                        self._condition.notify_all()
