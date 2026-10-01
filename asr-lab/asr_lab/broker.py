"""Round-robin inference workers with explicit per-job connection ownership."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .backends.base import create_backend

logger = logging.getLogger("asr_lab.broker")


def _gpu_snapshot() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "total_mib": round(total / (1024 * 1024), 1),
            "used_global_mib": round((total - free) / (1024 * 1024), 1),
            "process_allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
            "source": "torch_cuda_startup_snapshot",
            "scope": "device_global",
            "backend_attributed": False,
            "process_allocated_scope": "pytorch_allocator_only",
        }
    except Exception:
        return {"available": False}


@dataclass
class InferenceJob:
    connection_id: str
    source: str
    speaker: str
    pcm16le: bytes
    segment_start_s: float
    segment_end_s: float
    segment_wait_ms: float
    audio_duration_ms: float
    server_eos_at: float
    submitted_at: float
    result: asyncio.Future[dict[str, Any]]
    job_id: str


class InferenceBroker:
    def __init__(self, *, backend_name: str, model_id: str, model_revision: str | None, workers: int, queue_size: int):
        if workers < 1:
            raise ValueError("ASR_WORKERS must be a positive integer.")
        self.backend_name = backend_name
        self.model_id = model_id
        self.model_revision = model_revision
        self.worker_count = workers
        # Separate worker queues mirror the historical one-model-per-worker
        # design. A shared counter preserves the configured total queue bound.
        self.worker_queues: list[asyncio.Queue[InferenceJob | None]] = [
            asyncio.Queue() for _ in range(workers)
        ]
        self.max_queue_size = queue_size
        self.queued_jobs = 0
        self.next_worker_index = 0
        self.job_owners: dict[str, str] = {}
        self.tasks: list[asyncio.Task[None]] = []
        self.ready_workers = 0
        self.failed = False
        self.closing = False
        self.worker_metrics: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.ready_workers == self.worker_count and not self.failed and not self.closing

    async def start(self) -> None:
        for worker_index in range(self.worker_count):
            self.tasks.append(asyncio.create_task(self._worker(worker_index), name=f"asr-worker-{worker_index}"))
        try:
            while self.ready_workers < self.worker_count and not self.failed:
                if any(task.done() for task in self.tasks):
                    self.failed = True
                    break
                await asyncio.sleep(0.05)
            if not self.failed:
                return
        except asyncio.CancelledError:
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            raise
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        raise RuntimeError("backend_initialization_failed")

    async def _worker(self, worker_index: int) -> None:
        backend = create_backend(self.backend_name)
        queue = self.worker_queues[worker_index]
        try:
            before = _gpu_snapshot()
            load_started = time.perf_counter()
            await asyncio.to_thread(backend.load, self.model_id, self.model_revision or "")
            load_ms = (time.perf_counter() - load_started) * 1000
            after_load = _gpu_snapshot()
            warmup_started = time.perf_counter()
            await asyncio.to_thread(backend.warmup)
            warmup_ms = (time.perf_counter() - warmup_started) * 1000
            after_warmup = _gpu_snapshot()
            self.worker_metrics.append({
                "worker": worker_index,
                "model_load_ms": round(load_ms, 2),
                "warmup_ms": round(warmup_ms, 2),
                "vram_before": before,
                "vram_after_load": after_load,
                "vram_after_warmup": after_warmup,
            })
            self.ready_workers += 1
            while True:
                job = await queue.get()
                try:
                    if job is None:
                        return
                    self.queued_jobs -= 1
                    def transcribe_timed() -> tuple[str, float, float]:
                        started_at = time.perf_counter()
                        text_value = backend.transcribe(job.pcm16le)
                        finished_at = time.perf_counter()
                        return text_value, started_at, finished_at

                    text, inference_started_at, inference_finished_at = await asyncio.to_thread(transcribe_timed)
                    inference_ms = max(0.0, (inference_finished_at - inference_started_at) * 1000)
                    output = {
                        "type": "transcript",
                        "schema_version": 1,
                        "event": "transcript",
                        "text": text,
                        "speaker": job.speaker,
                        "start": job.segment_start_s,
                        "end": job.segment_end_s,
                        "source": job.source,
                        "backend": self.backend_name,
                        "model_revision": self.model_revision,
                        "final": True,
                        "MODEL_INFERENCE_MS": round(inference_ms, 2),
                        "SERVER_MODEL_INFERENCE_MS": round(inference_ms, 2),
                        "AUDIO_DURATION_MS": round(job.audio_duration_ms, 2),
                        "audio_duration_ms": round(job.audio_duration_ms, 2),
                        "SEGMENT_WAIT_MS": round(job.segment_wait_ms, 2),
                        "SERVER_TO_TRANSCRIPT_MS": round(
                            max(0.0, (time.perf_counter() - job.submitted_at) * 1000), 2
                        ),
                        "SERVER_ENDPOINTING_MS": round(max(0.0, job.submitted_at - job.server_eos_at) * 1000, 2),
                        "SERVER_QUEUE_WAIT_MS": round(max(0.0, inference_started_at - job.submitted_at) * 1000, 2),
                        "queue_wait_ms": round(max(0.0, inference_started_at - job.submitted_at) * 1000, 2),
                        "request_id": job.job_id,
                        "_internal_timing": {"model_finished_at": inference_finished_at},
                    }
                    if self.backend_name == "parakeet":
                        from .backends.parakeet import should_discard_historical_transcript

                        output["_discarded_historical_output"] = should_discard_historical_transcript(text)
                    owner = self.job_owners.get(job.job_id)
                    if owner == job.connection_id and not job.result.cancelled() and not job.result.done():
                        job.result.set_result(output)
                    elif owner != job.connection_id and not job.result.done():
                        job.result.set_exception(RuntimeError("job_owner_mismatch"))
                except Exception as exc:
                    logger.error("ASR inference failed worker=%d exception_type=%s", worker_index, type(exc).__name__)
                    if job is not None and not job.result.done():
                        job.result.set_exception(RuntimeError("inference_failed"))
                finally:
                    if job is not None:
                        self.job_owners.pop(job.job_id, None)
                    queue.task_done()
        except Exception as exc:
            self.failed = True
            logger.error("ASR worker initialization failed worker=%d exception_type=%s", worker_index, type(exc).__name__)
            raise
        finally:
            try:
                await asyncio.to_thread(backend.close)
            except Exception:
                pass

    async def transcribe(
        self, *, connection_id: str, source: str, speaker: str, pcm16le: bytes,
        segment_start_s: float, segment_end_s: float, segment_wait_ms: float,
        server_eos_at: float,
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("backend_not_ready")
        if self.queued_jobs >= self.max_queue_size:
            raise RuntimeError("inference_queue_full")
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        job = InferenceJob(
            connection_id=connection_id,
            source=source,
            speaker=speaker,
            pcm16le=pcm16le,
            segment_start_s=segment_start_s,
            segment_end_s=segment_end_s,
            segment_wait_ms=segment_wait_ms,
            audio_duration_ms=len(pcm16le) / 32.0,
            server_eos_at=server_eos_at,
            submitted_at=time.perf_counter(),
            result=future,
            job_id=uuid.uuid4().hex,
        )
        worker_index = self.next_worker_index
        self.next_worker_index = (self.next_worker_index + 1) % self.worker_count
        self.job_owners[job.job_id] = connection_id
        self.worker_queues[worker_index].put_nowait(job)
        self.queued_jobs += 1
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def close(self) -> None:
        self.closing = True
        active_indexes = [index for index, task in enumerate(self.tasks) if not task.done()]
        for worker_index in active_indexes:
            self.worker_queues[worker_index].put_nowait(None)
        active_tasks = [self.tasks[index] for index in active_indexes]
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
