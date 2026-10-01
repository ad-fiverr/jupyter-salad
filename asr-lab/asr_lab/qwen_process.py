"""Async JSON-line RPC client for the isolated Qwen/vLLM process."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from .qwen_streaming import StreamingError

_WORKER_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP",
    "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT", "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM", "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_LOGGING_LEVEL",
})


def qwen_worker_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pass only runtime/cache variables the isolated model worker requires."""
    source = os.environ if environ is None else environ
    return {key: value for key, value in source.items() if key in _WORKER_ENV_KEYS}


class QwenWorkerProcess:
    def __init__(
        self,
        *,
        python: str,
        script: str | Path,
        request_timeout_seconds: float = 600.0,
    ) -> None:
        self.python = python
        self.script = str(script)
        self.request_timeout_seconds = request_timeout_seconds
        self.process: asyncio.subprocess.Process | None = None
        self.reader_task: asyncio.Task[None] | None = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.write_lock = asyncio.Lock()
        self.ready = False
        self._closed = False
        self.model_report: dict[str, Any] = {}

    async def start(
        self,
        *,
        model_id: str,
        model_revision: str,
        gpu_memory_utilization: float,
        max_active_sessions: int,
        warmup_chunk_ms: int,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
    ) -> dict[str, Any]:
        self.process = await asyncio.create_subprocess_exec(
            self.python,
            "-u",
            self.script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            env=qwen_worker_environment(),
        )
        self.reader_task = asyncio.create_task(self._read_responses(), name="qwen-worker-rpc-reader")
        reply = await self.request(
            "load",
            model_id=model_id,
            model_revision=model_revision,
            gpu_memory_utilization=gpu_memory_utilization,
            max_active_sessions=max_active_sessions,
            warmup_chunk_ms=warmup_chunk_ms,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            timeout_seconds=1800.0,
        )
        report = reply.get("model_report")
        if not isinstance(report, dict):
            raise StreamingError("invalid_worker_response")
        self.model_report = report
        self.ready = True
        return report

    async def request(
        self,
        operation: str,
        *,
        timeout_seconds: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None or self._closed:
            raise StreamingError("stream_worker_unavailable")
        request_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        encoded = json.dumps(
            {"id": request_id, "operation": operation, "payload": payload},
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            async with self.write_lock:
                process.stdin.write(encoded)
                await process.stdin.drain()
            async with asyncio.timeout(timeout_seconds or self.request_timeout_seconds):
                return await future
        except TimeoutError as exc:
            self.pending.pop(request_id, None)
            future.cancel()
            raise StreamingError("stream_worker_timeout") from exc
        except Exception:
            self.pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise

    async def _read_responses(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                raw = await self.process.stdout.readline()
                if not raw:
                    break
                try:
                    response = json.loads(raw)
                    request_id = response.get("id")
                except (json.JSONDecodeError, UnicodeError):
                    continue
                future = self.pending.pop(request_id, None) if isinstance(request_id, str) else None
                if future is None or future.done():
                    continue
                if response.get("ok") is True and isinstance(response.get("result"), dict):
                    future.set_result(response["result"])
                else:
                    future.set_exception(StreamingError("stream_worker_failed"))
        finally:
            self.ready = False
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(StreamingError("stream_worker_exited"))
            self.pending.clear()

    async def close(self) -> None:
        self._closed = True
        self.ready = False
        process = self.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self.reader_task is not None and not self.reader_task.done():
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        for future in self.pending.values():
            if not future.done():
                future.set_exception(StreamingError("stream_worker_closed"))
        self.pending.clear()
