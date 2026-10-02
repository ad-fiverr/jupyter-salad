"""Private line-oriented subprocess for the pinned official Qwen/vLLM wrapper.

Stdout is a JSON-RPC channel only. Model/library chatter is redirected to stderr;
audio, transcripts and credentials are never logged by this worker.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import time
from typing import Any

QWEN_MAX_NEW_TOKENS = 128
QWEN_MAX_MODEL_LEN = 4096


def build_experiment_config(
    *,
    gpu_memory_utilization: float,
    max_active_sessions: int,
    warmup_chunk_ms: int,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
) -> dict[str, Any]:
    return {
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_active_sessions": max_active_sessions,
        "max_num_seqs": 1,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "max_model_len": QWEN_MAX_MODEL_LEN,
        "warmup_chunk_ms": warmup_chunk_ms,
        "unfixed_chunk_num": unfixed_chunk_num,
        "unfixed_token_num": unfixed_token_num,
    }


def runtime_provenance(
    torch_module: Any,
    version_lookup: Any | None = None,
    experiment_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read installed distribution/runtime versions without exposing environment data."""
    if version_lookup is None:
        from importlib.metadata import version as version_lookup

    def installed_version(distribution: str) -> str | None:
        try:
            return str(version_lookup(distribution))
        except Exception:
            # Missing metadata should be visible as unknown provenance, not
            # take down a model that already loaded successfully.
            return None

    return {
        "qwen_asr_version": installed_version("qwen-asr"),
        "vllm_version": installed_version("vllm"),
        "transformers_version": installed_version("transformers"),
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda) if torch_module.version.cuda else None,
        "experiment_config": dict(experiment_config or {}),
    }


class QwenWorkerEngine:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.sessions: dict[str, Any] = {}
        self.max_active_sessions = 0
        self.model_info: dict[str, Any] = {}

    def load(self, *, model_id: str, model_revision: str, gpu_memory_utilization: float,
             max_active_sessions: int, warmup_chunk_ms: int,
             unfixed_chunk_num: int, unfixed_token_num: int) -> dict[str, Any]:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ASRModel

        before = self._gpu_snapshot(torch)
        experiment_config = build_experiment_config(
            gpu_memory_utilization=gpu_memory_utilization,
            max_active_sessions=max_active_sessions,
            warmup_chunk_ms=warmup_chunk_ms,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
        )
        download_started = time.perf_counter()
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            token=os.environ.get("HF_TOKEN") or None,
        )
        snapshot_download_ms = (time.perf_counter() - download_started) * 1000
        engine_started = time.perf_counter()
        self.model = Qwen3ASRModel.LLM(
            model=snapshot_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=experiment_config["max_new_tokens"],
            max_model_len=experiment_config["max_model_len"],
            max_num_seqs=1,
        )
        engine_load_ms = (time.perf_counter() - engine_started) * 1000
        after_load = self._gpu_snapshot(torch)
        self.max_active_sessions = max_active_sessions

        warm_state = self.model.init_streaming_state(
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=warmup_chunk_ms / 1000.0,
        )
        warmup_started = time.perf_counter()
        self.model.streaming_transcribe(
            np.zeros((warmup_chunk_ms * 16,), dtype=np.float32), warm_state,
        )
        self.model.finish_streaming_transcribe(warm_state)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000
        after_warmup = self._gpu_snapshot(torch)
        self.model_info = {
            "model_id": model_id,
            "model_revision": model_revision,
            "runtime_provenance": runtime_provenance(torch, experiment_config=experiment_config),
            "experiment_config": experiment_config,
        }
        return {
            **self.model_info,
            "snapshot_download_ms": round(snapshot_download_ms, 2),
            "model_load_ms": round(engine_load_ms, 2),
            "warmup_ms": round(warmup_ms, 2),
            "vram_before": before,
            "vram_after_load": after_load,
            "vram_after_warmup": after_warmup,
        }

    @staticmethod
    def _gpu_snapshot(torch_module: Any) -> dict[str, Any]:
        try:
            if not torch_module.cuda.is_available():
                return {"available": False}
            torch_module.cuda.synchronize()
            free, total = torch_module.cuda.mem_get_info()
            return {
                "available": True,
                "device": torch_module.cuda.get_device_name(0),
                "total_mib": round(total / (1024 * 1024), 1),
                "used_global_mib": round((total - free) / (1024 * 1024), 1),
                "process_allocated_mib": round(torch_module.cuda.memory_allocated() / (1024 * 1024), 1),
                "scope": "device_global",
                "backend_attributed": False,
                "process_allocated_scope": "pytorch_allocator_only",
            }
        except Exception:
            return {"available": False}

    def command(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation == "load":
            return {"model_report": self.load(**payload)}
        if self.model is None:
            raise RuntimeError("model_not_loaded")
        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str) or len(stream_id) != 32:
            raise ValueError("invalid_stream_id")
        if operation == "init":
            if stream_id in self.sessions:
                raise ValueError("stream_exists")
            if len(self.sessions) >= self.max_active_sessions:
                raise RuntimeError("stream_capacity_exceeded")
            state_init_started = time.perf_counter()
            state = self.model.init_streaming_state(
                context=payload.get("context", ""),
                language=payload.get("language"),
                unfixed_chunk_num=int(payload["unfixed_chunk_num"]),
                unfixed_token_num=int(payload["unfixed_token_num"]),
                chunk_size_sec=float(payload["chunk_size_sec"]),
            )
            state_init_wall_ms = (time.perf_counter() - state_init_started) * 1000.0
            self.sessions[stream_id] = state
            return {"stream_id": stream_id, "active_sessions": len(self.sessions),
                    "stream_state_init_wall_ms": round(state_init_wall_ms, 3)}
        state = self.sessions.get(stream_id)
        if state is None:
            raise KeyError("stream_not_found")
        scheduler_key = payload.get("scheduler_key")
        if scheduler_key is not None:
            if (
                not isinstance(scheduler_key, dict)
                or set(scheduler_key) != {"connection_id", "stream_id", "scheduler_revision"}
                or scheduler_key.get("connection_id") != payload.get("connection_id")
                or scheduler_key.get("stream_id") != stream_id
                or not isinstance(scheduler_key.get("connection_id"), str)
                or not 1 <= len(scheduler_key["connection_id"]) <= 128
                or not isinstance(scheduler_key.get("scheduler_revision"), int)
                or scheduler_key["scheduler_revision"] < 1
            ):
                raise ValueError("invalid_scheduler_key")
        if operation == "push":
            result = self._push(state, payload)
        elif operation == "finish":
            result = self._finish(state)
        elif operation == "close":
            self.sessions.pop(stream_id, None)
            return {"stream_id": stream_id, "closed": True}
        else:
            raise ValueError("unsupported_operation")
        if scheduler_key is not None:
            result["scheduler_key"] = dict(scheduler_key)
        return result

    def _push(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        encoded = payload.get("pcm16le_base64")
        if not isinstance(encoded, str) or len(encoded) > 1_400_000:
            raise ValueError("invalid_audio")
        pcm = base64.b64decode(encoded, validate=True)
        if not pcm or len(pcm) % 2:
            raise ValueError("invalid_audio")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        before_chunk = int(state.chunk_id)
        started = time.perf_counter()
        self.model.streaming_transcribe(audio, state)
        decode_wall_ms = (time.perf_counter() - started) * 1000
        after_chunk = int(state.chunk_id)
        if after_chunk - before_chunk not in (0, 1):
            raise ValueError("invalid_decode_steps_delta")
        return {
            "decoded": after_chunk > before_chunk,
            "decode_steps_delta": max(0, after_chunk - before_chunk),
            "decode_wall_ms": round(decode_wall_ms, 3),
            "text": str(state.text or ""),
            "language": str(state.language or "") or None,
        }

    def _finish(self, state: Any) -> dict[str, Any]:
        before_chunk = int(state.chunk_id)
        started = time.perf_counter()
        self.model.finish_streaming_transcribe(state)
        decode_wall_ms = (time.perf_counter() - started) * 1000
        after_chunk = int(state.chunk_id)
        return {
            "decode_steps_delta": max(0, after_chunk - before_chunk),
            "decode_wall_ms": round(decode_wall_ms, 3),
            "text": str(state.text or ""),
            "language": str(state.language or "") or None,
        }


def main() -> int:
    engine = QwenWorkerEngine()
    for raw_line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            operation = request.get("operation")
            payload = request.get("payload")
            if not isinstance(request_id, str) or not isinstance(payload, dict):
                raise ValueError("invalid_request")
            with contextlib.redirect_stdout(sys.stderr):
                result = engine.command(str(operation), payload)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            # Error details may contain package paths or remote response data.
            # Return a stable code and exception class only, never message text.
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "code": "qwen_worker_operation_failed",
                "exception_type": type(exc).__name__,
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
