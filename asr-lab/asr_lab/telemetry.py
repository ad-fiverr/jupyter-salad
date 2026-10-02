"""Low-overhead, secret-free process, host and optional NVML telemetry."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def process_rss_mib(path: str | Path = "/proc/self/status") -> float | None:
    try:
        for line in Path(path).read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                kib = int(line.split()[1])
                return round(kib / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def system_memory_mib(path: str | Path = "/proc/meminfo") -> dict[str, float | None]:
    try:
        entries: dict[str, int] = {}
        for line in Path(path).read_text(encoding="ascii").splitlines():
            key, separator, rest = line.partition(":")
            if separator and key in {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"}:
                entries[key] = int(rest.split()[0])
        total = entries.get("MemTotal")
        available = entries.get("MemAvailable")
        if available is None and all(key in entries for key in ("MemFree", "Buffers", "Cached")):
            available = entries["MemFree"] + entries["Buffers"] + entries["Cached"]
        if total is None:
            return {"total_mib": None, "used_mib": None}
        return {
            "total_mib": round(total / 1024.0, 1),
            "used_mib": round(max(0, total - available) / 1024.0, 1) if available is not None else None,
        }
    except (OSError, ValueError, IndexError):
        return {"total_mib": None, "used_mib": None}


def _empty_gpu_telemetry(provider: str = "unavailable") -> dict[str, Any]:
    return {
        "available": False,
        "provider": provider,
        "scope": "device_global",
        "backend_attributed": False,
        "device": None,
        "utilization_pct": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "temperature_c": None,
        "power_w": None,
    }


def gpu_compute_state(broker: Any, settings: Any) -> dict[str, Any]:
    """Report CUDA/model compute visibility independently of NVML telemetry."""
    workers = getattr(broker, "worker_metrics", []) if broker is not None else []
    snapshots = [
        worker.get("vram_after_warmup", {})
        for worker in workers
        if isinstance(worker, dict)
    ]
    detected = next((item for item in snapshots if item.get("available") is True), None)
    if detected is None:
        detected = next((item for item in snapshots if item.get("device")), None)
    available = bool(detected and detected.get("available") is True)
    return {
        "available": available,
        "cuda": available,
        "device": detected.get("device") if detected else None,
        "backend": getattr(settings, "active_backend", getattr(settings, "backend", None)),
        "model_loaded": bool(broker and getattr(broker, "ready", False)),
    }


def _torch_gpu_fallback(torch_module: Any | None = None) -> dict[str, Any]:
    """Best-effort global CUDA memory snapshot; never imply model attribution."""
    try:
        if torch_module is None:
            import torch as torch_module
        cuda = torch_module.cuda
        if not cuda.is_available():
            return _empty_gpu_telemetry()
        device = cuda.get_device_name(0)
        free, total = cuda.mem_get_info()
        return {
            "available": True,
            "provider": "torch_fallback",
            "scope": "device_global",
            "backend_attributed": False,
            "device": str(device),
            "utilization_pct": None,
            "vram_used_mib": round((total - free) / (1024 * 1024), 1),
            "vram_total_mib": round(total / (1024 * 1024), 1),
            "temperature_c": None,
            "power_w": None,
        }
    except Exception:
        return _empty_gpu_telemetry()


def gpu_telemetry(nvml: Any | None = None, *, torch_module: Any | None = None) -> dict[str, Any]:
    """Read NVML if present, otherwise return a safe device-global Torch fallback."""
    initialized = False
    try:
        if nvml is None:
            import pynvml as nvml
        nvml.nvmlInit()
        initialized = True
        if nvml.nvmlDeviceGetCount() > 0:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            def read(callable_: Any) -> Any | None:
                try:
                    return callable_()
                except Exception:
                    return None

            name = read(lambda: nvml.nvmlDeviceGetName(handle))
            memory = read(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
            utilization = read(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
            temperature = read(
                lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            )
            power_mw = read(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if name is not None or memory is not None or utilization is not None:
                return {
                    "available": True,
                    "provider": "nvml",
                    "scope": "device_global",
                    "backend_attributed": False,
                    "device": str(name) if name is not None else None,
                    "utilization_pct": int(utilization.gpu) if utilization is not None else None,
                    "vram_used_mib": round(memory.used / (1024 * 1024), 1) if memory is not None else None,
                    "vram_total_mib": round(memory.total / (1024 * 1024), 1) if memory is not None else None,
                    "temperature_c": int(temperature) if temperature is not None else None,
                    "power_w": round(power_mw / 1000.0, 1) if power_mw is not None else None,
                }
    except Exception:
        pass
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
    return _torch_gpu_fallback(torch_module)


def collect_telemetry(
    broker: Any, settings: Any, qwen_scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an allowlisted schema; never include environment or process args."""
    compute = gpu_compute_state(broker, settings)
    gpu = gpu_telemetry()
    if not gpu.get("device") and compute.get("device"):
        gpu["device"] = compute["device"]
    snapshot = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": getattr(settings, "active_backend", settings.backend),
        "production_backend": settings.backend,
        "model_id": getattr(settings, "active_model_id", settings.model_id),
        "model_revision": getattr(settings, "active_model_revision", settings.model_revision),
        "model_loaded": bool(broker and broker.ready),
        "ready": bool(broker and broker.ready),
        "workers": settings.workers,
        "queue_depth": getattr(broker, "queued_jobs", None) if broker is not None else 0,
        "process_rss_mib": process_rss_mib(),
        "system_ram": system_memory_mib(),
        "gpu_compute": compute,
        "gpu_telemetry": gpu,
        # Backwards-compatible telemetry alias; provider/scope identify it.
        "gpu": gpu,
    }
    if qwen_scheduler is not None:
        snapshot["qwen_scheduler"] = {
            key: qwen_scheduler.get(key)
            for key in (
                "active_stream_count", "max_active_streams", "pending_decode_count",
                "active_decode_count", "ready_stream_count", "qwen_scheduler_backlog_ms",
                "qwen_scheduler_max_stream_backlog_ms", "qwen_scheduler_wait_p50_ms",
                "qwen_scheduler_stream_lag_ms", "qwen_scheduler_max_stream_lag_ms",
                "qwen_scheduler_wait_p95_ms", "qwen_scheduler_wait_max_ms",
                "qwen_decode_wall_p50_ms", "qwen_decode_wall_p95_ms",
                "qwen_decode_steps_delta_total", "qwen_scheduler_overrun_total",
                "qwen_decode_budget_overrun_total", "qwen_decode_slo_violation_total",
            )
        }
    return snapshot
