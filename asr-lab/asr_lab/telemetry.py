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


def gpu_telemetry(nvml: Any | None = None) -> dict[str, Any]:
    """Read one NVML snapshot without letting optional metrics break the API."""
    empty = {
        "available": False,
        "device": None,
        "utilization_pct": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "temperature_c": None,
        "power_w": None,
    }
    initialized = False
    try:
        if nvml is None:
            import pynvml as nvml
        nvml.nvmlInit()
        initialized = True
        if nvml.nvmlDeviceGetCount() < 1:
            return empty
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
        return {
            "available": name is not None or memory is not None or utilization is not None,
            "device": str(name) if name is not None else None,
            "utilization_pct": int(utilization.gpu) if utilization is not None else None,
            "vram_used_mib": round(memory.used / (1024 * 1024), 1) if memory is not None else None,
            "vram_total_mib": round(memory.total / (1024 * 1024), 1) if memory is not None else None,
            "temperature_c": int(temperature) if temperature is not None else None,
            "power_w": round(power_mw / 1000.0, 1) if power_mw is not None else None,
        }
    except Exception:
        return empty
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def collect_telemetry(broker: Any, settings: Any) -> dict[str, Any]:
    """Return an allowlisted schema; never include environment or process args."""
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": settings.backend,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "model_loaded": bool(broker and broker.ready),
        "ready": bool(broker and broker.ready),
        "workers": settings.workers,
        "queue_depth": getattr(broker, "queued_jobs", None) if broker is not None else 0,
        "process_rss_mib": process_rss_mib(),
        "system_ram": system_memory_mib(),
        "gpu": gpu_telemetry(),
    }
