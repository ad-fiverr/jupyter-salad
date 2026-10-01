from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_lab.telemetry import collect_telemetry, gpu_compute_state, gpu_telemetry, process_rss_mib, system_memory_mib


SECRET = "do-not-return-this-token"


class TelemetryTests(unittest.TestCase):
    def test_process_rss_and_system_ram_parse_linux_proc_files(self):
        with tempfile.TemporaryDirectory() as directory:
            rss = Path(directory, "status")
            rss.write_text("Name:\tpython\nVmRSS:\t2048 kB\n", encoding="ascii")
            memory = Path(directory, "meminfo")
            memory.write_text(
                "MemTotal:       1048576 kB\nMemAvailable:    262144 kB\n",
                encoding="ascii",
            )
            self.assertEqual(process_rss_mib(rss), 2.0)
            self.assertEqual(system_memory_mib(memory), {"total_mib": 1024.0, "used_mib": 768.0})

    def test_proc_telemetry_returns_none_when_files_are_unavailable(self):
        self.assertIsNone(process_rss_mib("Z:/missing/status"))
        self.assertEqual(system_memory_mib("Z:/missing/meminfo"), {"total_mib": None, "used_mib": None})

    def test_nvml_snapshot_converts_units_and_always_shuts_down(self):
        class Memory:
            used = 1024 * 1024 * 2048
            total = 1024 * 1024 * 24576

        class Utilization:
            gpu = 67

        class FakeNvml:
            NVML_TEMPERATURE_GPU = 0

            def __init__(self):
                self.shutdown_calls = 0

            def nvmlInit(self):
                return None

            def nvmlDeviceGetCount(self):
                return 1

            def nvmlDeviceGetHandleByIndex(self, index):
                self.assert_zero(index)
                return "gpu-handle"

            @staticmethod
            def assert_zero(index):
                assert index == 0

            @staticmethod
            def nvmlDeviceGetName(_handle):
                return b"NVIDIA RTX Test"

            @staticmethod
            def nvmlDeviceGetMemoryInfo(_handle):
                return Memory()

            @staticmethod
            def nvmlDeviceGetUtilizationRates(_handle):
                return Utilization()

            @staticmethod
            def nvmlDeviceGetTemperature(_handle, _sensor):
                return 55

            @staticmethod
            def nvmlDeviceGetPowerUsage(_handle):
                return 150_000

            def nvmlShutdown(self):
                self.shutdown_calls += 1

        fake = FakeNvml()
        self.assertEqual(gpu_telemetry(fake), {
            "available": True, "provider": "nvml", "scope": "device_global", "backend_attributed": False,
            "device": "NVIDIA RTX Test", "utilization_pct": 67,
            "vram_used_mib": 2048.0, "vram_total_mib": 24576.0,
            "temperature_c": 55, "power_w": 150.0,
        })
        self.assertEqual(fake.shutdown_calls, 1)

    def test_nvml_optional_measurement_failure_keeps_other_gpu_fields(self):
        fake = types.SimpleNamespace(
            NVML_TEMPERATURE_GPU=0,
            nvmlInit=lambda: None,
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda _index: "gpu",
            nvmlDeviceGetName=lambda _handle: "GPU",
            nvmlDeviceGetMemoryInfo=lambda _handle: types.SimpleNamespace(used=1024, total=2048),
            nvmlDeviceGetUtilizationRates=lambda _handle: types.SimpleNamespace(gpu=10),
            nvmlDeviceGetTemperature=lambda *_args: (_ for _ in ()).throw(RuntimeError("unsupported")),
            nvmlDeviceGetPowerUsage=lambda *_args: (_ for _ in ()).throw(RuntimeError("unsupported")),
            nvmlShutdown=lambda: None,
        )
        result = gpu_telemetry(fake)
        self.assertTrue(result["available"])
        self.assertEqual(result["provider"], "nvml")
        self.assertEqual(result["device"], "GPU")
        self.assertEqual(result["utilization_pct"], 10)
        self.assertIsNone(result["temperature_c"])
        self.assertIsNone(result["power_w"])

    def test_no_nvml_or_gpu_is_a_valid_unavailable_state(self):
        no_cuda = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"pynvml": None}):
            no_package = gpu_telemetry(torch_module=no_cuda)
        self.assertFalse(no_package["available"])
        self.assertEqual(no_package["provider"], "unavailable")
        self.assertIsNone(no_package["device"])

        fake_no_gpu = types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetCount=lambda: 0,
            nvmlShutdown=lambda: None,
        )
        no_device = gpu_telemetry(fake_no_gpu, torch_module=no_cuda)
        self.assertFalse(no_device["available"])

    def test_torch_fallback_reports_global_vram_not_torch_allocator_usage(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "NVIDIA RTX Test",
            mem_get_info=lambda: (8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024),
        ))
        with patch.dict(sys.modules, {"pynvml": None}):
            result = gpu_telemetry(torch_module=fake_torch)
        self.assertEqual(result["provider"], "torch_fallback")
        self.assertTrue(result["available"])
        self.assertEqual(result["scope"], "device_global")
        self.assertFalse(result["backend_attributed"])
        self.assertEqual(result["vram_used_mib"], 16_384.0)
        self.assertEqual(result["vram_total_mib"], 24_576.0)
        self.assertIsNone(result["utilization_pct"])
        self.assertNotIn("process_allocated_mib", result)

    def test_compute_gpu_status_uses_health_snapshot_independent_of_nvml(self):
        broker = types.SimpleNamespace(ready=True, worker_metrics=[{
            "vram_after_warmup": {"available": True, "device": "NVIDIA RTX 3090"},
        }])
        settings = types.SimpleNamespace(backend="faster_whisper")
        self.assertEqual(gpu_compute_state(broker, settings), {
            "available": True, "cuda": True, "device": "NVIDIA RTX 3090",
            "backend": "faster_whisper", "model_loaded": True,
        })

    def test_collected_schema_is_allowlisted_and_excludes_secrets(self):
        broker = types.SimpleNamespace(ready=True, queued_jobs=2)
        settings = types.SimpleNamespace(
            backend="parakeet", model_id="nvidia/parakeet-tdt-0.6b-v3",
            model_revision=None, workers=4, api_token=SECRET, hf_token=SECRET,
        )
        with patch("asr_lab.telemetry.process_rss_mib", return_value=100.0), \
             patch("asr_lab.telemetry.system_memory_mib", return_value={"total_mib": 1000.0, "used_mib": 500.0}), \
             patch("asr_lab.telemetry.gpu_compute_state", return_value={"available": True, "cuda": True, "device": "NVIDIA RTX Test"}), \
             patch("asr_lab.telemetry.gpu_telemetry", return_value={"available": False, "device": None}):
            snapshot = collect_telemetry(broker, settings)

        self.assertEqual(set(snapshot), {
            "schema_version", "timestamp", "backend", "production_backend", "model_id", "model_revision",
            "model_loaded", "ready", "workers", "queue_depth", "process_rss_mib",
            "system_ram", "gpu_compute", "gpu_telemetry", "gpu",
        })
        self.assertTrue(snapshot["model_loaded"])
        self.assertTrue(snapshot["gpu_compute"]["available"])
        self.assertEqual(snapshot["gpu"]["device"], "NVIDIA RTX Test")
        self.assertEqual(snapshot["queue_depth"], 2)
        self.assertNotIn(SECRET, json.dumps(snapshot))


if __name__ == "__main__":
    unittest.main()
