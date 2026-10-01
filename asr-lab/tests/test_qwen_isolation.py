import unittest
from types import SimpleNamespace

from asr_lab.qwen_process import qwen_worker_environment
from asr_lab.qwen_worker import build_experiment_config, runtime_provenance


class QwenIsolationTests(unittest.TestCase):
    def test_worker_environment_excludes_service_secrets(self):
        source = {
            "PATH": "/usr/bin",
            "HF_TOKEN": "hf-secret-for-child-only",
            "CUDA_VISIBLE_DEVICES": "0",
            "ASR_API_TOKEN": "service-secret",
            "JUPYTER_PASSWORD": "jupyter-secret",
            "SALAD_GATEWAY_URL": "https://private.example",
            "SALAD_API_KEY": "salad-secret",
        }
        passed = qwen_worker_environment(source)
        self.assertEqual(passed["PATH"], "/usr/bin")
        self.assertEqual(passed["HF_TOKEN"], "hf-secret-for-child-only")
        self.assertEqual(passed["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("ASR_API_TOKEN", passed)
        self.assertNotIn("JUPYTER_PASSWORD", passed)
        self.assertFalse(any(key.startswith("SALAD_") for key in passed))

    def test_runtime_provenance_uses_installed_versions_and_safe_experiment_settings(self):
        versions = {
            "qwen-asr": "0.0.6",
            "vllm": "0.14.0",
            "transformers": "4.57.1",
        }
        torch = SimpleNamespace(__version__="2.9.1+cu129", version=SimpleNamespace(cuda="12.9"))
        experiment = build_experiment_config(
            gpu_memory_utilization=0.65,
            max_active_sessions=2,
            warmup_chunk_ms=1000,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        report = runtime_provenance(torch, versions.__getitem__, experiment)
        self.assertEqual(report["qwen_asr_version"], "0.0.6")
        self.assertEqual(report["vllm_version"], "0.14.0")
        self.assertEqual(report["transformers_version"], "4.57.1")
        self.assertEqual(report["torch_version"], "2.9.1+cu129")
        self.assertEqual(report["torch_cuda_version"], "12.9")
        self.assertEqual(report["experiment_config"], experiment)
        self.assertEqual(experiment["max_new_tokens"], 128)
        self.assertEqual(experiment["max_model_len"], 4096)
        self.assertNotIn("HF_TOKEN", str(report))

    def test_missing_distribution_metadata_is_reported_as_unknown(self):
        torch = SimpleNamespace(__version__="2.9.1", version=SimpleNamespace(cuda=None))
        report = runtime_provenance(torch, lambda _: (_ for _ in ()).throw(LookupError()))
        self.assertIsNone(report["qwen_asr_version"])
        self.assertIsNone(report["vllm_version"])
        self.assertIsNone(report["transformers_version"])
        self.assertIsNone(report["torch_cuda_version"])


if __name__ == "__main__":
    unittest.main()
