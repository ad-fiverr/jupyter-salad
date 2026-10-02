from __future__ import annotations

import unittest
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


class AsrBenchmarkImageContractTests(unittest.TestCase):
    def test_dockerfile_copies_and_requires_every_benchmark_asset(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY asr-lab /opt/asr-lab", dockerfile)
        for asset in ("index.html", "app.mjs", "audio-worklet.mjs", "core.mjs", "style.css"):
            self.assertIn(f"/opt/asr-lab/asr_lab/benchmark_web/{asset}", dockerfile)
            self.assertTrue((ROOT / "asr-lab/asr_lab/benchmark_web" / asset).is_file())
        self.assertIn("unittest discover", dockerfile)

    def test_workflow_runs_browser_core_checks_for_asr_changes(self):
        workflow = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
        self.assertIn("'asr-lab/**'", workflow)
        self.assertIn("node --test asr-lab/tests/core.test.mjs", workflow)
        self.assertIn("node --check asr-lab/asr_lab/benchmark_web/app.mjs", workflow)

    def test_nvml_remains_optional_and_periodic_chart_render_is_single_path(self):
        requirements = (ROOT / "asr-lab/requirements-asr.txt").read_text(encoding="utf-8")
        verifier = (ROOT / "verify-runtime.sh").read_text(encoding="utf-8")
        telemetry = (ROOT / "asr-lab/asr_lab/telemetry.py").read_text(encoding="utf-8")
        app = (ROOT / "asr-lab/asr_lab/benchmark_web/app.mjs").read_text(encoding="utf-8")

        self.assertNotIn("nvidia-ml-py", requirements)
        self.assertNotIn('"nvidia-ml-py"', verifier)
        self.assertIn("import pynvml as nvml", telemetry)
        self.assertIn("except Exception", telemetry)

        interval = re.search(r"state\.timer\s*=\s*setInterval\((.*?),\s*1000\)", app)
        self.assertIsNotNone(interval)
        self.assertNotIn("renderCharts()", interval.group(1))
        poll = re.search(r"async function pollTelemetry\(\) \{(.*?)\n\}", app, re.S)
        self.assertIsNotNone(poll)
        self.assertEqual(poll.group(1).count("renderCharts()"), 1)

    def test_live_latency_groups_rtt_control_and_gpu_compute_provenance_are_present(self):
        telemetry = (ROOT / "asr-lab/asr_lab/telemetry.py").read_text(encoding="utf-8")
        service = (ROOT / "asr-lab/asr_lab/service.py").read_text(encoding="utf-8")
        core = (ROOT / "asr-lab/asr_lab/benchmark_web/core.mjs").read_text(encoding="utf-8")
        app = (ROOT / "asr-lab/asr_lab/benchmark_web/app.mjs").read_text(encoding="utf-8")
        page = (ROOT / "asr-lab/asr_lab/benchmark_web/index.html").read_text(encoding="utf-8")

        for metric in (
            "SERVER_ENDPOINTING_MS", "SERVER_QUEUE_WAIT_MS", "SERVER_MODEL_INFERENCE_MS",
            "SERVER_POSTPROCESS_MS", "SERVER_EOS_TO_TRANSCRIPT_MS", "CLIENT_EOS_TO_TRANSCRIPT_MS",
        ):
            self.assertIn(metric, app)
        for label in (
            "Audio PCM ms", "Endpointing server ms", "Queue ms", "Model adapter ms",
            "Postprocess ms", "Server EOS", "Client EOS",
        ):
            self.assertIn(label, page)
        self.assertIn("isinstance(message, BenchmarkPing)", service)
        self.assertIn('"event": "benchmark_pong"', service)
        self.assertIn('event: "benchmark_ping"', app)
        self.assertIn("formatMilliseconds", app)
        self.assertIn("PROXY_WS_RTT_MS", core)
        self.assertIn('"gpu_compute"', telemetry)
        self.assertIn('"gpu_telemetry"', telemetry)
        self.assertIn('"provider": "torch_fallback"', telemetry)
        self.assertIn('"backend_attributed": False', telemetry)
        self.assertNotIn("GPU no disponible", app)

    def test_real_fastapi_regression_is_in_the_image_unit_suite(self):
        regression = (ROOT / "asr-lab/tests/test_fastapi_real_import.py").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('from asr_lab.service import app', regression)
        self.assertIn('version("fastapi") == "0.142.1"', regression)
        self.assertIn('get("/asr/benchmark")', regression)
        self.assertIn('get("/asr/benchmark/app.mjs")', regression)
        self.assertIn('get("/asr/benchmark/not-allowlisted")', regression)
        self.assertIn("-s /opt/asr-lab/tests -v", dockerfile)


class QwenBuildArtifactContractTests(unittest.TestCase):
    def test_ci_build_enables_qwen_and_is_the_image_smoked_then_published(self):
        workflow = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
        build_steps = re.findall(
            r"(?ms)^      - name: Build linux/amd64 image for runner smoke tests\n"
            r"(?P<body>.*?)(?=^      - name: |\Z)",
            workflow,
        )
        self.assertEqual(len(build_steps), 1)
        build_step = build_steps[0]
        self.assertIn("uses: docker/build-push-action@", build_step)
        self.assertIn("tags: myblockchaincompany/jupyter-salad:ci", build_step)
        self.assertIn("build-args: |\n            INSTALL_QWEN_RUNTIME=1", build_step)
        self.assertIn("bash ci-smoke-test.sh myblockchaincompany/jupyter-salad:ci", workflow)
        self.assertEqual(workflow.count('docker tag "${image}:ci"'), 3)
        for published_tag in (
            "2.4.1-py3.11-cuda12.4.1", "latest", "sha-${SHORT_SHA}"
        ):
            self.assertIn(f'docker tag "${{image}}:ci" "${{image}}:{published_tag}"', workflow)
        self.assertLess(
            workflow.index("Test built image before publication"),
            workflow.index("Log in to Docker Hub after validation"),
        )
        self.assertLess(
            workflow.index("Log in to Docker Hub after validation"),
            workflow.index("Tag and publish the tested image"),
        )

    def test_dockerfile_and_smoke_verify_the_isolated_qwen_image_contract_without_weights(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        requirements = (ROOT / "asr-lab/requirements-qwen.txt").read_text(encoding="utf-8")
        verifier = (ROOT / "asr-lab/asr_lab/verify_qwen_runtime.py").read_text(encoding="utf-8")
        smoke = (ROOT / "ci-smoke-test.sh").read_text(encoding="utf-8")

        self.assertIn("ARG INSTALL_QWEN_RUNTIME=0", dockerfile)
        self.assertIn("QWEN_STREAMING_RUNTIME_AVAILABLE=${INSTALL_QWEN_RUNTIME}", dockerfile)
        self.assertIn('if [ "$INSTALL_QWEN_RUNTIME" = "1" ]', dockerfile)
        self.assertIn("python -m venv /opt/qwen-asr-venv", dockerfile)
        self.assertIn("--requirement /tmp/qwen-requirements.txt", dockerfile)
        self.assertIn("/opt/qwen-asr-venv/bin/python -m asr_lab.verify_qwen_runtime", dockerfile)
        self.assertIn("qwen-asr[vllm]==0.0.6", requirements)
        self.assertIn("vllm==0.14.0", requirements)
        self.assertIn('"qwen-asr": "0.0.6"', verifier)
        self.assertIn('"vllm": "0.14.0"', verifier)
        self.assertIn("from qwen_asr import Qwen3ASRModel", verifier)
        self.assertIn("import vllm", verifier)
        self.assertIn("weights=NOT_LOADED", verifier)

        self.assertIn("set_phase qwen_build_runtime_test", smoke)
        self.assertRegex(
            smoke,
            r"timeout 15s docker exec --interactive \"\$container_name\" \\\
\s*/opt/qwen-asr-venv/bin/python - <<'PY'",
        )
        self.assertIn('os.environ.get("QWEN_STREAMING_RUNTIME_AVAILABLE") == "1"', smoke)
        fake_smoke_test = (ROOT / "tests/test-ci-smoke-diagnostics.sh").read_text(encoding="utf-8")
        self.assertIn('qwen_script="$(cat)"', fake_smoke_test)
        self.assertIn('!= *" --interactive "*', fake_smoke_test)
        self.assertIn('version("qwen-asr") == "0.0.6"', fake_smoke_test)
        self.assertIn('version("vllm") == "0.14.0"', fake_smoke_test)
        self.assertIn('version("qwen-asr") == "0.0.6"', smoke)
        self.assertIn('version("vllm") == "0.14.0"', smoke)
        self.assertIn("QWEN_BUILD_RUNTIME_PRESENT=PASS", smoke)
        self.assertIn("weights=NOT_LOADED", smoke)


    def test_qwen_chunk_default_is_not_synthesized_by_entrypoint_or_docker(self):
        entrypoint = (ROOT / "salad-jupyter-entrypoint.sh").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotRegex(entrypoint, r"QWEN_(?:MODEL|STREAM)_CHUNK_MS")
        self.assertNotRegex(dockerfile, r"ENV\s+QWEN_(?:MODEL|STREAM)_CHUNK_MS")
        self.assertNotIn("QWEN_MODEL_CHUNK_MS=${QWEN_STREAM_CHUNK_MS", entrypoint)

    def test_browser_json_export_keeps_safe_qwen_startup_provenance_without_candidates(self):
        core = (ROOT / "asr-lab/asr_lab/benchmark_web/core.mjs").read_text(encoding="utf-8")
        app = (ROOT / "asr-lab/asr_lab/benchmark_web/app.mjs").read_text(encoding="utf-8")
        self.assertIn("export function safeQwenStartupEvidence", core)
        self.assertIn("startup_evidence: safeQwenStartupEvidence({", app)
        for field in (
            "readinessAtStart: state.readinessAtStart", "qwenStartMetrics: state.qwenStartMetrics",
            "health: state.health", "model_load_ms", "warmup_ms", "warmup_chunk_ms",
            "runtime_provenance", "FIRST_STREAM_INIT_MS", "FIRST_STREAM_STATE_INIT_WALL_MS",
            "FIRST_STREAM_INIT_RPC_OVERHEAD_MS",
        ):
            self.assertIn(field, core + app)
        safe_result = app[app.index("function safeResult()") : app.index("function download(")]
        self.assertIn("startup_evidence: safeQwenStartupEvidence", safe_result)
        self.assertLess(safe_result.index("startup_evidence:"), safe_result.index("streaming:"))


if __name__ == "__main__":
    unittest.main()
