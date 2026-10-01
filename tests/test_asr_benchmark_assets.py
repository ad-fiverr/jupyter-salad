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

    def test_real_fastapi_regression_is_in_the_image_unit_suite(self):
        regression = (ROOT / "asr-lab/tests/test_fastapi_real_import.py").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('from asr_lab.service import app', regression)
        self.assertIn('version("fastapi") == "0.142.1"', regression)
        self.assertIn('get("/asr/benchmark")', regression)
        self.assertIn('get("/asr/benchmark/app.mjs")', regression)
        self.assertIn('get("/asr/benchmark/not-allowlisted")', regression)
        self.assertIn("-s /opt/asr-lab/tests -v", dockerfile)


if __name__ == "__main__":
    unittest.main()
