#!/usr/bin/env bash
set -euo pipefail

verify_mode="${1:-full}"
if (($# > 1)); then
  printf 'Usage: %s [full|startup]\n' "$0" >&2
  exit 64
fi

case "$verify_mode" in
  full)
    ;;
  startup)
    # The image already ran the complete verifier during docker build. Keep the
    # normal container startup path to cheap metadata reads; importing NeMo and
    # Jupyter here can delay all listening services on every replica restart.
    /opt/asr-venv/bin/python - <<'PY'
import importlib.metadata
import sys

assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"
startup_expected = {
    "torch": "2.4.1+cu124",
    "nemo-toolkit": "2.4.0",
    "jupyterlab": "4.6.4",
    "jupyter_server": "2.21.1",
}
for package, wanted in startup_expected.items():
    actual = importlib.metadata.version(package)
    assert actual == wanted, f"Expected startup metadata {package}=={wanted}, got {actual}"
print("RUNTIME_VERIFIER_MODE=STARTUP_METADATA_ONLY")
print("PARAKEET_RUNTIME=NEMO")
PY
    exit 0
    ;;
  *)
    printf 'Unknown verifier mode: %s (expected full or startup).\n' "$verify_mode" >&2
    exit 64
    ;;
esac

python - <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import ipykernel
import ipywidgets
import jupyter_server_terminals
import jupyterlab
import jupyterlab_widgets
import notebook
import torch
from jupyter_server.serverapp import ServerApp

expected = {
    "jupyterlab": "4.6.4",
    "notebook": "7.6.3",
    "jupyter_server": "2.21.1",
    "ipywidgets": "8.1.9",
    "jupyterlab_widgets": "3.0.17",
    "jupyter_server_terminals": "0.5.4",
    "ipykernel": "7.3.0",
}

assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {platform.python_version()}"
assert torch.__version__.split("+", 1)[0] == "2.4.1", f"Unexpected PyTorch {torch.__version__}"
assert torch.version.cuda and torch.version.cuda.startswith("12.4"), (
    f"Expected PyTorch CUDA 12.4 runtime, got {torch.version.cuda!r}"
)

for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    assert actual == wanted, f"Expected {package}=={wanted}, got {actual}"

assert ipywidgets.IntSlider is not None
assert ServerApp is not None
assert jupyterlab.__version__ == expected["jupyterlab"]
assert notebook.__version__ == expected["notebook"]
assert jupyter_server_terminals.__name__ == "jupyter_server_terminals"
assert jupyterlab_widgets.__name__ == "jupyterlab_widgets"
assert ipykernel.__version__ == expected["ipykernel"]

terminals_distribution = importlib.metadata.distribution("jupyter_server_terminals")
terminals_config_files = [
    Path(terminals_distribution.locate_file(file)).resolve()
    for file in terminals_distribution.files or ()
    if file.name == "jupyter_server_terminals.json"
    and "jupyter_server_config.d" in Path(str(file)).parts
]
assert len(terminals_config_files) == 1, (
    "Expected jupyter_server_terminals to install one Jupyter server config fragment; "
    f"found {terminals_config_files}"
)
terminals_config = json.loads(terminals_config_files[0].read_text(encoding="utf-8"))
assert (
    terminals_config.get("ServerApp", {})
    .get("jpserver_extensions", {})
    .get("jupyter_server_terminals")
    is True
), f"Jupyter auto-enable config is invalid: {terminals_config_files[0]}"

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
for package, version in expected.items():
    print(f"{package}: {version}")

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print("GPU_RUNTIME_TEST=UNRUN (device detection is not a training test)")
else:
    print("GPU_RUNTIME_TEST=UNRUN (validate CUDA execution on a Salad GPU)")
PY

python - <<'PY'
import re
import subprocess

result = subprocess.run(
    ["jupyter", "server", "extension", "list"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
output = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", result.stdout).replace("\r", "")
print(output, end="")
if result.returncode != 0:
    raise SystemExit(f"jupyter server extension list exited with {result.returncode}")
if "Validation failed" in output:
    raise SystemExit("Jupyter Server reported an extension validation failure")
lines = [line.split() for line in output.splitlines()]
if not any(line[:2] == ["jupyter_server_terminals", "enabled"] for line in lines):
    raise SystemExit("jupyter_server_terminals was not discovered as enabled")
if not any(
    line and line[0] == "jupyter_server_terminals" and line[-1] == "OK"
    for line in lines
):
    raise SystemExit("Jupyter Server did not validate jupyter_server_terminals as OK")
print("Jupyter Server discovered and validated jupyter_server_terminals: OK")
PY

/opt/asr-venv/bin/python - <<'PY'
import importlib.metadata
import sys

expected = {
    "fastapi": "0.142.1",
    "uvicorn": "0.54.0",
    "websockets": "17.1",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.2",
    "numpy": "1.26.4",
    "librosa": "0.11.0",
    "nemo-toolkit": "2.4.0",
    "cuda-python": "12.3.0",
}
for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    assert actual == wanted, f"Expected ASR {package}=={wanted}, got {actual}"

import torch
assert torch.__version__ == "2.4.1+cu124", f"ASR environment changed base Torch: {torch.__version__}"
assert torch.version.cuda and torch.version.cuda.startswith("12.4"), (
    f"ASR environment changed base CUDA runtime: {torch.version.cuda!r}"
)

from packaging.markers import default_environment
from packaging.requirements import Requirement

nemo_distribution = importlib.metadata.distribution("nemo-toolkit")
marker_environment = {**default_environment(), "extra": "asr"}
checked_nemo_requirements = 0
for raw_requirement in nemo_distribution.requires or ():
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate(marker_environment):
        continue
    actual = importlib.metadata.version(requirement.name)
    if requirement.specifier and not requirement.specifier.contains(actual, prereleases=True):
        raise AssertionError(
            f"Installed NeMo ASR dependency {requirement.name}=={actual} does not satisfy "
            f"{requirement.specifier} from {raw_requirement}"
        )
    checked_nemo_requirements += 1

import nemo.collections.asr as nemo_asr
assert nemo_asr.models.ASRModel is not None

from asr_lab.config import MODEL_IDS
assert MODEL_IDS["parakeet"] == "nvidia/parakeet-tdt-0.6b-v3"

from asr_lab.service import app
assert app.title == "Salad ASR Lab"
assert "asr_lab.backends.parakeet" not in sys.modules
assert "asr_lab.backends.faster_whisper" not in sys.modules
print("ASR environment: pinned direct versions, NeMo ASR dependencies, inherited Torch, and lazy backend selection: OK")
print("GPU telemetry: optional NVML binding (GPU device sampling still requires Salad runtime)")
print(f"NeMo ASR metadata dependencies checked: {checked_nemo_requirements}")
print("VERIFY_RUNTIME_BASELINE=NEMO")
print("PARAKEET_RUNTIME=NEMO")
print("PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3")
if tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]) < (2, 5):
    print("NEMO_TORCH_CONFIGURATION=OUTSIDE_DOCUMENTED_SUPPORT (NeMo 2.4 documents PyTorch >=2.5)")
print("PARAKEET_NEMO_GPU_RUNTIME=UNRUN (model download, CUDA inference, and transcription require Salad GPU validation)")
PY
