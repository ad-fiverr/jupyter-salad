#!/usr/bin/env bash
set -euo pipefail

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
