#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.metadata
import platform
import sys

import ipykernel
import ipywidgets
import jupyter_server
import jupyter_server_terminals
import jupyterlab
import jupyterlab_widgets
import notebook
import torch

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
assert torch.__version__.split("+", 1)[0] == "2.4.0", f"Unexpected PyTorch {torch.__version__}"
assert torch.version.cuda and torch.version.cuda.startswith("12.4"), (
    f"Expected PyTorch CUDA 12.4 runtime, got {torch.version.cuda!r}"
)

for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    assert actual == wanted, f"Expected {package}=={wanted}, got {actual}"

assert ipywidgets.IntSlider is not None
assert jupyter_server.ServerApp is not None
assert jupyterlab.__version__ == expected["jupyterlab"]
assert notebook.__version__ == expected["notebook"]
assert jupyter_server_terminals.__name__ == "jupyter_server_terminals"
assert jupyterlab_widgets.__name__ == "jupyterlab_widgets"
assert ipykernel.__version__ == expected["ipykernel"]

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

jupyter server extension list 2>&1 | grep -Eq 'jupyter_server_terminals.*enabled'
printf '%s\n' 'Jupyter server terminal extension: enabled'
