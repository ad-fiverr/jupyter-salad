# syntax=docker/dockerfile:1

# Pinned to the exact RunPod PyTorch 2.4 / Python 3.11 / CUDA 12.4.1 image.
# GitHub Actions selects linux/amd64 through Buildx; keep FROM platform selection implicit.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04@sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_CONFIG_DIR=/tmp/jupyter-config \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime

WORKDIR /workspace

COPY requirements-jupyter.txt /tmp/requirements-jupyter.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      -r /tmp/requirements-jupyter.txt

RUN jupyter server extension enable --py jupyter_server_terminals --sys-prefix

COPY salad-jupyter-entrypoint.sh /usr/local/bin/salad-jupyter-entrypoint
COPY verify-runtime.sh /usr/local/bin/salad-jupyter-verify-runtime

RUN chmod 0755 \
      /usr/local/bin/salad-jupyter-entrypoint \
      /usr/local/bin/salad-jupyter-verify-runtime \
    && /usr/local/bin/salad-jupyter-verify-runtime \
    && nvcc --version | grep -Eq 'release 12\.4([,.]|$)'

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://[::1]:8888/login', timeout=3)"]

# Preserve the NVIDIA entrypoint inherited from the base image. Replacing CMD
# bypasses RunPod's /start.sh while retaining NVIDIA runtime initialization.
CMD ["/usr/local/bin/salad-jupyter-entrypoint"]
