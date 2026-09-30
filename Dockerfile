# syntax=docker/dockerfile:1

# Pinned to the exact RunPod PyTorch 2.4 / Python 3.11 / CUDA 12.4.1 image.
# GitHub Actions selects linux/amd64 through Buildx; keep FROM platform selection implicit.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04@sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_CONFIG_DIR=/tmp/jupyter-config \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime \
    PYTHONPATH=/opt/asr-lab \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default

COPY requirements-jupyter.txt /tmp/requirements-jupyter.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      -r /tmp/requirements-jupyter.txt

COPY asr-lab/requirements-asr.txt /tmp/asr-requirements.txt
COPY asr-lab/constraints-asr.txt /tmp/asr-constraints.txt
RUN python -m venv --system-site-packages /opt/asr-venv \
    && /opt/asr-venv/bin/python -m pip install --no-cache-dir --disable-pip-version-check \
      --constraint /tmp/asr-constraints.txt \
      --requirement /tmp/asr-requirements.txt

COPY asr-lab /opt/asr-lab
COPY nginx-main.conf /etc/nginx/nginx.conf
COPY nginx-salad.conf /etc/nginx/conf.d/default.conf
COPY salad_healthcheck.py /usr/local/bin/salad-healthcheck.py
COPY salad_nginx_diagnostics.py /usr/local/bin/salad-nginx-diagnostics.py
COPY salad-jupyter-entrypoint.sh /usr/local/bin/salad-jupyter-entrypoint
COPY salad-supervisor-watch.sh /usr/local/bin/salad-supervisor-watch
COPY verify-runtime.sh /usr/local/bin/salad-jupyter-verify-runtime
RUN chmod 0755 \
      /usr/local/bin/salad-jupyter-entrypoint \
      /usr/local/bin/salad-supervisor-watch \
      /usr/local/bin/salad-jupyter-verify-runtime \
    && nginx -t \
    && python /usr/local/bin/salad-nginx-diagnostics.py contract \
    && /usr/local/bin/salad-jupyter-verify-runtime \
    && nvcc --version | grep -Eq 'release 12\.4([,.]|$)'

# Fail the image build if the benchmark UI/worklet/tests are omitted from the
# image. Unit tests use the CI-safe fakes and never download or load model weights.
RUN test -s /opt/asr-lab/asr_lab/benchmark_web/index.html \
    && test -s /opt/asr-lab/asr_lab/benchmark_web/app.mjs \
    && test -s /opt/asr-lab/asr_lab/benchmark_web/audio-worklet.mjs \
    && test -s /opt/asr-lab/asr_lab/benchmark_web/core.mjs \
    && test -s /opt/asr-lab/asr_lab/benchmark_web/style.css \
    && PYTHONPATH=/opt/asr-lab /opt/asr-venv/bin/python -m unittest discover \
      -s /opt/asr-lab/tests -v

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "/usr/local/bin/salad-healthcheck.py"]

# Preserve NVIDIA's inherited ENTRYPOINT and replace only CMD.
CMD ["/usr/local/bin/salad-jupyter-entrypoint"]
