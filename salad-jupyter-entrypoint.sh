#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${JUPYTER_PASSWORD:-}" ]]; then
  printf '%s\n' 'JUPYTER_PASSWORD must be set to a non-empty value.' >&2
  exit 64
fi

mkdir -p /workspace "${JUPYTER_CONFIG_DIR:-/tmp/jupyter-config}" "${JUPYTER_RUNTIME_DIR:-/tmp/jupyter-runtime}" "${HF_HOME:-/workspace/.cache/huggingface}"
chmod 0700 "${JUPYTER_CONFIG_DIR:-/tmp/jupyter-config}" "${JUPYTER_RUNTIME_DIR:-/tmp/jupyter-runtime}"
umask 077

config_file="${JUPYTER_CONFIG_DIR:-/tmp/jupyter-config}/jupyter_server_config.py"
python - "$config_file" <<'PY'
import os
import sys
from pathlib import Path

from jupyter_server.auth import passwd

password = os.environ.pop("JUPYTER_PASSWORD", "")
if not password:
    raise SystemExit("JUPYTER_PASSWORD must be set to a non-empty value.")

hashed_password = passwd(password)
config = "\n".join(
    (
        "c = get_config()",
        "c.ServerApp.ip = '127.0.0.1'",
        "c.ServerApp.port = 8889",
        "c.ServerApp.root_dir = '/workspace'",
        "c.ServerApp.allow_unauthenticated_access = False",
        "c.ServerApp.port_retries = 0",
        "c.ServerApp.allow_remote_access = True",
        "c.ServerApp.trust_xheaders = True",
        "c.ServerApp.open_browser = False",
        "c.ServerApp.allow_root = True",
        "c.ServerApp.terminals_enabled = True",
        "c.ServerApp.websocket_ping_interval = 30",
        "c.ServerApp.identity_provider_class = 'jupyter_server.auth.identity.PasswordIdentityProvider'",
        "c.IdentityProvider.token = ''",
        "c.PasswordIdentityProvider.hashed_password = " + repr(hashed_password),
        "c.PasswordIdentityProvider.password_required = True",
        "c.PasswordIdentityProvider.allow_password_change = False",
        "",
    )
)
path = Path(sys.argv[1])
path.write_text(config, encoding="utf-8")
path.chmod(0o600)
PY

unset JUPYTER_PASSWORD
case "${SALAD_FULL_RUNTIME_VERIFY:-0}" in
  0)
    /usr/local/bin/salad-jupyter-verify-runtime startup
    ;;
  1)
    printf '%s\n' 'Running the optional full runtime verifier before starting services.'
    /usr/local/bin/salad-jupyter-verify-runtime full
    ;;
  *)
    printf '%s\n' 'SALAD_FULL_RUNTIME_VERIFY must be 0 or 1.' >&2
    exit 64
    ;;
esac

printf '%s\n' 'Starting selected ASR backend on 127.0.0.1:8765, JupyterLab on 127.0.0.1:8889, and the IPv6 gateway proxy on [::]:8888.'
/opt/asr-venv/bin/uvicorn asr_lab.service:app \
  --host 127.0.0.1 \
  --port 8765 \
  --no-access-log \
  --log-level info \
  --ws-ping-interval 30 \
  --ws-ping-timeout 20 \
  --timeout-graceful-shutdown 30 &
asr_pid=$!

# Keep model-download credentials available to ASR only. The shell, Jupyter, and
# Nginx lose them before either process starts; ASR retains its private copy.
unset ASR_API_TOKEN HF_TOKEN
jupyter lab --config="$config_file" --ip=127.0.0.1 --port=8889 &
jupyter_pid=$!
nginx -g 'daemon off;' &
nginx_pid=$!

source /usr/local/bin/salad-supervisor-watch
if salad_supervise_children "$asr_pid" "$jupyter_pid" "$nginx_pid"; then
  exit 0
else
  supervisor_status=$?
  printf 'A required container service exited unexpectedly (status %s).\n' "$supervisor_status" >&2
  exit "$supervisor_status"
fi
