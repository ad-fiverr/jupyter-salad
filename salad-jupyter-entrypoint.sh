#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${JUPYTER_PASSWORD:-}" ]]; then
  printf '%s\n' 'JUPYTER_PASSWORD must be set to a non-empty value.' >&2
  exit 64
fi

mkdir -p /workspace "${JUPYTER_CONFIG_DIR:-/tmp/jupyter-config}" "${JUPYTER_RUNTIME_DIR:-/tmp/jupyter-runtime}"
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
        "c.ServerApp.ip = '::'",
        "c.ServerApp.port = 8888",
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

# Do not pass the plaintext runtime secret into the Jupyter server or kernels.
unset JUPYTER_PASSWORD

/usr/local/bin/salad-jupyter-verify-runtime
printf '%s\n' 'Starting password-protected JupyterLab on [::]:8888 with /workspace as its root.'
exec jupyter lab --config="$config_file"
