#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TAILSCALE_AUTH_KEY:-}" ]]; then
  printf '%s\n' 'TAILSCALE_AUTH_KEY must be provided as a Salad secret.' >&2
  exit 64
fi

if [[ -z "${SALAD_MACHINE_ID:-}" ]]; then
  printf '%s\n' 'SALAD_MACHINE_ID must be provided by Salad.' >&2
  exit 64
fi

umask 077
socket=/run/tailscale/tailscaled.sock
mkdir -p /run/tailscale

# Do not pass the Salad secret to the long-running daemon or any ASR service.
env -u TAILSCALE_AUTH_KEY /usr/local/bin/tailscaled \
  --tun=userspace-networking \
  --state=mem: \
  --socket="$socket" \
  --socks5-server=127.0.0.1:1055 \
  --outbound-http-proxy-listen=127.0.0.1:1055 \
  >/dev/null 2>&1 &



# The auth key is passed only to the one-shot login client, never logged or
# written to disk. The daemon's state is memory-only and the key is removed
# from the environment before the original application starts.
env -u TAILSCALE_AUTH_KEY /usr/local/bin/tailscale --socket="$socket" up \
  --auth-key="$TAILSCALE_AUTH_KEY" \
  --hostname="$SALAD_MACHINE_ID" \
  --accept-dns=false \
  --accept-routes=false \
  --ssh=false
unset TAILSCALE_AUTH_KEY

tailscale_ip=$(/usr/local/bin/tailscale --socket="$socket" ip -4)
printf 'Private Tailscale node ready: hostname=%s ipv4=%s\n' "$SALAD_MACHINE_ID" "$tailscale_ip"

exec /usr/local/bin/salad-jupyter-entrypoint
