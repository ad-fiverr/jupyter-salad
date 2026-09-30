#!/usr/bin/env bash
set -euo pipefail

salad_supervise_children() {
  if (($# < 2)); then
    printf '%s\n' 'salad_supervise_children requires at least two child process IDs.' >&2
    return 64
  fi

  local requested_shutdown=0
  local child_status=0
  local pid
  local -a child_pids=("$@")

  _salad_mark_shutdown() {
    requested_shutdown=1
  }
  trap _salad_mark_shutdown TERM INT

  if wait -n "${child_pids[@]}"; then
    child_status=0
  else
    child_status=$?
  fi

  if ((requested_shutdown)); then
    child_status=0
  elif ((child_status == 0)); then
    # A required child must not silently exit successfully while the container
    # is expected to keep serving. Report an unexpected clean exit as failure.
    child_status=1
  fi

  trap - TERM INT
  for pid in "${child_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  return "$child_status"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  salad_supervise_children "$@"
fi
