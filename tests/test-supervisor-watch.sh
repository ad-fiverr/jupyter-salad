#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$script_dir/salad-supervisor-watch.sh"

killed=()
reaped=()
wait() {
  if [[ "${1:-}" == "-n" ]]; then
    if [[ "${SUPERVISOR_TEST_SIGNAL:-}" == "TERM" ]]; then
      _salad_mark_shutdown
      return 143
    fi
    return "${SUPERVISOR_TEST_CHILD_STATUS:-0}"
  fi
  reaped+=("$1")
  return 0
}
kill() {
  [[ "${1:-}" == "-TERM" ]] || return 1
  killed+=("${@:2}")
}

# A clean child exit before shutdown is an unexpected service exit.
SUPERVISOR_TEST_CHILD_STATUS=0
unset SUPERVISOR_TEST_SIGNAL
if salad_supervise_children 101 202; then
  printf '%s\n' 'Unexpected child exit was incorrectly treated as success.' >&2
  exit 1
else
  status=$?
fi
[[ "$status" -eq 1 ]]
[[ "${killed[*]}" == '101 202' ]]
[[ "${reaped[*]}" == '101 202' ]]

# A TERM-triggered wait interruption is treated as an intentional shutdown.
killed=()
reaped=()
SUPERVISOR_TEST_SIGNAL=TERM
if salad_supervise_children 303 404; then
  status=0
else
  status=$?
fi
[[ "$status" -eq 0 ]]
[[ "${killed[*]}" == '303 404' ]]
[[ "${reaped[*]}" == '303 404' ]]

printf '%s\n' 'Supervisor status mapping and cleanup checks: PASS'
