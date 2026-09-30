#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_directory="$(mktemp -d)"
fake_bin="$temporary_directory/bin"
mkdir -p "$fake_bin"
trap 'rm -rf "$temporary_directory"' EXIT

cat >"$fake_bin/docker" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

case "$1" in
  run)
    if [[ " $* " == *" --detach "* ]]; then
      printf '%s' "${JUPYTER_PASSWORD:-}" >"$SMOKE_FAKE_DIR/jupyter-secret"
      printf '%s' "${ASR_API_TOKEN:-}" >"$SMOKE_FAKE_DIR/asr-secret"
      exit 0
    fi
    printf '%s\n' 'JUPYTER_PASSWORD must be set to a non-empty value.'
    exit 64
    ;;
  inspect)
    format="${2:-}"
    if [[ "$format" == '--format' ]]; then
      format="${3:-}"
    fi
    case "$format" in
      *'.State.Status'*) printf '%s\n' 'running' ;;
      *'.State.Health.Status'*) printf '%s\n' 'unhealthy' ;;
      *'json .State.Health'*)
        printf '{"Status":"unhealthy","Output":"JUPYTER_PASSWORD=%s ASR_API_TOKEN=%s ?token=%s"}\n' \
          "$(cat "$SMOKE_FAKE_DIR/jupyter-secret")" \
          "$(cat "$SMOKE_FAKE_DIR/asr-secret")" \
          "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
        ;;
      *'json .State'*) printf '%s\n' '{"Status":"running"}' ;;
      *) exit 0 ;;
    esac
    ;;
  logs)
    printf 'JUPYTER_PASSWORD=%s ASR_API_TOKEN=%s ?token=%s HF_TOKEN=fixture-secret\n' \
      "$(cat "$SMOKE_FAKE_DIR/jupyter-secret")" \
      "$(cat "$SMOKE_FAKE_DIR/asr-secret")" \
      "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
    ;;
  top)
    printf 'PID CMD %s %s\n' \
      "$(cat "$SMOKE_FAKE_DIR/jupyter-secret")" \
      "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
    ;;
  rm)
    ;;
  *)
    printf 'Unexpected fake docker command: %s\n' "$*" >&2
    exit 99
    ;;
esac
SH

cat >"$fake_bin/python3" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
counter="$SMOKE_FAKE_DIR/python-count"
if [[ -e "$counter" ]]; then
  printf '%s\n' 'fixture-asr-token'
else
  : >"$counter"
  printf '%s\n' 'fixture-jupyter-password'
fi
SH

cat >"$fake_bin/timeout" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
shift
exec "$@"
SH

cat >"$fake_bin/seq" <<'SH'
#!/usr/bin/env bash
printf '%s\n' '1'
SH

cat >"$fake_bin/sleep" <<'SH'
#!/usr/bin/env bash
exit 0
SH

chmod 0755 "$fake_bin"/*
output_file="$temporary_directory/output.txt"
if PATH="$fake_bin:$PATH" SMOKE_FAKE_DIR="$temporary_directory" \
  "$repo_root/ci-smoke-test.sh" fixture-image >"$output_file" 2>&1; then
  printf '%s\n' 'Smoke diagnostics test expected the fake unhealthy container to fail.' >&2
  exit 1
fi

for expected in \
  'Missing-password check: passed' \
  '[docker inspect health]' \
  '[docker logs]' \
  '[active processes]' \
  'Jupyter did not become healthy (state: unhealthy)' \
  '[REDACTED_JUPYTER_PASSWORD]' \
  '[REDACTED_ASR_TOKEN]' \
  'HF_TOKEN=[REDACTED]'; do
  if ! grep -Fq "$expected" "$output_file"; then
    printf 'Expected diagnostic marker missing: %s\n' "$expected" >&2
    cat "$output_file" >&2
    exit 1
  fi
done

if grep -Eq 'fixture-jupyter-password|fixture-asr-token|fixture-secret' "$output_file"; then
  printf '%s\n' 'A fixture secret was printed in smoke failure diagnostics.' >&2
  exit 1
fi

printf '%s\n' 'CI smoke failure diagnostics and secret redaction: PASS'
