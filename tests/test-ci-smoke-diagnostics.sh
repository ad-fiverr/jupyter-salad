#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_directory="$(mktemp -d)"
fake_bin="$temporary_directory/bin"
mkdir -p "$fake_bin"
SMOKE_FAKE_DIR="$temporary_directory"
export SMOKE_FAKE_DIR
trap 'rm -rf "$temporary_directory"' EXIT

cat >"$fake_bin/docker" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

case "$1" in
  run)
    if [[ " $* " == *" --detach "* ]]; then
      printf '%s' "${JUPYTER_PASSWORD:-}" >"$SMOKE_FAKE_DIR/jupyter-secret"
      printf '%s' "${ASR_API_TOKEN:-}" >"$SMOKE_FAKE_DIR/asr-secret"
      if [[ "${SMOKE_FAKE_MODE:-}" == 'start_failure' ]]; then
        exit 23
      fi
      exit 0
    fi
    printf '%s\n' 'JUPYTER_PASSWORD must be set to a non-empty value.'
    exit 64
    ;;
  inspect)
    format=''
    previous=''
    for argument in "$@"; do
      if [[ "$previous" == '--format' ]]; then
        format="$argument"
        break
      fi
      previous="$argument"
    done
    if [[ "${SMOKE_FAKE_MODE:-}" == 'start_failure' ]]; then
      exit 1
    fi
    case "$format" in
      *'.State.Status'*) printf '%s\n' 'running' ;;
      *'.State.Health.Status'*) printf '%s\n' "${SMOKE_FAKE_HEALTH:-unhealthy}" ;;
      *'json .State.Health'*)
        printf '{"Status":"%s","Output":"JUPYTER_PASSWORD=%s ASR_API_TOKEN=%s ?token=%s"}\n' \
          "${SMOKE_FAKE_HEALTH:-unhealthy}" \
          "$(cat "$SMOKE_FAKE_DIR/jupyter-secret")" \
          "$(cat "$SMOKE_FAKE_DIR/asr-secret")" \
          "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
        ;;
      *'json .State'*) printf '%s\n' '{"Status":"running"}' ;;
      *) printf '%s\n' 'container-id' ;;
    esac
    ;;
  logs)
    if [[ "${SMOKE_FAKE_HEALTH:-}" == 'healthy' ]]; then
      printf '%s\n' 'service logs contain no credentials'
    else
      printf 'JUPYTER_PASSWORD=%s ASR_API_TOKEN=%s ?token=%s HF_TOKEN=fixture-secret\n' \
        "$(cat "$SMOKE_FAKE_DIR/jupyter-secret")" \
        "$(cat "$SMOKE_FAKE_DIR/asr-secret")" \
        "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
    fi
    ;;
  top)
    printf 'PID CMD %s %s\n' \
      "$(cat "$SMOKE_FAKE_DIR/jupyter-secret")" \
      "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
    ;;
  exec)
    if [[ " $* " == *" /opt/qwen-asr-venv/bin/python - "* ]]; then
      if [[ " $* " != *" --interactive "* ]]; then
        printf '%s\n' 'Qwen image probe did not keep STDIN open with --interactive.' >&2
        exit 96
      fi
      qwen_script="$(cat)"
      for required in \
        'QWEN_STREAMING_RUNTIME_AVAILABLE' \
        'version("qwen-asr") == "0.0.6"' \
        'version("vllm") == "0.14.0"'; do
        if ! grep -Fq -- "$required" <<<"$qwen_script"; then
          printf 'Qwen image probe stdin is missing check: %s\n' "$required" >&2
          exit 97
        fi
      done
      printf '%s\n' 'QWEN_BUILD_RUNTIME_PRESENT=PASS qwen_asr=0.0.6 vllm=0.14.0 weights=NOT_LOADED'
      exit 0
    fi
    if [[ " $* " == *" /usr/local/bin/salad-nginx-diagnostics.py "* ]]; then
      for argument in "$@"; do
        case "$argument" in
          config)
            if [[ "${SMOKE_FAKE_MODE:-}" == 'config_timeout' ]]; then
              exit 124
            fi
            printf '%s\n' 'NGINX_EFFECTIVE_CONFIG_CAPTURED=YES' \
              'config_source="/etc/nginx/nginx.conf"' \
              'config_source="/etc/nginx/conf.d/default.conf"' \
              'NGINX_CONF_D_INCLUDED=YES' \
              'NGINX_EFFECTIVE_IPV4_8888=YES' 'NGINX_EFFECTIVE_IPV6_8888=YES'
            printf 'HF_TOKEN=%s\n' "$(cat "$SMOKE_FAKE_DIR/asr-secret")"
            exit 0
            ;;
          listeners)
            printf '%s\n' 'NGINX_PROCESS_RUNNING=YES' 'ss_result=UNAVAILABLE fallback=proc' \
              'RUNTIME_TCP_8888_IPV4=NOT_LISTENING' 'RUNTIME_TCP_8888_IPV6=NOT_LISTENING' \
              'JUPYTER_8889=LISTENING' 'ASR_8765=LISTENING'
            exit 0
            ;;
          nginx_tcp_ipv4|nginx_tcp_ipv6)
            printf 'probe=%s result=FAIL exception=ConnectionRefusedError errno=111\n' "$argument"
            exit 1
            ;;
          jupyter_tcp|asr_tcp)
            printf 'probe=%s result=PASS exception=NONE errno=NONE\n' "$argument"
            exit 0
            ;;
        esac
      done
    fi
    for argument in "$@"; do
      case "$argument" in
        nginx_direct|jupyter_direct|asr_direct)
          printf 'probe=%s result=PASS status=200\n' "$argument"
          exit 0
          ;;
      esac
    done
    if [[ " $* " == *" touch /workspace/ci-smoke-root-marker.txt "* ]]; then
      exit 0
    elif [[ " $* " == *" --interactive "* ]]; then
      printf '%s\n' 'Container smoke test: IPv4 loopback, password gate, login, and /workspace root passed'
    elif [[ " $* " == *" CI_ASR_SMOKE_TOKEN "* ]]; then
      printf '%s\n' 'WebSocket auth check: passed' 'Credential isolation check: passed'
    else
      printf '%s\n' 'ASR health/readiness check: passed'
    fi
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
duration="$1"
shift
if [[ "$1" == 'docker' && "$2" =~ ^(inspect|logs|top|exec)$ ]]; then
  printf '%s %s %s\n' "$duration" "$1" "$2" >>"$SMOKE_FAKE_DIR/bounded-diagnostics"
fi
exec "$@"
SH

cat >"$fake_bin/sleep" <<'SH'
#!/usr/bin/env bash
exit 0
SH

chmod 0755 "$fake_bin"/*

run_smoke() {
  local mode="$1"
  local health="$2"
  local expected_status="$3"
  local output_file="$temporary_directory/${mode}-${health}.txt"
  rm -f "$SMOKE_FAKE_DIR/python-count" "$SMOKE_FAKE_DIR/bounded-diagnostics"
  local status=0
  if PATH="$fake_bin:$PATH" SMOKE_FAKE_DIR="$temporary_directory" \
    SMOKE_FAKE_MODE="$mode" SMOKE_FAKE_HEALTH="$health" \
    bash "$repo_root/ci-smoke-test.sh" fixture-image >"$output_file" 2>&1; then
    status=0
  else
    status=$?
  fi
  if [[ "$status" != "$expected_status" ]]; then
    printf 'Expected smoke status %s, got %s for mode=%s health=%s\n' \
      "$expected_status" "$status" "$mode" "$health" >&2
    cat "$output_file" >&2
    return 1
  fi
  printf '%s' "$output_file"
}

unhealthy_output="$(run_smoke normal unhealthy 1)"
for expected in \
  'Missing-password check: passed' \
  'SMOKE_PHASE=container_start' \
  'SMOKE_PHASE=health_wait' \
  'SMOKE_FAILURE phase=health_wait exit_code=1' \
  '--- Container diagnostics' \
  '[docker inspect state]' \
  '[docker inspect health]' \
  '[docker logs]' \
  '[active processes]' \
  '[nginx effective configuration and disk files]' \
  'NGINX_EFFECTIVE_CONFIG_CAPTURED=YES' \
  'NGINX_CONF_D_INCLUDED=YES' \
  'NGINX_EFFECTIVE_IPV4_8888=YES' \
  'NGINX_EFFECTIVE_IPV6_8888=YES' \
  'ss_result=UNAVAILABLE fallback=proc' \
  'RUNTIME_TCP_8888_IPV4=NOT_LISTENING' \
  'JUPYTER_8889=LISTENING' \
  'ASR_8765=LISTENING' \
  'probe=nginx_tcp_ipv4 result=FAIL exception=ConnectionRefusedError errno=111' \
  'probe=nginx_tcp_ipv6 result=FAIL exception=ConnectionRefusedError errno=111' \
  'probe=jupyter_tcp result=PASS' \
  'probe=asr_tcp result=PASS' \
  '[direct loopback probes]' \
  'probe=nginx_direct result=PASS' \
  'probe=jupyter_direct result=PASS' \
  'probe=asr_direct result=PASS' \
  '[REDACTED_JUPYTER_PASSWORD]' \
  '[REDACTED_ASR_TOKEN]' \
  'HF_TOKEN=[REDACTED]'; do
  if ! grep -Fq -- "$expected" "$unhealthy_output"; then
    printf 'Expected diagnostic marker missing: %s\n' "$expected" >&2
    cat "$unhealthy_output" >&2
    exit 1
  fi
done

tcp_line="$(grep -nF 'probe=nginx_tcp_ipv6' "$unhealthy_output" | head -n1 | cut -d: -f1)"
http_line="$(grep -nF 'probe=nginx_direct' "$unhealthy_output" | head -n1 | cut -d: -f1)"
if ((tcp_line >= http_line)); then
  printf '%s\n' 'TCP diagnostics did not precede HTTP probes.' >&2
  exit 1
fi

if [[ "$(grep -Fc -- '--- Container diagnostics' "$unhealthy_output")" != '1' ]]; then
  printf '%s\n' 'Failure diagnostics were printed more than once.' >&2
  cat "$unhealthy_output" >&2
  exit 1
fi
if grep -Eq 'fixture-jupyter-password|fixture-asr-token|fixture-secret' "$unhealthy_output"; then
  printf '%s\n' 'A fixture secret was printed in smoke failure diagnostics.' >&2
  exit 1
fi
if ! grep -Eq '^5s docker (inspect|logs|top|exec)$' "$SMOKE_FAKE_DIR/bounded-diagnostics"; then
  printf '%s\n' 'Docker diagnostics did not use bounded timeout calls.' >&2
  exit 1
fi
if grep -Ev '^5s docker (inspect|logs|top|exec)$' "$SMOKE_FAKE_DIR/bounded-diagnostics"; then
  printf '%s\n' 'A failure diagnostic call used an unexpected timeout budget.' >&2
  exit 1
fi

config_timeout_output="$(run_smoke config_timeout unhealthy 1)"
for expected in \
  '[nginx effective configuration] diagnostic command unavailable (exit_code=124)' \
  'RUNTIME_TCP_8888_IPV4=NOT_LISTENING' \
  'probe=nginx_tcp_ipv4 result=FAIL' \
  'probe=asr_direct result=PASS'; do
  if ! grep -Fq -- "$expected" "$config_timeout_output"; then
    printf 'Diagnostics stopped after config timeout; missing: %s\n' "$expected" >&2
    cat "$config_timeout_output" >&2
    exit 1
  fi
done

timeout_output="$(run_smoke normal starting 1)"
for expected in \
  'health_wait attempt=5 container_state=running health_status=starting' \
  'health_wait attempt=45 container_state=running health_status=starting' \
  'Timed out waiting for the IPv4 loopback /login healthcheck'; do
  if ! grep -Fq -- "$expected" "$timeout_output"; then
    printf 'Expected health-wait marker missing: %s\n' "$expected" >&2
    cat "$timeout_output" >&2
    exit 1
  fi
done

start_failure_output="$(run_smoke start_failure starting 23)"
for expected in \
  'SMOKE_PHASE=container_start' \
  'SMOKE_FAILURE phase=container_start exit_code=23' \
  '--- Container diagnostics' \
  'Container is not inspectable'; do
  if ! grep -Fq -- "$expected" "$start_failure_output"; then
    printf 'Expected EXIT trap marker missing: %s\n' "$expected" >&2
    cat "$start_failure_output" >&2
    exit 1
  fi
done
if [[ "$(grep -Fc -- '--- Container diagnostics' "$start_failure_output")" != '1' ]]; then
  printf '%s\n' 'Startup failure diagnostics were printed more than once.' >&2
  cat "$start_failure_output" >&2
  exit 1
fi

healthy_output="$(run_smoke normal healthy 0)"
for phase in container_start health_wait health_probe qwen_build_runtime_test jupyter_login_test asr_health_test websocket_test secret_log_check complete; do
  if ! grep -Fq "SMOKE_PHASE=$phase" "$healthy_output"; then
    printf 'Missing successful smoke phase: %s\n' "$phase" >&2
    cat "$healthy_output" >&2
    exit 1
  fi
done
for probe in nginx_direct jupyter_direct asr_direct; do
  if ! grep -Fq "probe=$probe result=PASS status=200" "$healthy_output"; then
    printf 'Missing successful direct endpoint probe: %s\n' "$probe" >&2
    cat "$healthy_output" >&2
    exit 1
  fi
done

if ! grep -Fq 'QWEN_BUILD_RUNTIME_PRESENT=PASS qwen_asr=0.0.6 vllm=0.14.0 weights=NOT_LOADED' "$healthy_output"; then
  printf '%s\n' 'Qwen image contract check did not run successfully.' >&2
  cat "$healthy_output" >&2
  exit 1
fi

printf '%s\n' 'CI smoke phases, failure diagnostics, bounded Docker calls, loopback probes, Qwen image contract, and redaction: PASS'
