#!/usr/bin/env bash
set -euo pipefail

image="${1:?Pass the locally loaded CI image tag.}"
run_id="${GITHUB_RUN_ID:-local}"
run_attempt="${GITHUB_RUN_ATTEMPT:-0}"
container_name="salad-jupyter-smoke-${run_id}-${run_attempt}"
missing_password_log="$(mktemp)"
container_logs="$(mktemp)"
smoke_password="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
smoke_asr_token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

cleanup() {
  docker rm --force "$container_name" >/dev/null 2>&1 || true
  rm -f "$missing_password_log" "$container_logs"
}
trap cleanup EXIT

if timeout 30s docker run --rm "$image" >"$missing_password_log" 2>&1; then
  printf '%s\n' 'Image unexpectedly started without JUPYTER_PASSWORD.' >&2
  exit 1
fi
if ! grep -Fq 'JUPYTER_PASSWORD must be set' "$missing_password_log"; then
  printf '%s\n' 'Image did not fail closed with the expected missing-password message.' >&2
  exit 1
fi
printf '%s\n' 'Missing-password check: passed'

JUPYTER_PASSWORD="$smoke_password" \
CI=true \
ASR_TEST_NO_MODEL=1 \
ASR_BACKEND=parakeet \
ASR_API_TOKEN="$smoke_asr_token" \
docker run \
  --detach \
  --rm \
  --name "$container_name" \
  --env JUPYTER_PASSWORD \
  --env CI \
  --env ASR_TEST_NO_MODEL \
  --env ASR_BACKEND \
  --env ASR_API_TOKEN \
  "$image" >/dev/null

health_status='starting'
for _ in $(seq 1 45); do
  health_status="$(docker inspect --format '{{.State.Health.Status}}' "$container_name" 2>/dev/null || true)"
  if [[ "$health_status" == 'healthy' ]]; then
    break
  fi
  if [[ "$health_status" == 'unhealthy' ]] || ! docker inspect "$container_name" >/dev/null 2>&1; then
    printf 'Jupyter did not become healthy (state: %s).\n' "$health_status" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$health_status" != 'healthy' ]]; then
  printf 'Timed out waiting for the IPv6 /login healthcheck (state: %s).\n' "$health_status" >&2
  exit 1
fi

docker exec "$container_name" sh -c 'touch /workspace/ci-smoke-root-marker.txt'

# Exercise the same IPv6 listener used by Salad, without an IPv4 host-port mapping.
CI_JUPYTER_TEST_PASSWORD="$smoke_password" \
docker exec --interactive \
  --env CI_JUPYTER_TEST_PASSWORD \
  "$container_name" \
  python3 - <<'PY'
import http.cookiejar
import json
import os
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPCookieProcessor,
    Request,
    build_opener,
)

base_url = "http://[::1]:8888"
password = os.environ["CI_JUPYTER_TEST_PASSWORD"]


class XsrfParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.value = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return
        attributes = dict(attrs)
        if attributes.get("name") == "_xsrf":
            self.value = attributes.get("value")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


anonymous = build_opener()
with anonymous.open(base_url + "/lab", timeout=10) as response:
    assert urlsplit(response.geturl()).path.rstrip("/") == "/login", (
        "Unauthenticated browser access did not reach the login page"
    )
    assert b"password" in response.read().lower()

anonymous_api = build_opener(NoRedirect())
try:
    response = anonymous_api.open(base_url + "/api/contents", timeout=10)
except HTTPError as error:
    assert error.code in (302, 401, 403), f"Unexpected unauthenticated API status {error.code}"
    if error.code == 302:
        assert urlsplit(error.headers.get("Location", "")).path.rstrip("/") == "/login"
else:
    raise AssertionError(f"Unauthenticated contents API returned {response.status}")

cookies = http.cookiejar.CookieJar()
authenticated = build_opener(HTTPCookieProcessor(cookies))
with authenticated.open(base_url + "/login", timeout=10) as response:
    parser = XsrfParser()
    parser.feed(response.read().decode("utf-8", errors="replace"))

xsrf = parser.value
if not xsrf:
    xsrf_cookie = next((cookie.value for cookie in cookies if cookie.name == "_xsrf"), None)
    xsrf = xsrf_cookie
assert xsrf, "Jupyter login page did not provide an XSRF token"

login_request = Request(
    base_url + "/login",
    data=urlencode({"password": password, "_xsrf": xsrf, "next": "/lab"}).encode("utf-8"),
    headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": base_url + "/login"},
)
with authenticated.open(login_request, timeout=10) as response:
    assert urlsplit(response.geturl()).path.rstrip("/") != "/login", "Password login was rejected"

with authenticated.open(base_url + "/api/contents", timeout=10) as response:
    contents = json.load(response)
names = {entry.get("name") for entry in contents.get("content", [])}
assert "ci-smoke-root-marker.txt" in names, "Authenticated contents root is not /workspace"

print("Container smoke test: IPv6 loopback, password gate, login, and /workspace root passed")
PY

docker exec "$container_name" python3 - <<'PY'
import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen

deadline = time.monotonic() + 30
while True:
    try:
        response = urlopen("http://127.0.0.1:8765/asr/health", timeout=2)
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(1)
with response:
    health = json.load(response)
    assert response.status == 200 and health["status"] == "ok"
    assert health["model_loaded"] is False
try:
    urlopen("http://127.0.0.1:8765/asr/readiness", timeout=5)
except HTTPError as error:
    assert error.code == 503
    assert json.load(error)["ready"] is False
else:
    raise AssertionError("CI smoke mode incorrectly reported a model as ready")
print("ASR health/readiness check: process alive and no-model readiness correctly reported")
PY

CI_ASR_SMOKE_TOKEN="$smoke_asr_token" \
docker exec --env CI_ASR_SMOKE_TOKEN \
  "$container_name" \
  /opt/asr-venv/bin/python - <<'PY'
import asyncio
import json
import os
from urllib.parse import urlencode

import websockets

base = "ws://127.0.0.1:8765/asr/ws"

async def check():
    try:
        async with websockets.connect(base + "?" + urlencode({"token": "wrong" * 8}), open_timeout=5):
            raise AssertionError("An invalid ASR token was accepted")
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        assert status == 403, f"Unexpected unauthorized status {status}"

    url = base + "?" + urlencode({"token": os.environ["CI_ASR_SMOKE_TOKEN"]})
    async with websockets.connect(url, open_timeout=5) as socket:
        message = json.loads(await socket.recv())
        assert message["event"] == "error" and message["code"] == "model_not_ready"

asyncio.run(check())

import pathlib
found_jupyter = False
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        command = (proc / "cmdline").read_bytes().lower()
        if b"jupyter" not in command or b"lab" not in command:
            continue
        found_jupyter = True
        environment = (proc / "environ").read_bytes().split(b"\0")
        keys = {entry.split(b"=", 1)[0] for entry in environment if b"=" in entry}
        assert b"ASR_API_TOKEN" not in keys
        assert b"HF_TOKEN" not in keys
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert found_jupyter, "Could not locate the JupyterLab process for credential isolation check"
print("WebSocket auth check: invalid token rejected; valid token receives test-mode not-ready response")
print("Credential isolation check: Jupyter process does not inherit ASR_API_TOKEN or HF_TOKEN")
PY

docker logs "$container_name" >"$container_logs" 2>&1
if grep -Fq "$smoke_password" "$container_logs"; then
  printf '%s\n' 'The temporary password appeared in container logs.' >&2
  exit 1
fi
if grep -Fq "$smoke_asr_token" "$container_logs"; then
  printf '%s\n' 'The temporary ASR token appeared in container logs.' >&2
  exit 1
fi
if grep -Eiq 'https?://[^[:space:]]+\?token=[[:alnum:]_-]{24,}' "$container_logs"; then
  printf '%s\n' 'A generated Jupyter URL token appeared in container logs.' >&2
  exit 1
fi

printf '%s\n' 'Secret/log leakage check: passed'
