from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ASR_ROOT = Path(__file__).resolve().parents[1]


REAL_FASTAPI_CHECK = r'''
import asyncio
import importlib.metadata

assert importlib.metadata.version("fastapi") == "0.142.1"
from asr_lab.service import app

assert app.title == "Salad ASR Lab"

async def get(path):
    messages = []
    request_delivered = False

    async def receive():
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    raw_path = path.encode("ascii")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 32100),
        "server": ("127.0.0.1", 80),
        "state": {},
        "extensions": {},
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], dict(start["headers"]), body

async def main():
    status, headers, body = await get("/asr/benchmark")
    assert status == 200, status
    assert b"text/html" in headers[b"content-type"]
    assert b"no-store" in headers[b"cache-control"]
    assert b"Live ASR" in body

    status, headers, body = await get("/asr/benchmark/app.mjs")
    assert status == 200, status
    assert body

    status, _headers, body = await get("/asr/benchmark/not-allowlisted")
    assert status == 404, status
    assert b"Not found" in body

asyncio.run(main())
print("FASTAPI_REAL_IMPORT_AND_ASGI_ROUTES=PASS fastapi=0.142.1")
'''


class FastAPIRealImportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI runtime dependency is unavailable")
    def test_pinned_fastapi_imports_service_and_serves_benchmark_routes(self):
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ASR_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        env["CI"] = "true"
        env["ASR_TEST_NO_MODEL"] = "1"
        env["ASR_BACKEND"] = "parakeet"
        result = subprocess.run(
            [sys.executable, "-c", REAL_FASTAPI_CHECK],
            cwd=ASR_ROOT.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("FASTAPI_REAL_IMPORT_AND_ASGI_ROUTES=PASS fastapi=0.142.1", result.stdout)


if __name__ == "__main__":
    unittest.main()
