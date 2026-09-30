"""Container-local HTTP readiness check for nginx over IPv4 loopback."""

import sys
from urllib.request import ProxyHandler, build_opener


HEALTH_URL = "http://127.0.0.1:8888/login"


def check_health() -> int:
    # A loopback check must stay inside the container, regardless of proxy env.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(HEALTH_URL, timeout=3) as response:
            if response.status != 200:
                print(f"healthcheck=failed status={response.status}", file=sys.stderr)
                return 1
    except Exception as error:
        # Keep health output diagnostic but never echo URLs, headers, or env values.
        print(f"healthcheck=failed exception={type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(check_health())
