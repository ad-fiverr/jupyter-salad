"""ASR application-layer WebSocket token check."""
from __future__ import annotations

import hmac


def token_matches(supplied: str | None, expected: str | None) -> bool:
    if not supplied or not expected:
        return False
    try:
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    except (AttributeError, UnicodeEncodeError):
        return False