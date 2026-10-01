"""Strict shared JSON/base64 PCM16 mono 16 kHz protocol."""
from __future__ import annotations

import base64
import binascii
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AudioChunk:
    source: str
    speaker: str
    pcm16le: bytes
    sample_rate: int
    request_id: str | None

    @property
    def sample_count(self) -> int:
        return len(self.pcm16le) // BYTES_PER_SAMPLE

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


@dataclass(frozen=True)
class FlushRequest:
    source: str
    request_id: str | None = None


@dataclass(frozen=True)
class StreamStart:
    source: str
    language: str
    context: str
    chunk_size_ms: int | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class BenchmarkPing:
    request_id: str


def _safe_json(raw: str | bytes, max_message_bytes: int) -> Any:
    if isinstance(raw, bytes):
        if len(raw) > max_message_bytes:
            raise ProtocolError("message_too_large", "WebSocket message exceeds the configured limit.")
        try:
            raw = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProtocolError("invalid_json", "Message must be UTF-8 JSON.") from exc
    if not isinstance(raw, str):
        raise ProtocolError("invalid_json", "Message must be a JSON object.")
    if len(raw.encode("utf-8")) > max_message_bytes:
        raise ProtocolError("message_too_large", "WebSocket message exceeds the configured limit.")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ProtocolError("invalid_json", "Message must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid_json", "Message must be a JSON object.")
    return value


def parse_message(
    raw: str | bytes,
    *,
    max_message_bytes: int,
    max_chunk_seconds: float,
    max_context_chars: int = 512,
) -> AudioChunk | FlushRequest | BenchmarkPing | StreamStart:
    value = _safe_json(raw, max_message_bytes)
    event = value.get("event")

    if event == "benchmark_ping":
        if set(value) != {"event", "request_id"}:
            raise ProtocolError("invalid_benchmark_ping", "Benchmark ping must contain only event and request_id.")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ProtocolError("invalid_request_id", "request_id must be a non-empty string up to 128 characters.")
        return BenchmarkPing(request_id=request_id)

    if event == "flush":
        if set(value) - {"event", "source", "request_id"}:
            raise ProtocolError("invalid_flush", "Flush message contains unsupported fields.")
        source = value.get("source")
        if source not in ("mic", "system"):
            raise ProtocolError("invalid_source", "source must be mic or system.")
        request_id = value.get("request_id")
        if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 128):
            raise ProtocolError("invalid_request_id", "request_id must be a short string.")
        return FlushRequest(source=source, request_id=request_id)

    if event == "stream_start":
        if set(value) - {"event", "source", "language", "context", "chunk_size_ms", "request_id"}:
            raise ProtocolError("invalid_stream_start", "Stream start contains unsupported fields.")
        source = value.get("source")
        if source not in ("mic", "system"):
            raise ProtocolError("invalid_source", "source must be mic or system.")
        language = value.get("language", "auto")
        if not isinstance(language, str) or not language or len(language) > 16:
            raise ProtocolError("invalid_language", "language must be a supported short code or auto.")
        language = language.strip().lower()
        if language != "auto" and not language.isalpha():
            raise ProtocolError("invalid_language", "language must be a supported short code or auto.")
        context = value.get("context", "")
        if not isinstance(context, str) or len(context) > max_context_chars:
            raise ProtocolError("invalid_context", "context exceeds the configured character limit.")
        if any(ord(char) < 32 or ord(char) == 127 for char in context):
            raise ProtocolError("invalid_context", "context must not contain control characters.")
        chunk_size_ms = value.get("chunk_size_ms")
        if chunk_size_ms is not None and (
            not isinstance(chunk_size_ms, int)
            or isinstance(chunk_size_ms, bool)
            or chunk_size_ms not in (250, 500, 1000, 2000)
        ):
            raise ProtocolError("invalid_stream_chunk", "chunk_size_ms must be one of 250, 500, 1000, or 2000.")
        request_id = value.get("request_id")
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id or len(request_id) > 128
        ):
            raise ProtocolError("invalid_request_id", "request_id must be a non-empty string up to 128 characters.")
        return StreamStart(
            source=source,
            language=language,
            context=" ".join(context.split()),
            chunk_size_ms=chunk_size_ms,
            request_id=request_id,
        )

    allowed = {"source", "speaker", "encoding", "sample_rate", "audio", "request_id", "type"}
    if set(value) - allowed:
        raise ProtocolError("unsupported_fields", "Audio message contains unsupported fields.")
    if value.get("type", "audio") != "audio":
        raise ProtocolError("invalid_type", "Expected an audio message or flush event.")

    source = value.get("source")
    if source not in ("mic", "system"):
        raise ProtocolError("invalid_source", "source must be mic or system.")

    speaker = value.get("speaker", "you" if source == "mic" else "them")
    if speaker not in ("you", "them", "system"):
        raise ProtocolError("invalid_speaker", "speaker must be you, them, or system.")
    if speaker == "system":
        speaker = "them"

    if value.get("encoding") != "pcm_int16":
        raise ProtocolError("unsupported_encoding", "encoding must be pcm_int16.")
    sample_rate = value.get("sample_rate")
    if isinstance(sample_rate, bool) or sample_rate != SAMPLE_RATE:
        raise ProtocolError("unsupported_sample_rate", "sample_rate must be 16000 Hz.")

    encoded = value.get("audio")
    if not isinstance(encoded, str) or not encoded:
        raise ProtocolError("invalid_audio", "audio must be non-empty base64.")
    try:
        pcm = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("invalid_base64", "audio must be valid base64.") from exc

    if not pcm:
        raise ProtocolError("invalid_audio", "audio must contain at least one sample.")
    if len(pcm) % BYTES_PER_SAMPLE:
        raise ProtocolError("invalid_pcm_length", "PCM16 payload must contain an even number of bytes.")
    duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    if duration > max_chunk_seconds:
        raise ProtocolError("chunk_too_large", "Audio chunk exceeds the configured duration limit.")

    request_id = value.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 128):
        raise ProtocolError("invalid_request_id", "request_id must be a short string.")

    return AudioChunk(
        source=source,
        speaker=speaker,
        pcm16le=pcm,
        sample_rate=SAMPLE_RATE,
        request_id=request_id,
    )


def pcm_rms_normalized(pcm16le: bytes) -> float:
    """Compute RMS using little-endian int16 without NumPy or audio logging."""
    count = len(pcm16le) // BYTES_PER_SAMPLE
    if count == 0:
        return 0.0
    total = 0.0
    for (sample,) in struct.iter_unpack("<h", pcm16le):
        total += float(sample) * float(sample)
    return math.sqrt(total / count) / 32768.0
