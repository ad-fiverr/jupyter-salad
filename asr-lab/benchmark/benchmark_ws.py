"""Send audio fixtures through the same authenticated WebSocket path as the app."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import statistics
import time
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1_600  # 100 ms, like the historical AudioWorklet client.


def read_fixture(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path.name}: fixture must be mono 16 kHz PCM")
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError(f"{path.name}: fixture must be uncompressed PCM16")
        pcm = audio.readframes(audio.getnframes())
        duration_ms = audio.getnframes() * 1000 / SAMPLE_RATE
    if not pcm or len(pcm) % 2:
        raise ValueError(f"{path.name}: invalid PCM16 payload")
    return pcm, duration_ms


def health_url(ws_url: str) -> str:
    parsed = urllib.parse.urlsplit(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, "/asr/health", "", ""))


def build_ws_url(ws_url: str, token: str) -> str:
    parsed = urllib.parse.urlsplit(ws_url)
    query = [(key, value) for key, value in urllib.parse.parse_qsl(parsed.query) if key != "token"]
    query.append(("token", token))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), ""))


def normalize_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return None
    row = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        next_row = [i]
        for j, hyp_word in enumerate(hyp, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (ref_word != hyp_word)))
        row = next_row
    return row[-1] / len(ref)


async def receive_fixture_results(ws, flush_request_id: str):
    transcripts = []
    errors = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        data = json.loads(raw)
        if data.get("event") == "error":
            errors.append({"code": data.get("code", "unknown")})
            continue
        if data.get("event") == "transcript" or data.get("type") == "transcript":
            transcripts.append({**data, "client_received_at": time.perf_counter()})
            continue
        if (
            data.get("event") == "flush_complete"
            and data.get("request_id") == flush_request_id
        ):
            return transcripts, errors, time.perf_counter()


def audio_end_to_last_transcript_ms(audio_end: float, transcripts: list[dict[str, object]]) -> float | None:
    """Measure from fixture end to the last transcript receipt, if any."""
    if not transcripts:
        return None
    received_at = transcripts[-1].get("client_received_at")
    if not isinstance(received_at, (int, float)):
        return None
    return max(0.0, (float(received_at) - audio_end) * 1000)


async def run_one(ws_url: str, token: str, fixture: Path) -> dict[str, object]:
    import websockets

    pcm, duration_ms = read_fixture(fixture)
    reference_file = fixture.with_suffix(".txt")
    reference = reference_file.read_text(encoding="utf-8").strip() if reference_file.exists() else None
    url = build_ws_url(ws_url, token)
    flush_request_id = uuid.uuid4().hex
    parsed = urllib.parse.urlsplit(ws_url)
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    t0 = time.perf_counter()
    async with websockets.connect(url, ping_interval=30, ping_timeout=20, max_size=1_048_576) as ws:
        ping_started = time.perf_counter()
        pong = await ws.ping()
        await asyncio.wait_for(pong, timeout=5)
        network_rtt_ms = (time.perf_counter() - ping_started) * 1000
        receive_task = asyncio.create_task(receive_fixture_results(ws, flush_request_id))
        for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
            chunk = pcm[offset:offset + CHUNK_SAMPLES * 2]
            payload = {
                "source": "mic",
                "speaker": "you",
                "encoding": "pcm_int16",
                "sample_rate": SAMPLE_RATE,
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
            await ws.send(json.dumps(payload, separators=(",", ":")))
            await asyncio.sleep(len(chunk) / (SAMPLE_RATE * 2))
        # Chunks are sent at real-time cadence. The timestamp after the final
        # chunk's duration is the end of the complete fixture, including any
        # trailing silence, rather than an inferred last-voice timestamp.
        audio_end = time.perf_counter()
        await ws.send(json.dumps({
            "event": "flush",
            "source": "mic",
            "request_id": flush_request_id,
        }, separators=(",", ":")))
        transcripts, errors, _flush_completed_at = await receive_task
        total_audio_end_to_transcript_ms = audio_end_to_last_transcript_ms(audio_end, transcripts)
    end_to_end_wall_ms = (time.perf_counter() - t0) * 1000
    segment_metrics = [
        {
            "MODEL_INFERENCE_MS": transcript.get("MODEL_INFERENCE_MS"),
            "SEGMENT_WAIT_MS": transcript.get("SEGMENT_WAIT_MS"),
            "SERVER_TO_TRANSCRIPT_MS": transcript.get("SERVER_TO_TRANSCRIPT_MS"),
            "SERVER_RECEIVE_TO_TRANSCRIPT_MS": transcript.get("SERVER_RECEIVE_TO_TRANSCRIPT_MS"),
            "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": transcript.get("SERVER_AUDIO_END_TO_TRANSCRIPT_MS"),
            "audio_duration_ms": transcript.get("audio_duration_ms"),
            "text": str(transcript.get("text", "")),
            "backend": transcript.get("backend"),
        }
        for transcript in transcripts
    ]
    inference_values = [
        float(segment["MODEL_INFERENCE_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("MODEL_INFERENCE_MS"), (int, float))
    ]
    inference_ms = sum(inference_values) if inference_values else None
    inference_rtf = float(inference_ms) / duration_ms if inference_ms is not None and duration_ms else None
    end_to_end_rtf = (
        total_audio_end_to_transcript_ms / duration_ms
        if total_audio_end_to_transcript_ms is not None and duration_ms
        else None
    )
    try:
        with urllib.request.urlopen(health_url(ws_url), timeout=5) as response:
            health = json.load(response)
    except Exception:
        health = {}
    hypothesis = " ".join(
        str(transcript.get("text", "")).strip()
        for transcript in transcripts
        if str(transcript.get("text", "")).strip()
    )
    server_values = [
        float(segment["SERVER_TO_TRANSCRIPT_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SERVER_TO_TRANSCRIPT_MS"), (int, float))
    ]
    server_receive_values = [
        float(segment["SERVER_RECEIVE_TO_TRANSCRIPT_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SERVER_RECEIVE_TO_TRANSCRIPT_MS"), (int, float))
    ]
    server_audio_end_values = [
        float(segment["SERVER_AUDIO_END_TO_TRANSCRIPT_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SERVER_AUDIO_END_TO_TRANSCRIPT_MS"), (int, float))
    ]
    segment_wait_values = [
        float(segment["SEGMENT_WAIT_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SEGMENT_WAIT_MS"), (int, float))
    ]
    return {
        "fixture": fixture.name,
        "ws_url": safe_url,
        "backend": (transcripts[0].get("backend") if transcripts else health.get("backend")),
        "model_id": health.get("model_id"),
        "model_revision": health.get("model_revision"),
        "gpu": [item.get("vram_after_warmup", {}).get("device") for item in health.get("worker_metrics", [])],
        "model_load_ms": [item.get("model_load_ms") for item in health.get("worker_metrics", [])],
        "MODEL_STATE_AT_RUN": "warm_model_ready",
        "audio_duration_ms": duration_ms,
        "segments": segment_metrics,
        "segment_count": len(segment_metrics),
        "MODEL_INFERENCE_MS": inference_ms,
        "SEGMENT_WAIT_MS": sum(segment_wait_values) if segment_wait_values else None,
        "SERVER_TO_TRANSCRIPT_MS": max(server_values) if server_values else None,
        "SERVER_RECEIVE_TO_TRANSCRIPT_MS": max(server_receive_values) if server_receive_values else None,
        "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": max(server_audio_end_values) if server_audio_end_values else None,
        "NETWORK_RTT_MS": round(network_rtt_ms, 2),
        "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS": (
            round(total_audio_end_to_transcript_ms, 2)
            if total_audio_end_to_transcript_ms is not None
            else None
        ),
        "end_to_end_wall_ms": round(end_to_end_wall_ms, 2),
        "MODEL_INFERENCE_RTF": round(inference_rtf, 4) if inference_rtf is not None else None,
        "END_TO_END_RTF": round(end_to_end_rtf, 4) if end_to_end_rtf is not None else None,
        "VRAM_MEASUREMENT": "UNAVAILABLE",
        "DEVICE_GLOBAL_VRAM_OBSERVATIONS": {
            "scope": "device_global_and_torch_allocator_not_backend_attributed",
            "before_load": [item.get("vram_before") for item in health.get("worker_metrics", [])],
            "after_load": [item.get("vram_after_load") for item in health.get("worker_metrics", [])],
            "after_warmup": [item.get("vram_after_warmup") for item in health.get("worker_metrics", [])],
        },
        "transcript": hypothesis,
        "wer": word_error_rate(reference, hypothesis) if reference is not None else None,
        "errors": errors,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.9999)))]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = (
        "MODEL_INFERENCE_MS", "SEGMENT_WAIT_MS", "SERVER_TO_TRANSCRIPT_MS",
        "SERVER_RECEIVE_TO_TRANSCRIPT_MS", "SERVER_AUDIO_END_TO_TRANSCRIPT_MS",
        "NETWORK_RTT_MS", "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS",
        "MODEL_INFERENCE_RTF", "END_TO_END_RTF",
    )
    summary: dict[str, object] = {
        "runs": len(rows),
        "errors": sum(bool(row.get("errors")) for row in rows),
        "server_metric_sample_unit": "segment",
        "client_metric_sample_unit": "fixture_run",
    }
    for key in numeric:
        segment_values = [
            float(segment[key])
            for row in rows
            for segment in row.get("segments", [])
            if key in {
                "MODEL_INFERENCE_MS", "SEGMENT_WAIT_MS", "SERVER_TO_TRANSCRIPT_MS",
                "SERVER_RECEIVE_TO_TRANSCRIPT_MS", "SERVER_AUDIO_END_TO_TRANSCRIPT_MS",
            }
            and isinstance(segment, dict)
            and isinstance(segment.get(key), (int, float))
        ]
        values = segment_values or [
            float(row[key]) for row in rows if isinstance(row.get(key), (int, float))
        ]
        summary[key + "_P50"] = statistics.median(values) if values else None
        summary[key + "_P95"] = percentile(values, 0.95)
    wer_values = [float(row["wer"]) for row in rows if isinstance(row.get("wer"), (int, float))]
    summary["wer_mean"] = statistics.mean(wer_values) if wer_values else None
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ws_url", help="for example wss://host.salad.cloud/asr/ws (token read from ASR_API_TOKEN)")
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        raise SystemExit("--repeat must be between 1 and 10")
    token = os.environ.get("ASR_API_TOKEN", "")
    if len(token.encode("utf-8")) < 24:
        raise SystemExit("ASR_API_TOKEN must be provided in the environment; it is never printed.")
    results = []
    summaries = {}
    for fixture in args.fixtures:
        rows = []
        for _ in range(args.repeat):
            try:
                row = await run_one(args.ws_url, token, fixture)
            except Exception as exc:
                row = {"fixture": fixture.name, "backend": "UNAVAILABLE", "errors": [type(exc).__name__]}
            rows.append(row)
        results.extend(rows)
        summaries[fixture.name] = summarize(rows)
    payload = json.dumps({"results": results, "summary_by_fixture": summaries}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    asyncio.run(main())

