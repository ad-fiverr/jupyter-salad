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


class BenchmarkRunError(RuntimeError):
    def __init__(self, code: str, backend: str):
        super().__init__(code)
        self.code = code
        self.backend = backend


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
    partials = []
    candidates = []
    errors = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        data = json.loads(raw)
        if data.get("event") == "error":
            errors.append({"code": data.get("code", "unknown")})
            continue
        if data.get("event") in {"partial_candidate", "partial_transcript"}:
            partials.append({**data, "client_received_at": time.perf_counter()})
            continue
        if data.get("event") == "final_candidate" or data.get("candidate_only") is True:
            candidates.append({**data, "client_received_at": time.perf_counter()})
            continue
        if data.get("event") == "transcript" or data.get("type") == "transcript":
            transcripts.append({**data, "client_received_at": time.perf_counter()})
            continue
        if (
            data.get("event") == "flush_complete"
            and data.get("request_id") == flush_request_id
        ):
            return transcripts, partials, candidates, errors, time.perf_counter()


def audio_end_to_last_transcript_ms(audio_end: float, transcripts: list[dict[str, object]]) -> float | None:
    """Measure from fixture end to the last transcript receipt, if any."""
    if not transcripts:
        return None
    received_at = transcripts[-1].get("client_received_at")
    if not isinstance(received_at, (int, float)):
        return None
    return max(0.0, (float(received_at) - audio_end) * 1000)


async def run_one(
    ws_url: str,
    token: str,
    fixture: Path,
    *,
    qwen_chunk_ms: int = 1000,
    qwen_language: str = "auto",
    qwen_context: str = "",
    benchmark_concurrency: int = 1,
) -> dict[str, object]:
    import websockets

    pcm, duration_ms = read_fixture(fixture)
    reference_file = fixture.with_suffix(".txt")
    reference = reference_file.read_text(encoding="utf-8").strip() if reference_file.exists() else None
    url = build_ws_url(ws_url, token)
    try:
        with urllib.request.urlopen(health_url(ws_url), timeout=5) as response:
            health = json.load(response)
    except Exception:
        health = {}
    backend = str(health.get("backend") or "unknown")
    flush_request_id = uuid.uuid4().hex
    parsed = urllib.parse.urlsplit(ws_url)
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    t0 = time.perf_counter()
    client_eos_at: float | None = None
    async with websockets.connect(url, ping_interval=30, ping_timeout=20, max_size=1_048_576) as ws:
        ping_started = time.perf_counter()
        pong = await ws.ping()
        await asyncio.wait_for(pong, timeout=5)
        network_rtt_ms = (time.perf_counter() - ping_started) * 1000
        if backend == "qwen3_asr":
            start_request_id = uuid.uuid4().hex
            await ws.send(json.dumps({
                "event": "stream_start",
                "source": "mic",
                "request_id": start_request_id,
                "language": qwen_language,
                "context": qwen_context,
                "chunk_size_ms": qwen_chunk_ms,
            }, separators=(",", ":")))
            started_message = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if started_message.get("event") != "stream_started":
                raise BenchmarkRunError(
                    str(started_message.get("code", "stream_start_failed")),
                    backend,
                )
        receive_task = asyncio.create_task(receive_fixture_results(ws, flush_request_id))
        audio_started_at = time.perf_counter()
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
        client_eos_at = audio_end
        await ws.send(json.dumps({
            "event": "flush",
            "source": "mic",
            "request_id": flush_request_id,
        }, separators=(",", ":")))
        transcripts, partials, candidates, errors, _flush_completed_at = await receive_task
        total_audio_end_to_transcript_ms = audio_end_to_last_transcript_ms(audio_end, transcripts)
        total_audio_end_to_candidate_ms = audio_end_to_last_transcript_ms(audio_end, candidates)
    end_to_end_wall_ms = (time.perf_counter() - t0) * 1000
    is_qwen = backend == "qwen3_asr"
    qwen_final = candidates[-1] if is_qwen and candidates else {}
    final_transcript = transcripts[-1] if transcripts else {}
    client_eos_to_transcript_ms = (
        max(0.0, (float(final_transcript["client_received_at"]) - client_eos_at) * 1000)
        if not is_qwen and client_eos_at is not None
        and isinstance(final_transcript.get("client_received_at"), (int, float))
        else None
    )
    client_eos_to_candidate_ms = (
        max(0.0, (float(qwen_final["client_received_at"]) - client_eos_at) * 1000)
        if is_qwen and client_eos_at is not None
        and isinstance(qwen_final.get("client_received_at"), (int, float))
        else None
    )
    server_eos_to_candidate_ms = (
        qwen_final.get("FINALIZATION_AFTER_SERVER_EOS_MS") if is_qwen else None
    )
    segment_metrics = [
        {
            "event": transcript.get("event"),
            "candidate_only": transcript.get("candidate_only") is True,
            "provisional": transcript.get("provisional") is True,
            "truth_status": transcript.get("truth_status"),
            "AUDIO_DURATION_MS": transcript.get("AUDIO_DURATION_MS", transcript.get("audio_duration_ms")),
            "SERVER_ENDPOINTING_MS": transcript.get("SERVER_ENDPOINTING_MS"),
            "SERVER_QUEUE_WAIT_MS": transcript.get("SERVER_QUEUE_WAIT_MS", transcript.get("queue_wait_ms")),
            "SERVER_MODEL_INFERENCE_MS": transcript.get("SERVER_MODEL_INFERENCE_MS", transcript.get("MODEL_INFERENCE_MS")),
            "SERVER_POSTPROCESS_MS": transcript.get("SERVER_POSTPROCESS_MS"),
            "SERVER_EOS_TO_TRANSCRIPT_MS": (
                transcript.get("SERVER_EOS_TO_TRANSCRIPT_MS")
                if isinstance(transcript.get("SERVER_EOS_TO_TRANSCRIPT_MS"), (int, float))
                else transcript.get("FINALIZATION_AFTER_SERVER_EOS_MS", transcript.get("SERVER_AUDIO_END_TO_TRANSCRIPT_MS"))
            ),
            "CLIENT_EOS_TO_TRANSCRIPT_MS": client_eos_to_transcript_ms,
            "MODEL_INFERENCE_MS": transcript.get("MODEL_INFERENCE_MS"),
            "SEGMENT_WAIT_MS": transcript.get("SEGMENT_WAIT_MS"),
            "SERVER_TO_TRANSCRIPT_MS": transcript.get("SERVER_TO_TRANSCRIPT_MS"),
            "SERVER_RECEIVE_TO_TRANSCRIPT_MS": transcript.get("SERVER_RECEIVE_TO_TRANSCRIPT_MS"),
            "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": transcript.get("SERVER_AUDIO_END_TO_TRANSCRIPT_MS"),
            "audio_duration_ms": transcript.get("AUDIO_DURATION_MS", transcript.get("audio_duration_ms")),
            "text": str(transcript.get("text", "")),
            "backend": transcript.get("backend"),
        }
        for transcript in transcripts
    ]
    inference_values = [
        float(segment["SERVER_MODEL_INFERENCE_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SERVER_MODEL_INFERENCE_MS"), (int, float))
    ]
    inference_ms = sum(inference_values) if inference_values else None
    inference_rtf = float(inference_ms) / duration_ms if inference_ms is not None and duration_ms else None
    end_to_end_rtf = (
        total_audio_end_to_transcript_ms / duration_ms
        if total_audio_end_to_transcript_ms is not None and duration_ms
        else None
    )
    hypothesis_events = candidates if is_qwen else transcripts
    hypothesis = " ".join(
        str(transcript.get("text", "")).strip()
        for transcript in hypothesis_events
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
        float(segment["SERVER_EOS_TO_TRANSCRIPT_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SERVER_EOS_TO_TRANSCRIPT_MS"), (int, float))
    ]
    segment_wait_values = [
        float(segment["SEGMENT_WAIT_MS"])
        for segment in segment_metrics
        if isinstance(segment.get("SEGMENT_WAIT_MS"), (int, float))
    ]
    endpoint_values = [float(s["SERVER_ENDPOINTING_MS"]) for s in segment_metrics if isinstance(s.get("SERVER_ENDPOINTING_MS"), (int, float))]
    queue_values = [float(s["SERVER_QUEUE_WAIT_MS"]) for s in segment_metrics if isinstance(s.get("SERVER_QUEUE_WAIT_MS"), (int, float))]
    postprocess_values = [float(s["SERVER_POSTPROCESS_MS"]) for s in segment_metrics if isinstance(s.get("SERVER_POSTPROCESS_MS"), (int, float))]
    partial_intervals = [
        max(0.0, (float(current["client_received_at"]) - float(previous["client_received_at"])) * 1000)
        for previous, current in zip(partials, partials[1:])
    ]
    first_partial_client_ms = (
        max(0.0, (float(partials[0]["client_received_at"]) - audio_started_at) * 1000)
        if partials else None
    )
    qwen_cumulative_decode_ms = qwen_final.get("QWEN_CUMULATIVE_DECODE_WALL_MS")
    transcript_wer = word_error_rate(reference, hypothesis) if reference is not None and not is_qwen else None
    candidate_wer = word_error_rate(reference, hypothesis) if reference is not None and is_qwen else None
    return {
        "fixture": fixture.name,
        "ws_url": safe_url,
        "backend": (transcripts[-1].get("backend") if transcripts else health.get("backend")),
        "model_id": health.get("model_id"),
        "model_revision": health.get("model_revision"),
        "runtime_provenance": health.get("runtime_provenance"),
        "gpu": [item.get("vram_after_warmup", {}).get("device") for item in health.get("worker_metrics", [])],
        "model_load_ms": [item.get("model_load_ms") for item in health.get("worker_metrics", [])],
        "MODEL_STATE_AT_RUN": "warm_model_ready",
        "audio_duration_ms": duration_ms,
        "AUDIO_DURATION_MS": duration_ms,
        "benchmark_concurrency": benchmark_concurrency,
        "transcript_mode": "STREAMING_PARTIALS" if is_qwen else "FINAL_SEGMENT",
        "streaming": {
            "streaming_class": "accumulated-audio-pseudostreaming" if is_qwen else None,
            "audio_push_interval_ms": 100 if is_qwen else None,
            "model_decode_chunk_ms": qwen_chunk_ms if is_qwen else None,
            "language": qwen_language if is_qwen else None,
            "partial_count": len(partials),
            "partials": partials,
            "FIRST_PARTIAL_MS": partials[0].get("FIRST_PARTIAL_MS") if partials else None,
            "CLIENT_FIRST_PARTIAL_MS": first_partial_client_ms,
            "PARTIAL_INTERVAL_MS_P50": statistics.median(partial_intervals) if partial_intervals else None,
            "PARTIAL_STABILITY": partials[-1].get("PARTIAL_STABILITY") if partials else None,
            "PARTIAL_REVISION_RATE": partials[-1].get("PARTIAL_REVISION_RATE") if partials else None,
            "SERVER_EOS_TO_FINAL_CANDIDATE_MS": server_eos_to_candidate_ms,
            "CLIENT_EOS_TO_FINAL_CANDIDATE_MS": client_eos_to_candidate_ms,
            "QWEN_DECODE_CALL_WALL_MS_SUM": qwen_cumulative_decode_ms,
            "QWEN_CUMULATIVE_DECODE_WALL_MS": qwen_cumulative_decode_ms,
            "QWEN_STREAM_RTF": qwen_final.get("QWEN_STREAM_RTF"),
            "duration_bucket_s": next((limit for limit in (5, 15, 30, 60) if duration_ms <= limit * 1000), ">60"),
        },
        "segments": segment_metrics,
        "segment_count": len(segment_metrics),
        "candidate_metrics": [
            {
                "event": candidate.get("event"),
                "candidate_only": True,
                "truth_status": "candidate_only",
                "SERVER_EOS_TO_FINAL_CANDIDATE_MS": candidate.get("FINALIZATION_AFTER_SERVER_EOS_MS"),
                "CLIENT_EOS_TO_FINAL_CANDIDATE_MS": client_eos_to_candidate_ms,
                "TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS": round(total_audio_end_to_candidate_ms, 2)
                if total_audio_end_to_candidate_ms is not None else None,
                "text": str(candidate.get("text", "")),
                "backend": candidate.get("backend"),
            }
            for candidate in candidates
        ],
        "candidate_count": len(candidates),
        "MODEL_INFERENCE_MS": inference_ms,
        "SEGMENT_WAIT_MS": sum(segment_wait_values) if segment_wait_values else None,
        "SERVER_ENDPOINTING_MS": sum(endpoint_values) if endpoint_values else None,
        "SERVER_QUEUE_WAIT_MS": sum(queue_values) if queue_values else None,
        "SERVER_MODEL_INFERENCE_MS": inference_ms,
        "SERVER_POSTPROCESS_MS": sum(postprocess_values) if postprocess_values else None,
        "SERVER_TO_TRANSCRIPT_MS": max(server_values) if server_values else None,
        "SERVER_RECEIVE_TO_TRANSCRIPT_MS": max(server_receive_values) if server_receive_values else None,
        "SERVER_AUDIO_END_TO_TRANSCRIPT_MS": max(server_audio_end_values) if server_audio_end_values else None,
        "SERVER_EOS_TO_TRANSCRIPT_MS": max(server_audio_end_values) if server_audio_end_values else None,
        "CLIENT_EOS_TO_TRANSCRIPT_MS": client_eos_to_transcript_ms,
        "SERVER_EOS_TO_FINAL_CANDIDATE_MS": server_eos_to_candidate_ms,
        "CLIENT_EOS_TO_FINAL_CANDIDATE_MS": client_eos_to_candidate_ms,
        "PROXY_WS_RTT_MS": None,
        "NETWORK_RTT_MS": round(network_rtt_ms, 2),
        "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS": (
            round(total_audio_end_to_transcript_ms, 2)
            if not is_qwen and total_audio_end_to_transcript_ms is not None
            else None
        ),
        "TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS": (
            round(total_audio_end_to_candidate_ms, 2)
            if is_qwen and total_audio_end_to_candidate_ms is not None
            else None
        ),
        "end_to_end_wall_ms": round(end_to_end_wall_ms, 2),
        "MODEL_INFERENCE_RTF": round(inference_rtf, 4) if inference_rtf is not None else None,
        "metric_definitions": {
            "AUDIO_DURATION_MS": "fixture PCM duration; not latency",
            "SERVER_ENDPOINTING_MS": "sum of server-reported last-voice-to-broker intervals across segments",
            "SERVER_QUEUE_WAIT_MS": "sum of server-reported broker-to-adapter-start intervals across segments",
            "SERVER_MODEL_INFERENCE_MS": "sum of backend.transcribe adapter wall times across segments",
            "SERVER_POSTPROCESS_MS": "sum of server post-adapter preparation intervals across segments",
            "SERVER_EOS_TO_TRANSCRIPT_MS": "maximum per-segment server-only EOS-to-ready interval",
            "CLIENT_EOS_TO_TRANSCRIPT_MS": "fixture CLI perf_counter from sending final flush/EOS to receiving the final transcript",
            "PROXY_WS_RTT_MS": "not measured by this CLI; browser application ping/pong metric",
            "NETWORK_RTT_MS": "native WebSocket protocol ping frame RTT; distinct from browser application ping/pong",
            "FIRST_PARTIAL_MS": "server monotonic time from receipt of the first audio chunk to the first non-empty full transcript replacement",
            "CLIENT_FIRST_PARTIAL_MS": "client perf_counter time from first fixture audio send to receipt of first partial",
            "PARTIAL_INTERVAL_MS_P50": "client receive interval median between successive complete partial replacements",
            "PARTIAL_STABILITY": "exact shared-prefix token count between previous and current full text divided by the previous token count",
            "PARTIAL_REVISION_RATE": "one minus PARTIAL_STABILITY; appended suffix tokens are excluded",
            "SERVER_EOS_TO_FINAL_CANDIDATE_MS": "Qwen server time from flush/EOS to final candidate ready; candidate only",
            "CLIENT_EOS_TO_FINAL_CANDIDATE_MS": "Qwen client time from sending flush/EOS to receiving final candidate; candidate only",
            "TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS": "fixture PCM end to Qwen final candidate receipt; candidate only",
            "QWEN_DECODE_CALL_WALL_MS_SUM": "final Qwen cumulative decode-call wall metric; includes decoded chunks even when transcript text did not change, not GPU kernel time",
            "QWEN_STREAM_RTF": "cumulative Qwen wrapper decode call wall time divided by audio duration; not GPU utilization",
        },
        "runtime_provenance": health.get("runtime_provenance"),
        "END_TO_END_RTF": round(end_to_end_rtf, 4) if end_to_end_rtf is not None else None,
        "VRAM_MEASUREMENT": "UNAVAILABLE",
        "DEVICE_GLOBAL_VRAM_OBSERVATIONS": {
            "scope": "device_global_and_torch_allocator_not_backend_attributed",
            "before_load": [item.get("vram_before") for item in health.get("worker_metrics", [])],
            "after_load": [item.get("vram_after_load") for item in health.get("worker_metrics", [])],
            "after_warmup": [item.get("vram_after_warmup") for item in health.get("worker_metrics", [])],
        },
        "transcript": None if is_qwen else hypothesis,
        "final_candidate": hypothesis if is_qwen else None,
        "candidate_only": is_qwen,
        "FINAL_WER": transcript_wer,
        "wer": transcript_wer,
        "FINAL_CANDIDATE_WER": candidate_wer,
        "errors": errors,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.9999)))]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = (
        "MODEL_INFERENCE_MS", "SERVER_MODEL_INFERENCE_MS", "SERVER_ENDPOINTING_MS",
        "SERVER_QUEUE_WAIT_MS", "SERVER_POSTPROCESS_MS", "SERVER_EOS_TO_TRANSCRIPT_MS",
        "CLIENT_EOS_TO_TRANSCRIPT_MS",
        "SEGMENT_WAIT_MS", "SERVER_TO_TRANSCRIPT_MS", "SERVER_RECEIVE_TO_TRANSCRIPT_MS",
        "SERVER_AUDIO_END_TO_TRANSCRIPT_MS", "NETWORK_RTT_MS", "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS",
        "MODEL_INFERENCE_RTF", "END_TO_END_RTF",
    )
    summary: dict[str, object] = {
        "runs": len(rows),
        "errors": sum(bool(row.get("errors")) for row in rows),
        "server_metric_sample_unit": "segment",
        "client_metric_sample_unit": "fixture_run",
    }
    production_rows = [
        row for row in rows
        if row.get("candidate_only") is not True and row.get("truth_status") != "candidate_only"
    ]
    for key in numeric:
        segment_values = [
            float(segment[key])
            for row in production_rows
            for segment in row.get("segments", [])
            if key in {
                "MODEL_INFERENCE_MS", "SERVER_MODEL_INFERENCE_MS", "SERVER_ENDPOINTING_MS",
                "SERVER_QUEUE_WAIT_MS", "SERVER_POSTPROCESS_MS", "SERVER_EOS_TO_TRANSCRIPT_MS",
                "SEGMENT_WAIT_MS", "SERVER_TO_TRANSCRIPT_MS", "SERVER_RECEIVE_TO_TRANSCRIPT_MS",
                "SERVER_AUDIO_END_TO_TRANSCRIPT_MS",
            }
            and isinstance(segment, dict)
            and segment.get("candidate_only") is not True
            and segment.get("truth_status") != "candidate_only"
            and isinstance(segment.get(key), (int, float))
        ]
        values = segment_values or [
            float(row[key]) for row in production_rows if isinstance(row.get(key), (int, float))
        ]
        summary[key + "_P50"] = statistics.median(values) if values else None
        summary[key + "_P95"] = percentile(values, 0.95)
    wer_values = [float(row["wer"]) for row in production_rows if isinstance(row.get("wer"), (int, float))]
    summary["wer_mean"] = statistics.mean(wer_values) if wer_values else None
    summary["FINAL_WER_P50"] = statistics.median(wer_values) if wer_values else None
    summary["FINAL_WER_P95"] = percentile(wer_values, 0.95)
    qwen_attempts = [row for row in rows if row.get("backend") == "qwen3_asr"]
    qwen_rows = [
        row for row in rows
        if row.get("backend") == "qwen3_asr" and isinstance(row.get("streaming"), dict)
    ]
    growth: dict[str, dict[str, object]] = {}
    for index, limit in enumerate((5, 15, 30, 60)):
        lower = (0, 5, 15, 30)[index] * 1000
        selected = [
            row for row in qwen_rows
            if lower < float(row.get("audio_duration_ms", 0)) <= limit * 1000
        ]
        decode = [
            float(row["streaming"]["QWEN_CUMULATIVE_DECODE_WALL_MS"])
            for row in selected
            if isinstance(row["streaming"].get("QWEN_CUMULATIVE_DECODE_WALL_MS"), (int, float))
        ]
        growth[str(limit) + "s"] = {
            "samples": len(selected),
            "cumulative_decode_wall_ms_p50": statistics.median(decode) if decode else None,
            "cumulative_decode_wall_ms_p95": percentile(decode, 0.95),
            "stream_rtf_p50": statistics.median([
                float(row["streaming"]["QWEN_STREAM_RTF"])
                for row in selected
                if isinstance(row["streaming"].get("QWEN_STREAM_RTF"), (int, float))
            ]) if any(isinstance(row["streaming"].get("QWEN_STREAM_RTF"), (int, float)) for row in selected) else None,
            "server_eos_to_final_candidate_ms_p50": statistics.median([
                float(row["streaming"]["SERVER_EOS_TO_FINAL_CANDIDATE_MS"])
                for row in selected
                if isinstance(row["streaming"].get("SERVER_EOS_TO_FINAL_CANDIDATE_MS"), (int, float))
            ]) if any(isinstance(row["streaming"].get("SERVER_EOS_TO_FINAL_CANDIDATE_MS"), (int, float)) for row in selected) else None,
        }
    growth[">60s"] = {
        "samples": sum(float(row.get("audio_duration_ms", 0)) > 60_000 for row in qwen_rows),
        "cumulative_decode_wall_ms_p50": None,
        "cumulative_decode_wall_ms_p95": None,
        "stream_rtf_p50": None,
        "server_eos_to_final_candidate_ms_p50": None,
    }
    first_server_values = [
        float(row["streaming"]["FIRST_PARTIAL_MS"])
        for row in qwen_rows
        if isinstance(row["streaming"].get("FIRST_PARTIAL_MS"), (int, float))
    ]
    first_client_values = [
        float(row["streaming"]["CLIENT_FIRST_PARTIAL_MS"])
        for row in qwen_rows
        if isinstance(row["streaming"].get("CLIENT_FIRST_PARTIAL_MS"), (int, float))
    ]
    candidate_latency_metrics = (
        "SERVER_EOS_TO_FINAL_CANDIDATE_MS",
        "CLIENT_EOS_TO_FINAL_CANDIDATE_MS",
        "TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS",
    )
    candidate_wer_values = [
        float(row["FINAL_CANDIDATE_WER"])
        for row in qwen_rows
        if isinstance(row.get("FINAL_CANDIDATE_WER"), (int, float))
    ]
    summary["qwen_streaming"] = {
        "runs": len(qwen_attempts),
        "successful_runs": len(qwen_rows),
        "partial_count_total": sum(int(row["streaming"].get("partial_count", 0)) for row in qwen_rows),
        "FIRST_PARTIAL_MS_P50": statistics.median(first_server_values) if first_server_values else None,
        "CLIENT_FIRST_PARTIAL_MS_P50": statistics.median(first_client_values) if first_client_values else None,
        "final_candidate_latency_ms": {
            key: {
                "p50": statistics.median([
                    float(row[key]) for row in qwen_rows if isinstance(row.get(key), (int, float))
                ]) if any(isinstance(row.get(key), (int, float)) for row in qwen_rows) else None,
                "p95": percentile([
                    float(row[key]) for row in qwen_rows if isinstance(row.get(key), (int, float))
                ], 0.95),
            }
            for key in candidate_latency_metrics
        },
        "FINAL_CANDIDATE_WER_MEAN": statistics.mean(candidate_wer_values) if candidate_wer_values else None,
        "FINAL_CANDIDATE_WER_P50": statistics.median(candidate_wer_values) if candidate_wer_values else None,
        "FINAL_CANDIDATE_WER_P95": percentile(candidate_wer_values, 0.95),
        "concurrency_runs": {
            str(concurrency): {
                "runs": sum(row.get("benchmark_concurrency") == concurrency for row in qwen_attempts),
                "server_eos_to_final_candidate_ms_p50": statistics.median([
                    float(row["streaming"]["SERVER_EOS_TO_FINAL_CANDIDATE_MS"])
                    for row in qwen_rows
                    if row.get("benchmark_concurrency") == concurrency
                    and isinstance(row["streaming"].get("SERVER_EOS_TO_FINAL_CANDIDATE_MS"), (int, float))
                ]) if any(
                    row.get("benchmark_concurrency") == concurrency
                    and isinstance(row["streaming"].get("SERVER_EOS_TO_FINAL_CANDIDATE_MS"), (int, float))
                    for row in qwen_rows
                ) else None,
                "capacity_errors": sum(
                    any(error.get("code") == "stream_capacity_exceeded" for error in row.get("errors", []) if isinstance(error, dict))
                    for row in qwen_attempts
                    if row.get("benchmark_concurrency") == concurrency
                ),
            }
            for concurrency in sorted({
                row.get("benchmark_concurrency")
                for row in qwen_attempts
                if isinstance(row.get("benchmark_concurrency"), int)
            })
        },
        "decode_cost_by_duration_bucket": growth,
    }
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ws_url", help="for example wss://host.salad.cloud/asr/ws (token read from ASR_API_TOKEN)")
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qwen-chunk-ms", type=int, choices=(250, 500, 1000, 2000), default=1000)
    parser.add_argument("--qwen-language", default="auto")
    parser.add_argument("--qwen-context", default="")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="maximum fixture runs at once (1-128); server session/connection limits still apply",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        raise SystemExit("--repeat must be between 1 and 10")
    if args.concurrency < 1 or args.concurrency > 128:
        raise SystemExit("--concurrency must be between 1 and 128")
    token = os.environ.get("ASR_API_TOKEN", "")
    if len(token.encode("utf-8")) < 24:
        raise SystemExit("ASR_API_TOKEN must be provided in the environment; it is never printed.")
    results = []
    summaries = {}
    for fixture in args.fixtures:
        rows = []
        for batch_start in range(0, args.repeat, args.concurrency):
            batch_size = min(args.concurrency, args.repeat - batch_start)
            pending = [
                run_one(
                    args.ws_url,
                    token,
                    fixture,
                    qwen_chunk_ms=args.qwen_chunk_ms,
                    qwen_language=args.qwen_language,
                    qwen_context=args.qwen_context,
                    benchmark_concurrency=batch_size,
                )
                for _ in range(batch_size)
            ]
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                if isinstance(outcome, Exception):
                    code = outcome.code if isinstance(outcome, BenchmarkRunError) else type(outcome).__name__
                    backend = outcome.backend if isinstance(outcome, BenchmarkRunError) else "UNAVAILABLE"
                    rows.append({
                        "fixture": fixture.name,
                        "backend": backend,
                        "benchmark_concurrency": batch_size,
                        "errors": [{"code": code}],
                    })
                else:
                    rows.append(outcome)
        results.extend(rows)
        summaries[fixture.name] = summarize(rows)
    payload = json.dumps({"results": results, "summary_by_fixture": summaries}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    asyncio.run(main())

