from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.benchmark_ws import (
    max_qwen_event_metric,
    receive_fixture_results,
    resolve_concurrency_levels,
    run_one,
    summarize,
    _worker_warmup_chunk_ms,
)


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)

    async def recv(self):
        return next(self.messages)


class FakeHttpResponse:
    def __init__(self, value):
        self.body = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *args):
        return self.body.read(*args)


class QwenBenchmarkSummaryTests(unittest.TestCase):
    def test_final_candidate_event_is_not_collected_as_production_transcript(self):
        ws = FakeWebSocket([
            json.dumps({"event": "partial_candidate", "candidate_only": True, "text": "hola", "QWEN_SCHEDULER_WAIT_MS": 12, "QWEN_DECODE_BACKLOG_MS": 250, "STREAM_LAG_MS": 500, "PENDING_DECODE_COUNT": 2, "ACTIVE_STREAM_COUNT": 6}),
            json.dumps({"event": "final_candidate", "candidate_only": True, "text": "hola mundo", "DECODE_OVERRUN": True}),
            json.dumps({"event": "flush_complete", "request_id": "flush-1"}),
        ])
        transcripts, partials, candidates, errors, _ = asyncio.run(
            receive_fixture_results(ws, "flush-1")
        )
        self.assertEqual(transcripts, [])
        self.assertEqual([item["event"] for item in partials], ["partial_candidate"])
        self.assertEqual([item["event"] for item in candidates], ["final_candidate"])
        self.assertEqual(errors, [])
        self.assertEqual(partials[0]["PENDING_DECODE_COUNT"], 2)
        self.assertEqual(partials[0]["ACTIVE_STREAM_COUNT"], 6)
        self.assertTrue(candidates[0]["DECODE_OVERRUN"])

    def test_candidate_metrics_are_isolated_from_production_transcript_kpis(self):
        rows = [
            {
                "backend": "qwen3_asr",
                "event": "final_candidate",
                "candidate_only": True,
                "truth_status": "candidate_only",
                "audio_duration_ms": 4_000,
                "benchmark_concurrency": 2,
                "SERVER_EOS_TO_TRANSCRIPT_MS": 120,
                "CLIENT_EOS_TO_TRANSCRIPT_MS": 155,
                "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS": 180,
                "wer": 0.1,
                "FINAL_CANDIDATE_WER": 0.1,
                "errors": [],
                "segments": [{
                    "SERVER_EOS_TO_TRANSCRIPT_MS": 120,
                    "candidate_only": True,
                }],
                "streaming": {
                    "partial_count": 3,
                    "FIRST_PARTIAL_MS": 500,
                    "CLIENT_FIRST_PARTIAL_MS": 540,
                    "QWEN_CUMULATIVE_DECODE_WALL_MS": 900,
                    "QWEN_STREAM_RTF": 0.225,
                    "QWEN_SCHEDULER_WAIT_MS_P50": 20,
                    "QWEN_SCHEDULER_WAIT_JOB_SAMPLE_COUNT": 6,
                    "QWEN_SCHEDULER_WAIT_MS_CANDIDATE_EVENT_SAMPLES": [12],
                    "QWEN_SCHEDULER_WAIT_STREAM_P50_MS": 20,
                    "QWEN_SCHEDULER_WAIT_STREAM_P95_MS": 50,
                    "QWEN_SCHEDULER_WAIT_STREAM_MAX_MS": 75,
                    "QWEN_DECODE_CALL_WALL_MS_P50": 130,
                    "QWEN_DECODE_CALL_WALL_MS_P95": 160,
                    "QWEN_DECODE_CALL_WALL_JOB_SAMPLE_COUNT": 6,
                    "QWEN_DECODE_CALL_WALL_MS_CANDIDATE_EVENT_SAMPLES": [80],
                    "QWEN_DECODE_WALL_STREAM_P50_MS": 130,
                    "QWEN_DECODE_WALL_STREAM_P95_MS": 160,
                    "QWEN_DECODE_BACKLOG_MS_MAX": 700,
                    "STREAM_LAG_MS_MAX": 900,
                    "PENDING_DECODE_COUNT_MAX": 3,
                    "ACTIVE_STREAM_COUNT_MAX": 2,
                    "DECODE_OVERRUN_COUNT": 1,
                    "SERVER_EOS_TO_FINAL_CANDIDATE_MS": 120,
                },
                "SERVER_EOS_TO_FINAL_CANDIDATE_MS": 120,
                "CLIENT_EOS_TO_FINAL_CANDIDATE_MS": 155,
                "TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS": 180,
            },
            {
                "backend": "qwen3_asr",
                "candidate_only": True,
                "truth_status": "candidate_only",
                "audio_duration_ms": 12_000,
                "benchmark_concurrency": 2,
                "SERVER_EOS_TO_TRANSCRIPT_MS": 220,
                "CLIENT_EOS_TO_TRANSCRIPT_MS": 255,
                "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS": 280,
                "wer": 0.2,
                "FINAL_CANDIDATE_WER": 0.2,
                "errors": [{"code": "stream_capacity_exceeded"}],
                "segments": [{
                    "SERVER_EOS_TO_TRANSCRIPT_MS": 220,
                    "candidate_only": True,
                }],
                "streaming": {
                    "partial_count": 4,
                    "FIRST_PARTIAL_MS": 650,
                    "CLIENT_FIRST_PARTIAL_MS": 700,
                    "QWEN_CUMULATIVE_DECODE_WALL_MS": 2_400,
                    "QWEN_STREAM_RTF": 0.2,
                    "QWEN_SCHEDULER_WAIT_MS_P50": 40,
                    "QWEN_SCHEDULER_WAIT_JOB_SAMPLE_COUNT": 8,
                    "QWEN_SCHEDULER_WAIT_MS_CANDIDATE_EVENT_SAMPLES": [30, 50],
                    "QWEN_SCHEDULER_WAIT_STREAM_P50_MS": 40,
                    "QWEN_SCHEDULER_WAIT_STREAM_P95_MS": 60,
                    "QWEN_SCHEDULER_WAIT_STREAM_MAX_MS": 90,
                    "QWEN_DECODE_CALL_WALL_MS_P50": 300,
                    "QWEN_DECODE_CALL_WALL_MS_P95": 310,
                    "QWEN_DECODE_CALL_WALL_JOB_SAMPLE_COUNT": 8,
                    "QWEN_DECODE_CALL_WALL_MS_CANDIDATE_EVENT_SAMPLES": [290, 310],
                    "QWEN_DECODE_WALL_STREAM_P50_MS": 300,
                    "QWEN_DECODE_WALL_STREAM_P95_MS": 310,
                    "QWEN_DECODE_BACKLOG_MS_MAX": 1400,
                    "STREAM_LAG_MS_MAX": 1800,
                    "PENDING_DECODE_COUNT_MAX": 5,
                    "ACTIVE_STREAM_COUNT_MAX": 2,
                    "SCHEDULER_FENCE_OVERRUN_COUNT": 1,
                    "SERVER_EOS_TO_FINAL_CANDIDATE_MS": 220,
                },
                "SERVER_EOS_TO_FINAL_CANDIDATE_MS": 220,
                "CLIENT_EOS_TO_FINAL_CANDIDATE_MS": 255,
                "TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS": 280,
            },
            {
                "backend": "parakeet",
                "candidate_only": False,
                "audio_duration_ms": 4_000,
                "SERVER_EOS_TO_TRANSCRIPT_MS": 80,
                "CLIENT_EOS_TO_TRANSCRIPT_MS": 105,
                "TOTAL_AUDIO_END_TO_TRANSCRIPT_MS": 110,
                "wer": 0.05,
                "segments": [{
                    "SERVER_EOS_TO_TRANSCRIPT_MS": 80,
                    "candidate_only": False,
                }],
                "errors": [],
            },
        ]

        result = summarize(rows)
        qwen = result["qwen_streaming"]
        self.assertEqual(qwen["runs"], 2)
        self.assertEqual(qwen["partial_count_total"], 7)
        self.assertEqual(qwen["decode_cost_by_duration_bucket"]["5s"]["cumulative_decode_wall_ms_p50"], 900)
        self.assertEqual(qwen["decode_cost_by_duration_bucket"]["15s"]["cumulative_decode_wall_ms_p50"], 2400)
        self.assertEqual(qwen["concurrency_runs"]["2"]["runs"], 2)
        self.assertEqual(qwen["concurrency_runs"]["2"]["capacity_errors"], 1)
        self.assertEqual(qwen["concurrency_runs"]["2"]["per_stream_scheduler_wait_job_p50_ms_median"], 30)
        self.assertEqual(qwen["concurrency_runs"]["2"]["per_stream_scheduler_wait_job_p95_ms_p95"], 60)
        self.assertEqual(qwen["concurrency_runs"]["2"]["scheduler_wait_job_sample_count_total"], 14)
        self.assertEqual(qwen["concurrency_runs"]["2"]["scheduler_wait_ms_candidate_event_p95"], 50)
        self.assertEqual(qwen["concurrency_runs"]["2"]["per_stream_decode_wall_job_p95_ms_p95"], 310)
        self.assertEqual(qwen["concurrency_runs"]["2"]["decode_wall_job_sample_count_total"], 14)
        self.assertEqual(qwen["concurrency_runs"]["2"]["decode_call_wall_ms_candidate_event_p95"], 310)
        self.assertEqual(qwen["concurrency_runs"]["2"]["backlog_audio_ms_max"], 1400)
        self.assertEqual(qwen["concurrency_runs"]["2"]["pending_decode_count_max"], 5)
        self.assertEqual(qwen["concurrency_runs"]["2"]["active_stream_count_max"], 2)
        self.assertEqual(qwen["final_candidate_latency_ms"]["SERVER_EOS_TO_FINAL_CANDIDATE_MS"]["p50"], 170)
        self.assertEqual(qwen["final_candidate_latency_ms"]["CLIENT_EOS_TO_FINAL_CANDIDATE_MS"]["p50"], 205)
        self.assertEqual(qwen["final_candidate_latency_ms"]["TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS"]["p50"], 230)
        self.assertAlmostEqual(qwen["FINAL_CANDIDATE_WER_MEAN"], 0.15)
        self.assertEqual(result["SERVER_EOS_TO_TRANSCRIPT_MS_P50"], 80)
        self.assertEqual(result["CLIENT_EOS_TO_TRANSCRIPT_MS_P50"], 105)
        self.assertEqual(result["TOTAL_AUDIO_END_TO_TRANSCRIPT_MS_P50"], 110)
        self.assertAlmostEqual(result["wer_mean"], 0.05)
        self.assertEqual(qwen["scheduler"]["per_stream_scheduler_wait_job_p50_ms"]["p50"], 30)
        self.assertEqual(qwen["scheduler"]["per_stream_scheduler_wait_job_p95_ms"]["p95"], 60)
        self.assertEqual(qwen["scheduler"]["per_stream_decode_wall_job_p95_ms"]["max"], 310)
        self.assertEqual(qwen["scheduler"]["scheduler_wait_ms_candidate_event_samples"]["p50"], 30)
        self.assertEqual(qwen["scheduler"]["decode_call_wall_ms_candidate_event_samples"]["p95"], 310)
        self.assertEqual(qwen["scheduler"]["scheduler_wait_job_sample_count_per_run"]["max"], 8)
        self.assertEqual(qwen["scheduler"]["pending_decode_count_high_water"]["max"], 5)
        self.assertEqual(qwen["scheduler"]["active_stream_count_high_water"]["max"], 2)
        self.assertEqual(qwen["scheduler"]["backlog_audio_ms"]["max"], 1400)
        self.assertEqual(qwen["decode_overrun_count"], 1)
        self.assertEqual(qwen["scheduler_fence_overrun_count"], 1)

    def test_concurrency_matrix_is_exactly_the_approved_experimental_levels(self):
        self.assertEqual(resolve_concurrency_levels(3), [3])
        self.assertEqual(resolve_concurrency_levels(3, matrix=True), [1, 2, 4, 6])

    def test_scheduler_high_water_marks_are_not_lost_between_sparse_events(self):
        events = [
            {"QWEN_DECODE_BACKLOG_MS": 80, "QWEN_DECODE_BACKLOG_MAX_MS": 310},
            {"STREAM_LAG_MS": 120, "STREAM_LAG_MAX_MS": 475},
        ]
        self.assertEqual(
            max_qwen_event_metric(events, "QWEN_DECODE_BACKLOG_MS", "QWEN_DECODE_BACKLOG_MAX_MS"),
            310,
        )
        self.assertEqual(max_qwen_event_metric(events, "STREAM_LAG_MS", "STREAM_LAG_MAX_MS"), 475)

    def test_all_decode_jobs_are_canonical_when_candidate_event_samples_are_sparse(self):
        summary = summarize([{
            "backend": "qwen3_asr", "run_status": "completed", "benchmark_concurrency": 1,
            "candidate_only": True, "truth_status": "candidate_only", "errors": [],
            "streaming": {
                "partial_count": 1,
                "QWEN_SCHEDULER_WAIT_MS_P50": 11,
                "QWEN_SCHEDULER_WAIT_MS_P95": 27,
                "QWEN_SCHEDULER_WAIT_STREAM_P95_MS": 27,
                "QWEN_SCHEDULER_WAIT_JOB_SAMPLE_COUNT": 5,
                "QWEN_SCHEDULER_WAIT_MS_CANDIDATE_EVENT_SAMPLES": [11],
                "QWEN_DECODE_CALL_WALL_MS_P50": 90,
                "QWEN_DECODE_CALL_WALL_MS_P95": 140,
                "QWEN_DECODE_CALL_WALL_JOB_SAMPLE_COUNT": 5,
                "QWEN_DECODE_CALL_WALL_MS_CANDIDATE_EVENT_SAMPLES": [90],
            },
        }])
        qwen = summary["qwen_streaming"]
        self.assertEqual(qwen["successful_runs"], 1)
        self.assertEqual(qwen["scheduler"]["scheduler_wait_job_sample_count_per_run"]["max"], 5)
        self.assertEqual(qwen["scheduler"]["scheduler_wait_ms_candidate_event_samples"]["max"], 11)
        self.assertEqual(qwen["scheduler"]["per_stream_scheduler_wait_job_p95_ms"]["max"], 27)
        self.assertEqual(qwen["scheduler"]["decode_wall_job_sample_count_per_run"]["max"], 5)
        self.assertEqual(qwen["scheduler"]["per_stream_decode_wall_job_p95_ms"]["max"], 140)

    def test_terminal_error_fences_late_candidates_and_restarted_streams(self):
        async def scenario():
            ws = FakeWebSocket([
                json.dumps({"event": "partial_candidate", "stream_id": "stream-a", "text": "old", "final": False}),
                json.dumps({"event": "error", "code": "stream_scheduler_overrun"}),
                json.dumps({"event": "final_candidate", "stream_id": "stream-a", "text": "late", "final": True}),
                json.dumps({"event": "stream_started", "stream_id": "stream-b"}),
                json.dumps({"event": "final_candidate", "stream_id": "stream-b", "text": "cross-stream", "final": True}),
                json.dumps({"event": "flush_complete", "request_id": "flush-1"}),
            ])
            terminal = asyncio.Event()
            return await receive_fixture_results(
                ws, "flush-1", expected_stream_id="stream-a", terminal_event=terminal,
            ), terminal

        (transcripts, partials, candidates, errors, _), terminal = asyncio.run(scenario())
        self.assertEqual(transcripts, [])
        self.assertEqual(partials, [])
        self.assertEqual(candidates, [])
        self.assertTrue(terminal.is_set())
        self.assertEqual(errors[0]["code"], "stream_scheduler_overrun")
        self.assertEqual(errors[1]["code"], "unexpected_stream_restart")

    def test_overrun_without_final_candidate_keeps_wer_null_and_all_job_metrics(self):
        class FakeRunWebSocket:
            def __init__(self):
                self.messages = asyncio.Queue()
                self.audio_sends = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def ping(self):
                future = asyncio.get_running_loop().create_future()
                future.set_result(None)
                return future

            async def recv(self):
                return await self.messages.get()

            async def send(self, raw):
                message = json.loads(raw)
                if message.get("event") == "stream_start":
                    await self.messages.put(json.dumps({
                        "event": "stream_started", "stream_id": "stream-fixture",
                        "model_chunk_ms": 250, "audio_push_interval_ms": 100,
                        "effective_max_backlog_ms": 1000, "qwen_max_backlog_chunks": 4,
                        "FIRST_STREAM_INIT_MS": 7, "FIRST_STREAM_STATE_INIT_WALL_MS": 2,
                        "FIRST_STREAM_INIT_RPC_OVERHEAD_MS": 5,
                    }))
                elif "audio" in message:
                    self.audio_sends += 1
                    if self.audio_sends <= 2:
                        await self.messages.put(json.dumps({
                            "event": "partial_candidate", "stream_id": "stream-fixture",
                            "text": f"partial-{self.audio_sends}", "candidate_only": True,
                            "truth_status": "candidate_only", "provisional": True, "final": False,
                            "QWEN_SCHEDULER_WAIT_MS": 8,
                            "QWEN_DECODE_CALL_WALL_MS": 90,
                            "FIRST_AUDIO_TO_FIRST_DECODE_READY_MS": 10,
                            "FIRST_SCHEDULER_WAIT_MS": 3,
                            "FIRST_AUDIO_TO_FIRST_DECODE_START_MS": 13,
                            "EPOCH_FIRST_DECODE_WALL_MS": 90,
                            "QWEN_DECODE_SLO_TARGET_MS": 100,
                            "QWEN_DECODE_SLO_VIOLATION": False,
                            "SERVER_FIRST_PARTIAL_MS": 13,
                            "FIRST_PARTIAL_MS": 13,
                            "QWEN_CUMULATIVE_DECODE_WALL_MS": 90,
                            "QWEN_STREAM_RTF": 0.18,
                        }))
                    elif self.audio_sends == 3:
                        await self.messages.put(json.dumps({
                            "event": "error", "code": "stream_scheduler_overrun",
                            "scheduler": {
                                "reason": "per_stream_backlog_limit",
                                "accepted_audio_ms": 500,
                                "rejected_audio_ms": 100,
                                "terminal_metrics": {
                                    "scheduler_wait_p50_ms": 12,
                                    "scheduler_wait_p95_ms": 18,
                                    "scheduler_wait_max_ms": 22,
                                    "scheduler_wait_sample_count": 3,
                                    "decode_wall_p50_ms": 90,
                                    "decode_wall_p95_ms": 110,
                                    "decode_wall_max_ms": 120,
                                    "decode_wall_sample_count": 3,
                                    "pending_jobs": 4,
                                    "stream_pending_jobs": 3,
                                    "active_stream_count": 6,
                                    "backlog_audio_ms": 750,
                                    "stream_lag_ms": 875,
                                    "max_backlog_audio_ms": 900,
                                    "max_stream_lag_ms": 1100,
                                    "accepted_audio_total_ms": 500,
                                    "dispatched_audio_total_ms": 250,
                                    "scheduler_metric_history_limit": 256,
                                    "overrun_reason": "per_stream_backlog_limit",
                                    "overrun_limit_kind": "backlog_ms",
                                    "overrun_limit_value": 1000,
                                    "stream_id": "must-not-export",
                                    "connection_id": "must-not-export",
                                    "transcript": "must-not-export",
                                    "pcm": "must-not-export",
                                    "secret": "must-not-export",
                                },
                            },
                        }))
                elif message.get("event") == "flush":
                    await self.messages.put(json.dumps({
                        "event": "flush_complete", "request_id": message["request_id"],
                    }))

        ws = FakeRunWebSocket()

        class HttpResponse(FakeHttpResponse):
            pass

        fake_websockets = SimpleNamespace(connect=lambda *_args, **_kwargs: ws)
        readiness_response = HttpResponse({"ready": True})
        health_response = HttpResponse({
            "backend": "qwen3_asr", "model_id": "Qwen/Qwen3-ASR-1.7B",
            "model_revision": "revision-fixture", "runtime_provenance": {
                "qwen_asr_version": "0.0.6", "vllm_version": "0.14.0",
                "transformers_version": "fixture-transformers", "torch_version": "fixture-torch",
                "torch_cuda_version": "fixture-cuda", "experiment_config": {"warmup_chunk_ms": 1000},
            },
            "worker_metrics": [{
                "model_load_ms": 42, "warmup_ms": 5,
                "experiment_config": {"warmup_chunk_ms": 1000},
            }],
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture.wav"
            with wave.open(str(fixture), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\x00\x00" * 6400)
            fixture.with_suffix(".txt").write_text("hello world", encoding="utf-8")

            with (
                patch.dict(sys.modules, {"websockets": fake_websockets}),
                patch("urllib.request.urlopen", side_effect=[readiness_response, health_response]),
            ):
                row = asyncio.run(run_one("ws://localhost/asr/ws", "token", fixture, qwen_chunk_ms=250))

        self.assertEqual(row["run_status"], "failed")
        self.assertIsNone(row["final_candidate"])
        self.assertIsNone(row["FINAL_CANDIDATE_WER"])
        self.assertTrue(row["readiness_at_start"]["ready"])
        self.assertEqual(row["MODEL_STATE_AT_RUN"], "warm_model_ready")
        self.assertEqual(row["model_revision"], "revision-fixture")
        self.assertEqual(row["model_load_ms"], [42])
        self.assertEqual(row["warmup_ms"], [5])
        self.assertEqual(row["warmup_chunk_ms"], [1000])
        self.assertEqual(row["model_id"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(row["runtime_provenance"]["qwen_asr_version"], "0.0.6")
        self.assertEqual(row["runtime_provenance"]["vllm_version"], "0.14.0")
        self.assertEqual(row["runtime_provenance"]["transformers_version"], "fixture-transformers")
        self.assertEqual(row["runtime_provenance"]["torch_version"], "fixture-torch")
        self.assertEqual(row["runtime_provenance"]["torch_cuda_version"], "fixture-cuda")
        self.assertEqual(row["streaming"]["effective_max_backlog_ms"], 1000)
        self.assertEqual(row["streaming"]["qwen_max_backlog_chunks"], 4)
        self.assertEqual(row["streaming"]["FIRST_STREAM_INIT_MS"], 7)
        self.assertEqual(row["streaming"]["FIRST_STREAM_INIT_RPC_OVERHEAD_MS"], 5)
        self.assertEqual(row["streaming"]["FIRST_AUDIO_TO_FIRST_DECODE_START_MS"], 13)
        self.assertEqual(row["streaming"]["EPOCH_FIRST_DECODE_WALL_MS"], 90)
        self.assertFalse(row["streaming"]["QWEN_DECODE_SLO_VIOLATION"])
        self.assertEqual(row["streaming"]["SERVER_FIRST_PARTIAL_MS"], 13)
        self.assertEqual(row["streaming"]["FIRST_PARTIAL_MS"], 13)
        self.assertEqual(row["streaming"]["QWEN_CUMULATIVE_DECODE_WALL_MS"], 90)
        self.assertEqual(row["streaming"]["QWEN_STREAM_RTF"], 0.18)
        self.assertIsInstance(row["streaming"]["CLIENT_FIRST_PARTIAL_MS"], (int, float))
        self.assertGreaterEqual(row["streaming"]["CLIENT_FIRST_PARTIAL_MS"], 0)
        self.assertEqual(row["streaming"]["QWEN_SCHEDULER_WAIT_MS_P50"], 12)
        self.assertEqual(row["streaming"]["QWEN_SCHEDULER_WAIT_MS_P95"], 18)
        self.assertEqual(row["streaming"]["QWEN_SCHEDULER_WAIT_JOB_SAMPLE_COUNT"], 3)
        self.assertEqual(row["streaming"]["QWEN_DECODE_CALL_WALL_MS_P50"], 90)
        self.assertEqual(row["streaming"]["QWEN_DECODE_CALL_WALL_MS_P95"], 110)
        self.assertEqual(row["streaming"]["QWEN_DECODE_CALL_WALL_JOB_SAMPLE_COUNT"], 3)
        self.assertEqual(row["streaming"]["QWEN_TERMINAL_PENDING_JOBS"], 4)
        self.assertEqual(row["streaming"]["QWEN_TERMINAL_BACKLOG_AUDIO_MS"], 750)
        self.assertEqual(row["streaming"]["QWEN_ACCEPTED_AUDIO_MS_TOTAL"], 500)
        self.assertEqual(row["streaming"]["QWEN_DISPATCHED_AUDIO_MS_TOTAL"], 250)

        exported_error = row["errors"][0]
        self.assertEqual(exported_error["scheduler"]["terminal_metrics"]["overrun_reason"], "per_stream_backlog_limit")
        self.assertFalse({"stream_id", "connection_id", "transcript", "pcm", "secret"} & exported_error["scheduler"]["terminal_metrics"].keys())
        summary = summarize([row])["qwen_streaming"]
        self.assertIsNone(summary["FINAL_CANDIDATE_WER_MEAN"])
        self.assertIsNone(summary["FINAL_CANDIDATE_WER_P50"])
        self.assertIsNone(summary["FINAL_CANDIDATE_WER_P95"])
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(summary["scheduler"]["scheduler_wait_job_sample_count_per_run"]["max"], 3)
        self.assertEqual(summary["scheduler"]["per_stream_decode_wall_job_p95_ms"]["max"], 110)

        successful_row = dict(row)
        successful_row.update({
            "run_status": "completed",
            "final_candidate": "hello world",
            "FINAL_CANDIDATE_WER": 0.25,
            "errors": [],
        })
        failed_row_with_numeric_wer = dict(row)
        failed_row_with_numeric_wer["FINAL_CANDIDATE_WER"] = 1.0

        wer_summary = summarize([failed_row_with_numeric_wer, successful_row])["qwen_streaming"]
        self.assertEqual(wer_summary["FINAL_CANDIDATE_WER_MEAN"], 0.25)
        self.assertEqual(wer_summary["FINAL_CANDIDATE_WER_P50"], 0.25)
        self.assertEqual(wer_summary["FINAL_CANDIDATE_WER_P95"], 0.25)
        self.assertEqual(wer_summary["failed_runs"], 1)
        self.assertEqual(wer_summary["scheduler"]["scheduler_wait_job_sample_count_per_run"]["max"], 3)

    def test_warmup_chunk_top_level_legacy_schema_remains_supported(self):
        self.assertEqual(_worker_warmup_chunk_ms({"warmup_chunk_ms": 750}), 750)
        self.assertEqual(_worker_warmup_chunk_ms({"experiment_config": {"warmup_chunk_ms": 500}}), 500)
        self.assertIsNone(_worker_warmup_chunk_ms({}))


if __name__ == "__main__":
    unittest.main()
