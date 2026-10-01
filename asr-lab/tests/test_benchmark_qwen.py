from __future__ import annotations

import asyncio
import json
import unittest

from benchmark.benchmark_ws import receive_fixture_results, summarize


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)

    async def recv(self):
        return next(self.messages)


class QwenBenchmarkSummaryTests(unittest.TestCase):
    def test_final_candidate_event_is_not_collected_as_production_transcript(self):
        ws = FakeWebSocket([
            json.dumps({"event": "partial_candidate", "candidate_only": True, "text": "hola"}),
            json.dumps({"event": "final_candidate", "candidate_only": True, "text": "hola mundo"}),
            json.dumps({"event": "flush_complete", "request_id": "flush-1"}),
        ])
        transcripts, partials, candidates, errors, _ = asyncio.run(
            receive_fixture_results(ws, "flush-1")
        )
        self.assertEqual(transcripts, [])
        self.assertEqual([item["event"] for item in partials], ["partial_candidate"])
        self.assertEqual([item["event"] for item in candidates], ["final_candidate"])
        self.assertEqual(errors, [])

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
        self.assertEqual(qwen["final_candidate_latency_ms"]["SERVER_EOS_TO_FINAL_CANDIDATE_MS"]["p50"], 170)
        self.assertEqual(qwen["final_candidate_latency_ms"]["CLIENT_EOS_TO_FINAL_CANDIDATE_MS"]["p50"], 205)
        self.assertEqual(qwen["final_candidate_latency_ms"]["TOTAL_AUDIO_END_TO_FINAL_CANDIDATE_MS"]["p50"], 230)
        self.assertAlmostEqual(qwen["FINAL_CANDIDATE_WER_MEAN"], 0.15)
        self.assertEqual(result["SERVER_EOS_TO_TRANSCRIPT_MS_P50"], 80)
        self.assertEqual(result["CLIENT_EOS_TO_TRANSCRIPT_MS_P50"], 105)
        self.assertEqual(result["TOTAL_AUDIO_END_TO_TRANSCRIPT_MS_P50"], 110)
        self.assertAlmostEqual(result["wer_mean"], 0.05)


if __name__ == "__main__":
    unittest.main()
