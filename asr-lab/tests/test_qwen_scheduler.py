from __future__ import annotations

import asyncio
import unittest

from asr_lab.qwen_scheduler import QwenDecodeScheduler, SchedulerError


class QwenSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.jobs = []
        self.results = []
        self.faults = []
        self.active = 0
        self.max_active = 0
        self.scheduler = QwenDecodeScheduler(
            execute=self.execute,
            on_result=self.on_result,
            on_fault=self.on_fault,
            max_pending_jobs=24,
            max_backlog_chunks=4,
            max_active_streams=6,
        )
        await self.scheduler.start()

    async def asyncTearDown(self):
        self.release.set()
        await self.scheduler.close()

    async def execute(self, key, job):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if not self.release.is_set():
                await self.release.wait()
            self.jobs.append((key, job.kind, job.pcm16le, job.cursor_end_samples))
            return {
                "decoded": True,
                "decode_wall_ms": 10.0,
                "decode_steps_delta": 1,
                "text": f"text-{key.connection_id}-{key.scheduler_revision}",
                "scheduler_key": {
                    "connection_id": key.connection_id,
                    "stream_id": key.stream_id,
                    "scheduler_revision": key.scheduler_revision,
                },
            }
        finally:
            self.active -= 1

    async def on_result(self, key, job, reply, metrics):
        self.results.append((key, job, reply, dict(metrics)))

    async def on_fault(self, connection_id, stream_id, code, details):
        self.faults.append((connection_id, stream_id, code, details))

    async def register(self, connection, stream, chunk=250):
        await self.scheduler.register(connection, stream, chunk)

    async def test_250ms_windows_split_100ms_frames_without_losing_pcm(self):
        self.release.set()
        await self.register("a", "stream-a")
        frame = b"\x01\x00" * 1600
        await self.scheduler.append_pcm("a", "stream-a", frame)
        await self.scheduler.append_pcm("a", "stream-a", frame)
        self.assertEqual(self.jobs, [])
        await self.scheduler.append_pcm("a", "stream-a", frame)
        await asyncio.wait_for(self.started.wait(), 1)
        await self._wait_for(lambda: len(self.jobs) == 1)
        self.assertEqual(len(self.jobs[0][2]), 4000 * 2)
        stream = self.scheduler.streams[("a", "stream-a")]
        self.assertEqual(len(stream.tail) // 2, 800)
        self.assertEqual(stream.accepted_total_samples, 4800)
        self.assertEqual(stream.dispatched_total_samples, 4000)

    async def test_500_and_1000ms_windows_match_100ms_pcm_frame_boundaries(self):
        self.release.set()
        frame = b"\x01\x00" * 1600
        for chunk_ms in (500, 1000):
            stream_name = f"stream-{chunk_ms}"
            await self.register(str(chunk_ms), stream_name, chunk_ms)
            for _ in range(chunk_ms // 100):
                await self.scheduler.append_pcm(str(chunk_ms), stream_name, frame)
            expected_jobs = len(self.jobs) + 1
            await self._wait_for(lambda: len(self.jobs) >= expected_jobs)
            self.assertEqual(len(self.jobs[-1][2]), chunk_ms * 16 * 2)

    async def test_round_robin_services_each_ready_stream_before_hot_stream_repeats(self):
        await self.register("A", "sa")
        for name in "BCDEF":
            await self.register(name, f"s{name.lower()}")
        pcm = b"\x02\x00" * 4000
        for name in "ABCDEF":
            await self.scheduler.append_pcm(name, f"s{name.lower()}", pcm)
        self.release.set()
        await self._wait_for(lambda: len(self.jobs) == 6)
        self.assertEqual([row[0].connection_id for row in self.jobs], list("ABCDEF"))
        self.assertEqual(self.max_active, 1)
        self.assertLessEqual(self.scheduler.active_jobs, 1)

    async def test_pending_and_active_stream_high_water_marks_cover_the_six_stream_group(self):
        await self.register("A", "sa")
        await self.scheduler.append_pcm("A", "sa", b"\x01\x00" * 4000)
        await asyncio.wait_for(self.started.wait(), 1)
        for name in "BCDEF":
            await self.register(name, f"s{name.lower()}")
            await self.scheduler.append_pcm(name, f"s{name.lower()}", b"\x02\x00" * 4000)
        for name in "ABCDEF":
            snapshot = self.scheduler.stream_snapshot(name, f"s{name.lower()}")
            self.assertEqual(snapshot["active_stream_count_max"], 6)
            self.assertEqual(snapshot["scheduler_pending_jobs_max"], 5)
        self.release.set()
        await self._wait_for(lambda: len(self.jobs) == 6)

    async def test_late_ready_stream_is_not_starved_by_a_hot_stream(self):
        await self.register("hot", "hot-stream")
        await self.register("quiet", "quiet-stream")
        await self.scheduler.append_pcm("hot", "hot-stream", b"\x09\x00" * 12_000)
        await asyncio.wait_for(self.started.wait(), 1)
        await self.scheduler.append_pcm("quiet", "quiet-stream", b"\x0a\x00" * 4_000)
        self.release.set()
        await self._wait_for(lambda: len(self.jobs) == 4)
        order = [row[0].connection_id for row in self.jobs]
        self.assertEqual(order.count("hot"), 3)
        self.assertEqual(order.count("quiet"), 1)
        self.assertLess(order.index("quiet"), 3)

    async def test_bounded_global_queue_fences_only_overflow_stream(self):
        self.scheduler.max_pending_jobs = 2
        await self.register("A", "sa")
        await self.register("B", "sb")
        pcm = b"\x03\x00" * (4000 * 3)
        with self.assertRaisesRegex(SchedulerError, "stream_scheduler_overrun"):
            await self.scheduler.append_pcm("A", "sa", pcm)
        self.assertEqual(self.faults[0][3]["rejected_audio_ms"], 750.0)
        self.assertEqual(self.faults[0][3]["reason"], "global_pending_job_limit")
        await self.scheduler.append_pcm("B", "sb", b"\x04\x00" * 4000)
        self.assertNotIn(("A", "sa"), self.scheduler.streams)
        self.assertIn(("B", "sb"), self.scheduler.streams)
        self.assertEqual(self.scheduler.snapshot()["qwen_scheduler_overrun_total"], 1)
        self.assertEqual(self.faults[0][2], "stream_scheduler_overrun")
        self.release.set()
        await self._wait_for(lambda: len(self.jobs) == 1)
        self.assertEqual(self.jobs[0][0].connection_id, "B")

    async def test_overrun_fault_captures_bounded_metrics_before_stream_fence(self):
        self.release.set()
        await self.register("private-connection", "private-stream")
        await self.scheduler.append_pcm("private-connection", "private-stream", b"\x01\x00" * 4000)
        await self._wait_for(lambda: self.scheduler.active_jobs == 0 and len(self.jobs) == 1)

        with self.assertRaisesRegex(SchedulerError, "stream_scheduler_overrun"):
            await self.scheduler.append_pcm(
                "private-connection", "private-stream", b"\x02\x00" * (4000 * 5),
            )

        details = self.faults[-1][3]
        metrics = details["terminal_metrics"]
        self.assertEqual(metrics["scheduler_wait_sample_count"], 1)
        self.assertEqual(metrics["decode_wall_sample_count"], 1)
        self.assertEqual(metrics["decode_wall_p50_ms"], 10.0)
        self.assertEqual(metrics["pending_jobs"], 0)
        self.assertEqual(metrics["backlog_audio_ms"], 0.0)
        self.assertEqual(metrics["accepted_audio_total_ms"], 250.0)
        self.assertEqual(metrics["dispatched_audio_total_ms"], 250.0)
        self.assertEqual(metrics["overrun_reason"], "per_stream_backlog_limit")
        self.assertEqual(metrics["overrun_limit_kind"], "backlog_ms")
        self.assertEqual(metrics["overrun_limit_value"], 1000.0)
        self.assertFalse({"connection_id", "stream_id", "pcm", "transcript", "secret"} & metrics.keys())

    async def test_disconnect_fences_active_result_and_new_stream_uses_new_identity(self):
        await self.register("client", "old-stream")
        await self.scheduler.append_pcm("client", "old-stream", b"\x05\x00" * 4000)
        await asyncio.wait_for(self.started.wait(), 1)
        active = self.scheduler._active_job
        self.assertIsNotNone(active)
        old_key = active[1]
        await self.scheduler.fence_stream("client", "old-stream", "disconnect")
        await self.register("client", "new-stream")
        self.release.set()
        await self._wait_for(lambda: self.scheduler.active_jobs == 0)
        self.assertEqual(self.results, [])
        self.assertFalse(self.scheduler.is_current(old_key))
        self.assertIn(("client", "new-stream"), self.scheduler.streams)

    async def test_finish_drains_residual_pcm_before_ordered_terminal_job(self):
        self.release.set()
        await self.register("client", "stream", 250)
        frame = b"\x06\x00" * 1600
        for _ in range(3):
            await self.scheduler.append_pcm("client", "stream", frame)
        finished = await self.scheduler.finish_stream("client", "stream")
        await self._wait_for(lambda: len(self.jobs) == 3)
        self.assertEqual([row[1] for row in self.jobs], ["push", "push", "finish"])
        self.assertEqual([len(row[2]) // 2 for row in self.jobs], [4000, 800, 0])
        self.assertEqual(sum(len(row[2]) // 2 for row in self.jobs), 4800)
        self.assertEqual(self.jobs[-1][0].scheduler_revision, finished.key.scheduler_revision)
        self.assertEqual(self.max_active, 1)

    async def test_metrics_separate_backlog_wait_decode_and_steps(self):
        self.release.set()
        await self.register("client", "stream", 250)
        await self.scheduler.append_pcm("client", "stream", b"\x07\x00" * 4000)
        await self._wait_for(lambda: len(self.results) == 1)
        metrics = self.results[0][3]
        self.assertIn("qwen_scheduler_wait_ms", metrics)
        self.assertIn("qwen_decode_backlog_ms", metrics)
        self.assertIn("stream_lag_ms", metrics)
        self.assertEqual(metrics["qwen_decode_steps_delta"], 1)
        self.assertEqual(self.scheduler.snapshot()["qwen_decode_steps_delta_total"], 1)

    async def test_decode_wall_over_chunk_is_measured_and_pcm_is_not_truncated(self):
        self.release.set()
        captured = []
        async def slow_execute(key, job):
            captured.append(job.pcm16le)
            return {"decode_wall_ms": 300.0, "decode_steps_delta": 1, "text": "x"}
        self.scheduler.execute = slow_execute
        await self.register("client", "slow", 250)
        pcm = b"\x08\x00" * 4000
        await self.scheduler.append_pcm("client", "slow", pcm)
        await self._wait_for(lambda: len(self.results) == 1)
        metrics = self.results[0][3]
        self.assertEqual(captured, [pcm])
        self.assertTrue(metrics["decode_overrun"])
        self.assertEqual(self.scheduler.snapshot()["qwen_decode_budget_overrun_total"], 1)

    async def _wait_for(self, predicate):
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.005)
        self.fail("timed out waiting for scheduler state")


if __name__ == "__main__":
    unittest.main()
