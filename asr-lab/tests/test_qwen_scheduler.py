from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from asr_lab.qwen_scheduler import QwenDecodeScheduler, RequestKey, SchedulerError


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
        await asyncio.wait_for(self.started.wait(), 1)
        await self._wait_for(lambda: len(self.jobs) == 1)
        self.assertEqual(len(self.jobs[0][2]) // 2, 1600)
        self.assertEqual(self.jobs[0][2], frame)
        for _ in range(3):
            await self.scheduler.append_pcm("a", "stream-a", frame)
        await self._wait_for(lambda: len(self.jobs) == 2)
        self.assertEqual([len(job[2]) // 2 for job in self.jobs], [1600, 4000])
        stream = self.scheduler.streams[("a", "stream-a")]
        self.assertEqual(len(stream.tail) // 2, 800)
        self.assertEqual(stream.accepted_total_samples, 6400)
        self.assertEqual(stream.dispatched_total_samples, 5600)
        self.assertEqual(b"".join(job[2] for job in self.jobs) + bytes(stream.tail), frame * 4)

    async def test_500_and_1000ms_windows_match_100ms_pcm_frame_boundaries(self):
        self.release.set()
        frame = b"\x01\x00" * 1600
        for chunk_ms in (500, 1000):
            stream_name = f"stream-{chunk_ms}"
            await self.register(str(chunk_ms), stream_name, chunk_ms)
            first_job_count = len(self.jobs)
            await self.scheduler.append_pcm(str(chunk_ms), stream_name, frame)
            await self._wait_for(lambda: len(self.jobs) >= first_job_count + 1)
            self.assertEqual(len(self.jobs[-1][2]) // 2, 1600)
            for _ in range(chunk_ms // 100):
                await self.scheduler.append_pcm(str(chunk_ms), stream_name, frame)
            await self._wait_for(lambda: len(self.jobs) >= first_job_count + 2)
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

    async def test_50ms_round_robin_gives_competing_stream_turn_between_split_jobs(self):
        await self.register("A", "sa", 50)
        await self.register("B", "sb", 50)
        await self.scheduler.append_pcm("A", "sa", b"\x11\x00" * 1600)
        await self.scheduler.append_pcm("B", "sb", b"\x22\x00" * 800)
        self.release.set()
        await self._wait_for(lambda: len(self.jobs) == 3)
        self.assertEqual([row[0].connection_id for row in self.jobs], ["A", "B", "A"])
        self.assertEqual([len(row[2]) // 2 for row in self.jobs], [800, 800, 800])
        self.assertEqual([row[0].scheduler_revision for row in self.jobs], [1, 1, 2])
        self.assertEqual(self.max_active, 1)

    async def test_50ms_backlog_overrun_fences_only_owner_and_preserves_competitor(self):
        await self.register("A", "sa", 50)
        await self.register("B", "sb", 50)
        frame_a = b"\x31\x00" * 1600
        frame_b = b"\x42\x00" * 1600
        await self.scheduler.append_pcm("A", "sa", frame_a)
        await asyncio.wait_for(self.started.wait(), 1)
        await self.scheduler.append_pcm("A", "sa", frame_a)
        await self.scheduler.append_pcm("B", "sb", frame_b)
        with self.assertRaisesRegex(SchedulerError, "stream_scheduler_overrun"):
            await self.scheduler.append_pcm("A", "sa", frame_a)
        self.assertNotIn(("A", "sa"), self.scheduler.streams)
        self.assertIn(("B", "sb"), self.scheduler.streams)
        self.release.set()
        await self._wait_for(lambda: len([row for row in self.jobs if row[0].connection_id == "B"]) == 2)
        self.assertEqual(self.max_active, 1)
        self.assertEqual(self.faults[-1][2], "stream_scheduler_overrun")

    async def test_wrong_scheduler_revision_is_not_current_on_same_active_stream(self):
        await self.register("revision", "stream-revision", 50)
        await self.scheduler.append_pcm(
            "revision", "stream-revision", b"\x01\x00" * 800
        )
        await asyncio.wait_for(self.started.wait(), 1)

        stream = self.scheduler.streams[("revision", "stream-revision")]
        current_key = stream.active_key
        self.assertIsNotNone(current_key)
        self.assertTrue(self.scheduler.is_current(current_key))

        mismatched_revision = RequestKey(
            current_key.connection_id,
            current_key.stream_id,
            current_key.scheduler_revision + 1,
        )
        self.assertFalse(self.scheduler.is_current(mismatched_revision))

        self.release.set()
        await self._wait_for(lambda: len(self.jobs) == 1)

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
        self.assertEqual([len(row[2]) // 2 for row in self.jobs], [1600, 3200, 0])
        self.assertEqual(sum(len(row[2]) // 2 for row in self.jobs), 4800)
        self.assertEqual(self.jobs[-1][0].scheduler_revision, finished.key.scheduler_revision)
        self.assertEqual(self.max_active, 1)

    async def test_first_subwindow_audio_is_pushed_before_eos_once_and_unpadded(self):
        self.release.set()
        await self.register("short", "stream", 250)
        pcm = b"\x16\x00" * 800

        await self.scheduler.append_pcm("short", "stream", pcm)
        await self._wait_for(lambda: len(self.jobs) == 1)

        self.assertEqual(self.jobs[0][1], "push")
        self.assertEqual(self.jobs[0][2], pcm)
        self.assertEqual(self.jobs[0][3], 800)
        stream = self.scheduler.streams[("short", "stream")]
        self.assertEqual(bytes(stream.tail), b"")

        await self.scheduler.finish_stream("short", "stream")
        await self._wait_for(lambda: len(self.jobs) == 2)
        self.assertEqual([job[1] for job in self.jobs], ["push", "finish"])

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


    async def test_50ms_jobs_are_distinct_and_150ms_tail_cursors_are_exact(self):
        self.release.set()
        await self.register("small", "s50", 50)
        samples = [((i % 30000) - 15000) for i in range(1600)]
        import struct
        pcm = struct.pack("<1600h", *samples)
        await self.scheduler.append_pcm("small", "s50", pcm)
        await self._wait_for(lambda: len(self.jobs) == 2)
        self.assertEqual([len(row[2]) // 2 for row in self.jobs], [800, 800])
        self.assertEqual([row[0].scheduler_revision for row in self.jobs], [1, 2])
        self.assertEqual(b"".join(row[2] for row in self.jobs), pcm)
        self.assertEqual([row[3] for row in self.jobs], [800, 1600])
        await self.register("mid", "s150", 150)
        frame = struct.pack("<1600h", *[i - 800 for i in range(1600)])
        start = len(self.jobs)
        await self.scheduler.append_pcm("mid", "s150", frame)
        await self._wait_for(lambda: len(self.jobs) == start + 1)
        await self.scheduler.append_pcm("mid", "s150", frame)
        await self.scheduler.append_pcm("mid", "s150", frame)
        await self._wait_for(lambda: len(self.jobs) == start + 2)
        mid_jobs = self.jobs[start:]
        stream = self.scheduler.streams[("mid", "s150")]
        self.assertEqual([len(row[2]) // 2 for row in mid_jobs], [1600, 2400])
        self.assertEqual([row[3] for row in mid_jobs], [1600, 4000])
        self.assertEqual(len(stream.tail) // 2, 800)
        self.assertEqual(b"".join(row[2] for row in mid_jobs) + bytes(stream.tail), frame * 3)

    async def test_100_150_200_250ms_windows_preserve_exact_pcm_through_eos(self):
        self.release.set()
        import struct
        frames = [struct.pack("<1600h", *[frame * 5000 + i for i in range(1600)]) for frame in range(3)]
        expected_samples = {
            100: [1600, 1600, 1600],
            150: [1600, 2400, 800],
            200: [1600, 3200],
            250: [1600, 3200],
        }
        for chunk_ms, sample_counts in expected_samples.items():
            connection, stream = f"eos-{chunk_ms}", f"s-{chunk_ms}"
            before = len(self.jobs)
            await self.register(connection, stream, chunk_ms)
            for frame in frames:
                await self.scheduler.append_pcm(connection, stream, frame)
            await self.scheduler.finish_stream(connection, stream)
            jobs = [job for job in self.jobs[before:] if job[0].connection_id == connection]
            pushes = [job for job in jobs if job[1] == "push"]
            finishes = [job for job in jobs if job[1] == "finish"]
            self.assertEqual([len(job[2]) // 2 for job in pushes], sample_counts, f"chunk={chunk_ms}")
            self.assertEqual(len(finishes), 1, f"chunk={chunk_ms}")
            self.assertEqual(jobs[-1][1], "finish", f"chunk={chunk_ms}")
            self.assertEqual(b"".join(job[2] for job in pushes), b"".join(frames), f"chunk={chunk_ms}")
            self.assertEqual(sum(len(job[2]) // 2 for job in pushes), 4800, f"chunk={chunk_ms}")

    async def test_invalid_multi_step_reply_fences_only_its_owner(self):
        self.release.set()
        async def invalid_for_a(key, job):
            return {"decode_wall_ms": 10, "decode_steps_delta": 2 if key.connection_id == "A" else 1}
        self.scheduler.execute = invalid_for_a
        await self.register("A", "sa", 50)
        await self.register("B", "sb", 50)
        frame = b"\x11\x00" * 800
        await self.scheduler.append_pcm("A", "sa", frame)
        await self.scheduler.append_pcm("B", "sb", frame)
        await self._wait_for(lambda: len(self.results) + len(self.faults) >= 2)
        self.assertNotIn(("A", "sa"), self.scheduler.streams)
        self.assertIn(("B", "sb"), self.scheduler.streams)
        self.assertEqual(self.faults[0][2], "invalid_worker_response")
        self.assertEqual([row[0].connection_id for row in self.results], ["B"])


    async def test_first_decode_ready_wait_start_clocks_reconcile_without_sleep(self):
        from types import SimpleNamespace
        import asr_lab.qwen_scheduler as scheduler_module
        self.release.set()
        await self.register("clock", "stream-clock", 50)
        with patch.object(scheduler_module, "time", SimpleNamespace(perf_counter=lambda: 0.015)):
            await self.scheduler.append_pcm("clock", "stream-clock", b"\x01\x00" * 400, received_at=0.010)
            await self.scheduler.append_pcm("clock", "stream-clock", b"\x02\x00" * 400, received_at=0.012)
            await self._wait_for(lambda: len(self.results) == 1)
        metrics = self.results[0][3]
        first_audio = metrics["first_audio_at"]
        ready_at = metrics["job_ready_at"]
        dispatch_at = metrics["dispatch_at"]
        ready_ms = (ready_at - first_audio) * 1000
        wait_ms = (dispatch_at - ready_at) * 1000
        start_ms = (dispatch_at - first_audio) * 1000
        self.assertAlmostEqual(ready_ms, 0.0)
        self.assertAlmostEqual(wait_ms, 5.0)
        self.assertAlmostEqual(start_ms, 5.0)
        self.assertAlmostEqual(ready_ms + wait_ms, start_ms)

    async def test_absolute_decode_slo_is_independent_of_chunk_ratio(self):
        self.release.set()
        walls = {"absolute": (1000, 150.0), "ratio": (50, 90.0)}
        async def execute(key, job):
            _, wall = walls[key.connection_id]
            return {"decode_wall_ms": wall, "decode_steps_delta": 1}
        self.scheduler.execute = execute
        await self.register("absolute", "sa", 1000)
        await self.register("ratio", "sr", 50)
        await self.scheduler.append_pcm("absolute", "sa", b"\x01\x00" * 16000)
        await self.scheduler.append_pcm("ratio", "sr", b"\x02\x00" * 800)
        await self._wait_for(lambda: len(self.results) == 2)
        metrics = {row[0].connection_id: row[3] for row in self.results}
        self.assertTrue(metrics["absolute"]["qwen_decode_slo_violation"])
        self.assertFalse(metrics["absolute"]["decode_overrun"])
        self.assertFalse(metrics["ratio"]["qwen_decode_slo_violation"])
        self.assertTrue(metrics["ratio"]["decode_overrun"])
        self.assertEqual(self.scheduler.snapshot()["qwen_decode_slo_violation_total"], 1)


if __name__ == "__main__":
    unittest.main()
