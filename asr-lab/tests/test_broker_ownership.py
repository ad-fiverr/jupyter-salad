from __future__ import annotations

import asyncio
import unittest
from unittest.mock import Mock, patch

from asr_lab.broker import InferenceBroker


class FakeBackend:
    def __init__(self):
        self.model_id = "fake-model"
        self.seen_jobs = []

    def load(self, model_id, model_revision):
        self.model_id = model_id

    def warmup(self):
        return None

    def transcribe(self, pcm16le: bytes) -> str:
        marker = pcm16le[:1].decode("ascii")
        self.seen_jobs.append(marker)
        import time
        time.sleep(0.12)
        return marker

    def close(self):
        return None


class FixedTextBackend(FakeBackend):
    def transcribe(self, pcm16le: bytes) -> str:
        return "You!"


class BrokerOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_worker_owns_a_model_and_round_robin_jobs_keep_connection_ownership(self):
        backends = []

        def create_worker_backend(name):
            backend = FakeBackend()
            backends.append(backend)
            return backend

        backend_factory = Mock(side_effect=create_worker_backend)
        with patch("asr_lab.broker.create_backend", backend_factory):
            broker = InferenceBroker(backend_name="parakeet", model_id="fake", model_revision=None, workers=2, queue_size=4)
            await broker.start()
            try:
                task_a = asyncio.create_task(broker.transcribe(
                    connection_id="client-a", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=100,
                ))
                await asyncio.sleep(0.01)
                self.assertIn("client-a", broker.job_owners.values())
                task_b = asyncio.create_task(broker.transcribe(
                    connection_id="client-b", source="system", speaker="them", pcm16le=b"B" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=100,
                ))
                await asyncio.sleep(0.01)
                self.assertIn("client-b", broker.job_owners.values())
                result_a, result_b = await asyncio.gather(task_a, task_b)
                self.assertEqual(result_a["text"], "A")
                self.assertEqual(result_b["text"], "B")
                self.assertEqual(result_a["speaker"], "you")
                self.assertEqual(result_b["speaker"], "them")
                self.assertIn("MODEL_INFERENCE_MS", result_a)
                self.assertIn("SEGMENT_WAIT_MS", result_b)
                self.assertEqual(backend_factory.call_count, 2)
                self.assertEqual(sorted(backend.seen_jobs for backend in backends), [["A"], ["B"]])
                self.assertEqual(sorted(metric["worker"] for metric in broker.worker_metrics), [0, 1])
                self.assertEqual(broker.next_worker_index, 0)
                self.assertEqual(broker.job_owners, {})
            finally:
                await broker.close()

    async def test_disconnected_client_late_result_is_discarded(self):
        with patch("asr_lab.broker.create_backend", return_value=FakeBackend()):
            broker = InferenceBroker(backend_name="parakeet", model_id="fake", model_revision="fixed", workers=1, queue_size=4)
            await broker.start()
            try:
                abandoned = asyncio.create_task(broker.transcribe(
                    connection_id="gone", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=100,
                ))
                await asyncio.sleep(0.005)
                abandoned.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await abandoned
                result = await broker.transcribe(
                    connection_id="survivor", source="mic", speaker="them", pcm16le=b"B" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=100,
                )
                self.assertEqual(result["text"], "B")
                self.assertEqual(result["speaker"], "them")
            finally:
                await broker.close()

    async def test_queue_limit_remains_global_across_worker_queues(self):
        with patch("asr_lab.broker.create_backend", return_value=FakeBackend()):
            broker = InferenceBroker(backend_name="parakeet", model_id="fake", model_revision=None, workers=2, queue_size=1)
            await broker.start()
            try:
                # Hold both workers outside the queue after they dequeue their first
                # jobs; at most the configured number may wait in the queues.
                first = asyncio.create_task(broker.transcribe(
                    connection_id="one", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=0,
                ))
                await asyncio.sleep(0.01)
                second = asyncio.create_task(broker.transcribe(
                    connection_id="two", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=0,
                ))
                await asyncio.sleep(0.01)
                third = asyncio.create_task(broker.transcribe(
                    connection_id="three", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=0,
                ))
                await asyncio.sleep(0.01)
                fourth = asyncio.create_task(broker.transcribe(
                    connection_id="four", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=0,
                ))
                with self.assertRaisesRegex(RuntimeError, "inference_queue_full"):
                    await fourth
                await asyncio.gather(first, second, third)
            finally:
                await broker.close()

    async def test_hallucination_filter_is_only_applied_to_parakeet(self):
        with patch("asr_lab.broker.create_backend", return_value=FixedTextBackend()):
            parakeet = InferenceBroker(backend_name="parakeet", model_id="fake", model_revision=None, workers=1, queue_size=2)
            await parakeet.start()
            try:
                result = await parakeet.transcribe(
                    connection_id="parakeet-client", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=0,
                )
                self.assertTrue(result["_discarded_historical_output"])
            finally:
                await parakeet.close()

        with patch("asr_lab.broker.create_backend", return_value=FixedTextBackend()):
            faster_whisper = InferenceBroker(backend_name="faster_whisper", model_id="fake", model_revision="fixed", workers=1, queue_size=2)
            await faster_whisper.start()
            try:
                result = await faster_whisper.transcribe(
                    connection_id="fw-client", source="mic", speaker="you", pcm16le=b"A" * 32000,
                    segment_start_s=0, segment_end_s=1, segment_wait_ms=0,
                )
                self.assertEqual(result["text"], "You!")
                self.assertNotIn("_discarded_historical_output", result)
            finally:
                await faster_whisper.close()


if __name__ == "__main__":
    unittest.main()
