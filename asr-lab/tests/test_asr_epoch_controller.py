from __future__ import annotations

import inspect
import unittest

from asr_lab.asr_epoch_controller import (
    EpochControllerError,
    EpochControllerState,
    LocalAcousticEpochController,
)
from asr_lab.asr_fencing import ExecutionFence, FenceOutcome


class SequenceAllocator:
    def __init__(self, *epoch_ids: str) -> None:
        self.epoch_ids = iter(epoch_ids)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return next(self.epoch_ids)


class LocalAcousticEpochControllerTests(unittest.TestCase):
    def make_controller(self, *epoch_ids: str, initial_epoch_seq: int = 41):
        allocator = SequenceAllocator(*epoch_ids)
        controller = LocalAcousticEpochController(
            asr_job_id="job-a",
            speech_segment_id="segment-a",
            initial_epoch_seq=initial_epoch_seq,
            epoch_id_allocator=allocator,
        )
        return controller, allocator

    def test_initial_epoch_uses_supplied_identity_seed_and_allocator(self):
        controller, allocator = self.make_controller("epoch-a")

        self.assertEqual(controller.state, EpochControllerState.ACTIVE)
        self.assertEqual(
            controller.current,
            ExecutionFence("job-a", "epoch-a", 41, "segment-a"),
        )
        self.assertEqual(allocator.calls, 1)
        self.assertFalse(hasattr(controller.current, "turn_id"))

    def test_initial_seed_is_required_and_must_be_an_integer_not_bool(self):
        with self.assertRaises(TypeError):
            LocalAcousticEpochController(
                asr_job_id="job-a",
                speech_segment_id="segment-a",
            )

        for invalid_seed in (True, False, 1.0, "1", None):
            with self.subTest(seed=invalid_seed):
                with self.assertRaises(EpochControllerError) as raised:
                    LocalAcousticEpochController(
                        asr_job_id="job-a",
                        speech_segment_id="segment-a",
                        initial_epoch_seq=invalid_seed,
                        epoch_id_allocator=lambda: "epoch-a",
                    )
                self.assertEqual(raised.exception.code, "invalid_initial_epoch_seq")

    def test_empty_or_non_string_admitted_identity_is_rejected(self):
        invalid_cases = (
            ("", "segment-a", "invalid_job_id"),
            ("  ", "segment-a", "invalid_job_id"),
            (None, "segment-a", "invalid_job_id"),
            ("job-a", "", "invalid_speech_segment_id"),
            ("job-a", " \t", "invalid_speech_segment_id"),
            ("job-a", None, "invalid_speech_segment_id"),
        )
        for job_id, segment_id, expected_code in invalid_cases:
            with self.subTest(job_id=job_id, segment_id=segment_id):
                with self.assertRaises(EpochControllerError) as raised:
                    LocalAcousticEpochController(
                        asr_job_id=job_id,
                        speech_segment_id=segment_id,
                        initial_epoch_seq=41,
                        epoch_id_allocator=lambda: "epoch-a",
                    )
                self.assertEqual(raised.exception.code, expected_code)

    def test_default_id_allocator_produces_nonempty_execution_identity(self):
        controller = LocalAcousticEpochController(
            asr_job_id="job-a",
            speech_segment_id="segment-a",
            initial_epoch_seq=41,
        )
        self.assertTrue(controller.current.epoch_id)

    def test_invalid_allocator_configuration_and_output_are_rejected(self):
        with self.assertRaises(EpochControllerError) as raised:
            LocalAcousticEpochController(
                asr_job_id="job-a",
                speech_segment_id="segment-a",
                initial_epoch_seq=41,
                epoch_id_allocator=object(),
            )
        self.assertEqual(raised.exception.code, "invalid_epoch_id_allocator")

        for invalid_id in ("", " \t", None, 123):
            with self.subTest(epoch_id=invalid_id):
                with self.assertRaises(EpochControllerError) as raised:
                    LocalAcousticEpochController(
                        asr_job_id="job-a",
                        speech_segment_id="segment-a",
                        initial_epoch_seq=41,
                        epoch_id_allocator=lambda value=invalid_id: value,
                    )
                self.assertEqual(raised.exception.code, "invalid_epoch_id")

    def test_lifecycle_prepares_then_promotes_exact_successor_once(self):
        controller, allocator = self.make_controller("epoch-a", "epoch-b")
        original = controller.current

        controller.request_rollover()
        self.assertEqual(controller.state, EpochControllerState.ROLLOVER_REQUESTED)
        self.assertIs(controller.current, original)

        prepared_epoch_id = controller.prepare_successor()
        self.assertEqual(controller.state, EpochControllerState.SUCCESSOR_PREPARED)
        self.assertEqual(prepared_epoch_id, "epoch-b")
        self.assertIs(controller.current, original)
        self.assertEqual(controller.prepared_epoch_id, "epoch-b")
        self.assertEqual(controller.current.epoch_seq, 41)
        self.assertEqual(allocator.calls, 2)

        activated = controller.activate_successor()
        self.assertEqual(controller.state, EpochControllerState.ACTIVE)
        self.assertEqual(activated, ExecutionFence("job-a", "epoch-b", 42, "segment-a"))
        self.assertIs(controller.current, activated)
        self.assertIsNone(controller.prepared_epoch_id)
        self.assertEqual(allocator.calls, 2)
        self.assertEqual(controller.current.epoch_seq, 42)

    def test_multiple_successors_advance_monotonically_once_each(self):
        controller, allocator = self.make_controller(
            "epoch-a", "epoch-b", "epoch-c", "epoch-d"
        )
        sequences = [controller.current.epoch_seq]

        for expected_id, expected_seq in (
            ("epoch-b", 42),
            ("epoch-c", 43),
            ("epoch-d", 44),
        ):
            controller.request_rollover()
            prepared_epoch_id = controller.prepare_successor()
            self.assertEqual(prepared_epoch_id, expected_id)
            self.assertEqual(controller.current.epoch_seq, expected_seq - 1)
            controller.activate_successor()
            self.assertEqual(controller.current.epoch_id, expected_id)
            sequences.append(controller.current.epoch_seq)

        self.assertEqual(sequences, [41, 42, 43, 44])
        self.assertEqual(allocator.calls, 4)

    def test_invalid_lifecycle_transitions_fail_deterministically(self):
        controller, _ = self.make_controller("epoch-a", "epoch-b")

        with self.assertRaises(EpochControllerError) as raised:
            controller.prepare_successor()
        self.assertEqual(raised.exception.code, "invalid_transition")
        first_message = str(raised.exception)
        with self.assertRaises(EpochControllerError) as raised_again:
            controller.prepare_successor()
        self.assertEqual(str(raised_again.exception), first_message)

        with self.assertRaises(EpochControllerError) as raised:
            controller.activate_successor()
        self.assertEqual(raised.exception.code, "invalid_transition")

        controller.request_rollover()
        with self.assertRaises(EpochControllerError) as raised:
            controller.request_rollover()
        self.assertEqual(raised.exception.code, "invalid_transition")

        controller.prepare_successor()
        with self.assertRaises(EpochControllerError) as raised:
            controller.prepare_successor()
        self.assertEqual(raised.exception.code, "invalid_transition")

    def test_current_and_obsolete_classification_delegates_to_a2(self):
        controller, _ = self.make_controller("epoch-a", "epoch-b")
        current = controller.current
        self.assertEqual(controller.classify(current), FenceOutcome.ACCEPT)

        controller.request_rollover()
        prepared_epoch_id = controller.prepare_successor()
        prepared_candidate = ExecutionFence("job-a", prepared_epoch_id, 42, "segment-a")
        self.assertEqual(controller.classify(prepared_candidate), FenceOutcome.INVALID_SCOPE)
        self.assertEqual(controller.current, current)

        activated = controller.activate_successor()
        self.assertEqual(activated, prepared_candidate)
        self.assertEqual(controller.classify(activated), FenceOutcome.ACCEPT)
        self.assertEqual(controller.classify(current), FenceOutcome.STALE)
        self.assertEqual(
            controller.classify(ExecutionFence("job-b", "epoch-x", 42, "segment-a")),
            FenceOutcome.INVALID_SCOPE,
        )
        self.assertEqual(
            controller.classify(ExecutionFence("job-a", "epoch-x", 42, "segment-b")),
            FenceOutcome.INVALID_SCOPE,
        )

    def test_historical_epoch_id_reuse_is_rejected(self):
        controller, _ = self.make_controller("epoch-a", "epoch-b", "epoch-a")
        controller.request_rollover()
        controller.prepare_successor()
        controller.activate_successor()
        controller.request_rollover()

        with self.assertRaises(EpochControllerError) as raised:
            controller.prepare_successor()
        self.assertEqual(raised.exception.code, "epoch_id_reused")
        self.assertEqual(controller.state, EpochControllerState.ROLLOVER_REQUESTED)
        self.assertIsNone(controller.prepared_epoch_id)

    def test_current_epoch_id_reuse_is_rejected(self):
        controller, _ = self.make_controller("epoch-a", "epoch-a")
        controller.request_rollover()
        current = controller.current

        with self.assertRaises(EpochControllerError) as raised:
            controller.prepare_successor()
        self.assertEqual(raised.exception.code, "epoch_id_reused")
        self.assertIs(controller.current, current)

    def test_epoch_id_allocation_is_independent_of_scheduler_and_pcm(self):
        controller, _ = self.make_controller("epoch-a", "epoch-b")
        public_method_parameters = {
            name: set(inspect.signature(getattr(type(controller), name)).parameters)
            for name in ("request_rollover", "prepare_successor", "activate_successor", "classify")
        }
        all_parameters = set().union(*public_method_parameters.values())

        self.assertTrue(
            all(
                "pcm" not in name.lower() and "audio" not in name.lower()
                for name in all_parameters
            )
        )
        self.assertFalse(hasattr(controller, "pcm"))
        self.assertFalse(hasattr(controller, "audio"))
        self.assertFalse(hasattr(controller, "connection_id"))
        self.assertFalse(hasattr(controller, "stream_id"))
        self.assertFalse(hasattr(controller, "scheduler_revision"))
        self.assertFalse(hasattr(controller, "turn_id"))
        self.assertFalse(hasattr(controller.current, "turn_id"))
        self.assertNotIn("EOS", {state.value for state in EpochControllerState})
        self.assertFalse(hasattr(controller, "finish"))
        self.assertFalse(hasattr(controller, "eos"))


if __name__ == "__main__":
    unittest.main()
