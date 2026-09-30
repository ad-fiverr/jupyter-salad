from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

from asr_lab.backends.parakeet import (
    HALLUCINATION_BLACKLIST,
    PARAKEET_MODEL,
    ParakeetBackend,
    normalize_historical_transcript,
    should_discard_historical_transcript,
)


class ParakeetNemoTests(unittest.TestCase):
    def test_nemo_loader_cuda_eval_warmup_and_transcribe_follow_historical_api(self):
        import numpy as np

        model = Mock()
        model.cuda.return_value = model
        model.transcribe.side_effect = [[], [types.SimpleNamespace(text=" Hola. ")]]
        asr_model = types.SimpleNamespace(from_pretrained=Mock(return_value=model))
        nemo_asr = types.ModuleType("nemo.collections.asr")
        nemo_asr.models = types.SimpleNamespace(ASRModel=asr_model)
        nemo = types.ModuleType("nemo")
        nemo.__path__ = []
        collections = types.ModuleType("nemo.collections")
        collections.__path__ = []
        collections.asr = nemo_asr
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=Mock(return_value=True))

        with patch.dict(sys.modules, {
            "torch": torch,
            "nemo": nemo,
            "nemo.collections": collections,
            "nemo.collections.asr": nemo_asr,
        }):
            backend = ParakeetBackend()
            backend.load(PARAKEET_MODEL, "ignored-transformers-revision")
            model.eval.assert_called_once_with()
            model.cuda.assert_called_once_with()
            asr_model.from_pretrained.assert_called_once_with(PARAKEET_MODEL)
            backend.warmup()
            warmup_audio = model.transcribe.call_args_list[0].args[0][0]
            self.assertEqual(warmup_audio.shape, (16_000,))
            self.assertEqual(warmup_audio.dtype, np.float32)
            self.assertEqual(backend.transcribe(b"\x00\x40" * 8_000), "Hola.")
            self.assertEqual(model.transcribe.call_count, 2)

    def test_exact_historical_blacklist_and_normalization_are_restored(self):
        expected = {
            "you", "thank you", "oh", "bye", "subtitles by",
            "gracias por ver el video", "gracias por ver el vídeo",
            "suscríbete al canal", "yeah", "let's go", "mm-hmm",
        }
        self.assertEqual(HALLUCINATION_BLACKLIST, expected)
        self.assertEqual(normalize_historical_transcript(" ¡GRACIAS POR VER EL VÍDEO! "), "gracias por ver el vídeo")
        for phrase in expected:
            with self.subTest(phrase=phrase):
                self.assertTrue(should_discard_historical_transcript(phrase.upper() + "!"))
        self.assertFalse(should_discard_historical_transcript("thank you very much"))


if __name__ == "__main__":
    unittest.main()
