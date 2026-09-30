from __future__ import annotations

import sys
import unittest

from asr_lab.backends.base import create_backend


BACKEND_MODULES = {
    "parakeet": "asr_lab.backends.parakeet",
    "faster_whisper": "asr_lab.backends.faster_whisper",
}


class BackendSelectionTests(unittest.TestCase):
    def tearDown(self):
        for module in BACKEND_MODULES.values():
            sys.modules.pop(module, None)

    def _assert_only_selected_backend_imports(self, selected: str):
        self.tearDown()
        backend = create_backend(selected)
        inactive = "faster_whisper" if selected == "parakeet" else "parakeet"
        self.assertEqual(backend.name, selected)
        self.assertIn(BACKEND_MODULES[selected], sys.modules)
        self.assertNotIn(BACKEND_MODULES[inactive], sys.modules)

    def test_parakeet_selection_does_not_import_faster_whisper(self):
        self._assert_only_selected_backend_imports("parakeet")

    def test_faster_whisper_selection_does_not_import_parakeet(self):
        self._assert_only_selected_backend_imports("faster_whisper")


if __name__ == "__main__":
    unittest.main()
