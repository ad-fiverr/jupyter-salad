import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NGINX_CONFIG = ROOT / "nginx-salad.conf"
SMOKE_SCRIPT = ROOT / "ci-smoke-test.sh"


class NginxListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = NGINX_CONFIG.read_text(encoding="utf-8")

    def test_ipv4_listener_is_configured_on_port_8888(self):
        self.assertRegex(
            self.config,
            re.compile(r"^\s*listen\s+0\.0\.0\.0:8888\s*;\s*$", re.MULTILINE),
        )

    def test_salad_ipv6_listener_is_preserved_on_port_8888(self):
        self.assertRegex(
            self.config,
            re.compile(r"^\s*listen\s+\[::\]:8888\s+ipv6only=on\s*;\s*$", re.MULTILINE),
        )

    def test_smoke_routes_nginx_functional_checks_over_ipv4(self):
        smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("nginx_direct 'http://127.0.0.1:8888/login'", smoke)
        self.assertIn('base_url = "http://127.0.0.1:8888"', smoke)
        self.assertNotIn("http://[::1]:8888", smoke)

    def test_smoke_keeps_jupyter_and_asr_direct_probes(self):
        smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("jupyter_direct 'http://127.0.0.1:8889/login'", smoke)
        self.assertIn("asr_direct 'http://127.0.0.1:8765/asr/health'", smoke)


if __name__ == "__main__":
    unittest.main()
