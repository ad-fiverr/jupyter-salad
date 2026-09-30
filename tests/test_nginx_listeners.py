import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NGINX_CONFIG = ROOT / "nginx-salad.conf"
NGINX_MAIN = ROOT / "nginx-main.conf"
SMOKE_SCRIPT = ROOT / "ci-smoke-test.sh"
DOCKERFILE = ROOT / "Dockerfile"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"


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

    def test_owned_main_config_includes_only_the_salad_server_file(self):
        main = NGINX_MAIN.read_text(encoding="utf-8")
        self.assertRegex(main, re.compile(r"^events\s*\{", re.MULTILINE))
        self.assertRegex(main, re.compile(r"^http\s*\{", re.MULTILINE))
        includes = re.findall(r"^\s*include\s+([^;]+);\s*$", main, re.MULTILINE)
        self.assertEqual(includes, ["/etc/nginx/conf.d/default.conf"])
        self.assertNotIn("conf.d/*.conf", main)
        self.assertNotIn("sites-enabled", main)
        # No local static files are served; upstreams supply their own MIME types.
        self.assertIn("default_type application/octet-stream;", main)

    def test_dockerfile_owns_main_config_before_effective_build_gate(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        main_copy = dockerfile.index("COPY nginx-main.conf /etc/nginx/nginx.conf")
        server_copy = dockerfile.index("COPY nginx-salad.conf /etc/nginx/conf.d/default.conf")
        syntax_check = dockerfile.index("&& nginx -t")
        contract = dockerfile.index("salad-nginx-diagnostics.py contract")
        verifier = dockerfile.index("    && /usr/local/bin/salad-jupyter-verify-runtime")
        self.assertLess(main_copy, syntax_check)
        self.assertLess(server_copy, syntax_check)
        self.assertLess(syntax_check, contract)
        self.assertLess(contract, verifier)
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("'nginx-main.conf'", workflow)

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
