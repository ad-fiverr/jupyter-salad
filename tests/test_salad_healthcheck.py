import contextlib
import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("salad_healthcheck", ROOT / "salad_healthcheck.py")
assert SPEC is not None and SPEC.loader is not None
healthcheck = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(healthcheck)


class HealthcheckTests(unittest.TestCase):
    @patch.dict(os.environ, {"http_proxy": "http://proxy.invalid:3128", "HTTP_PROXY": "http://proxy.invalid:3128"})
    def test_empty_proxy_handler_disables_environment_proxy(self):
        default_opener = build_opener()
        default_proxy_handlers = [handler for handler in default_opener.handlers if isinstance(handler, ProxyHandler)]
        self.assertTrue(any(handler.proxies.get("http") for handler in default_proxy_handlers))

        opener = build_opener(ProxyHandler({}))
        proxy_handlers = [handler for handler in opener.handlers if isinstance(handler, ProxyHandler)]

        self.assertFalse(any(handler.proxies for handler in proxy_handlers))

    @patch.dict(os.environ, {"http_proxy": "http://proxy.invalid:3128", "HTTP_PROXY": "http://proxy.invalid:3128"})
    @patch.object(healthcheck, "build_opener")
    def test_uses_proxy_disabled_opener_for_ipv4_loopback(self, build_opener):
        response = contextlib.nullcontext(Mock(status=200))
        opener = Mock()
        opener.open.return_value = response
        build_opener.return_value = opener

        self.assertEqual(healthcheck.check_health(), 0)

        handler = build_opener.call_args.args[0]
        self.assertIsInstance(handler, ProxyHandler)
        self.assertEqual(handler.proxies, {})
        opener.open.assert_called_once_with("http://127.0.0.1:8888/login", timeout=3)

    @patch.object(healthcheck, "build_opener")
    def test_failure_reports_exception_type_without_exception_details(self, build_opener):
        opener = Mock()
        opener.open.side_effect = OSError("secret-proxy-password")
        build_opener.return_value = opener
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            self.assertEqual(healthcheck.check_health(), 1)

        self.assertIn("healthcheck=failed exception=OSError", stderr.getvalue())
        self.assertNotIn("secret-proxy-password", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
