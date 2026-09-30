import contextlib
import errno
import importlib.util
import io
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nginx_diagnostics", ROOT / "salad_nginx_diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
MAIN = "user www-data; worker_processes auto; events { worker_connections 1024; } http { default_type application/octet-stream; include /etc/nginx/conf.d/default.conf; }"
SERVER = (ROOT / "nginx-salad.conf").read_text(encoding="utf-8")
DUMP = f"# configuration file /etc/nginx/nginx.conf:\n{MAIN}\n# configuration file /etc/nginx/conf.d/default.conf:\n{SERVER}\n"
RUNPOD_MAIN = "events { worker_connections 1024; } http { server { listen 9091; } server { listen 3001; } server { listen 7861; } server { listen 8081; } server { listen 8001; } server { listen 7270; } }"


def output_of(function, *args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = function(*args)
    return status, output.getvalue()


def proc_table(address, port, state="0A"):
    return f"sl local_address rem_address st\n0: {address}:{port:04X} 00000000:0000 {state} 0 0\n"


class ConfigContractTests(unittest.TestCase):
    def test_dump_proves_actual_default_config_is_loaded(self):
        files = diag.split_effective_config(DUMP)
        self.assertTrue(all(diag.effective_contract(files).values()))

    def test_syntax_success_without_loaded_default_fails_contract(self):
        result = subprocess.CompletedProcess(["nginx", "-T"], 0, DUMP.split("# configuration file /etc/nginx/conf.d/default.conf:")[0], "test is successful")
        with patch.object(diag.subprocess, "run", return_value=result) as command, \
             patch.object(Path, "read_text", return_value=SERVER):
            status, output = output_of(diag.capture_config, True)
        self.assertEqual(status, 1)
        self.assertIn("NGINX_CONF_D_INCLUDED=NO", output)
        self.assertIn("NGINX_EFFECTIVE_IPV4_8888=NO", output)
        self.assertIn("NGINX_DISK_DEFAULT_IPV4_8888=YES", output)
        self.assertIn("NGINX_MAIN_ONLY_SALAD_INCLUDE=YES", output)
        self.assertIn("NGINX_BUILD_CONTRACT=FAIL", output)
        command.assert_called_once_with(["nginx", "-T"], capture_output=True, text=True, timeout=5, check=False)

    def test_missing_ipv6_and_split_servers_fail_contract(self):
        for server in (
            "server { listen 0.0.0.0:8888; }",
            "server { listen 0.0.0.0:8888; } server { listen [::]:8888 ipv6only=on; }",
        ):
            self.assertFalse(all(diag.effective_contract({str(diag.MAIN_CONFIG): MAIN, str(diag.SERVER_CONFIG): server}).values()))

    def test_runpod_effective_config_with_correct_disk_file_is_proven_failure(self):
        # nginx -T shows the inherited main file only; the correct server file
        # is read separately from disk, as in the supplied failed-build output.
        dump = f"# configuration file /etc/nginx/nginx.conf:\n{RUNPOD_MAIN}\n"
        result = subprocess.CompletedProcess(["nginx", "-T"], 0, dump, "test is successful")

        def disk_file(path, *args, **kwargs):
            return SERVER if path.as_posix() == diag.SERVER_CONFIG else RUNPOD_MAIN

        with patch.object(diag.subprocess, "run", return_value=result), \
             patch.object(Path, "read_text", autospec=True, side_effect=disk_file):
            status, output = output_of(diag.capture_config, True)
        self.assertEqual(status, 1)
        self.assertIn("NGINX_CONF_D_INCLUDED=NO", output)
        self.assertIn("NGINX_EFFECTIVE_IPV4_8888=NO", output)
        self.assertIn("NGINX_EFFECTIVE_IPV6_8888=NO", output)
        self.assertIn("listen 9091;", output)
        self.assertIn("listen 7270;", output)
        self.assertIn("NGINX_DISK_DEFAULT_IPV4_8888=YES", output)
        self.assertIn("NGINX_DISK_DEFAULT_IPV6_8888=YES", output)
        self.assertIn("NGINX_BUILD_CONTRACT=FAIL", output)

    def test_owned_main_plus_salad_server_passes_all_effective_contract_flags(self):
        files = diag.split_effective_config(DUMP)
        facts = diag.effective_contract(files)
        self.assertTrue(all(facts.values()), facts)

    def test_contract_rejects_wildcard_additional_include_or_extra_server(self):
        for main in (
            MAIN.replace("default.conf", "*.conf"),
            MAIN.replace("include /etc/nginx/conf.d/default.conf;", "include /etc/nginx/conf.d/default.conf; include /etc/nginx/sites-enabled/*;"),
        ):
            facts = diag.effective_contract({str(diag.MAIN_CONFIG): main, str(diag.SERVER_CONFIG): SERVER})
            self.assertFalse(all(facts.values()), facts)
        extra_server = SERVER + "\nserver { listen 9091; }\n"
        facts = diag.effective_contract({str(diag.MAIN_CONFIG): MAIN, str(diag.SERVER_CONFIG): extra_server})
        self.assertFalse(all(facts.values()), facts)

    def test_contract_rejects_server_block_declared_in_owned_main_config(self):
        main_with_server = MAIN.replace("http {", "http { server { listen 9091; }")
        facts = diag.effective_contract({
            str(diag.MAIN_CONFIG): main_with_server,
            str(diag.SERVER_CONFIG): SERVER,
        })
        self.assertFalse(facts["NGINX_EFFECTIVE_SERVER_TOPOLOGY_OWNED"], facts)
        self.assertFalse(all(facts.values()), facts)

    def test_contract_rejects_inherited_effective_file_markers(self):
        files = diag.split_effective_config(DUMP.replace(
            "# configuration file /etc/nginx/conf.d/default.conf:",
            "# configuration file /etc/nginx/extra-runpod.conf:",
        ))
        self.assertFalse(all(diag.effective_contract(files).values()))

    def test_comments_and_quoted_text_cannot_spoof_listeners(self):
        config = '# server { listen 0.0.0.0:8888; }\nserver { set $x "listen [::]:8888 ipv6only=on;"; }'
        facts = diag.effective_contract({str(diag.MAIN_CONFIG): MAIN, str(diag.SERVER_CONFIG): config})
        self.assertFalse(facts["NGINX_EFFECTIVE_IPV4_8888"])
        self.assertFalse(facts["NGINX_EFFECTIVE_IPV6_8888"])

    def test_summary_whitelists_config_and_never_dumps_headers(self):
        config = SERVER + '\nserver { proxy_set_header Authorization "Bearer fixture-secret"; set $x "JUPYTER_PASSWORD=fixture-password"; include "$SECRET"; }'
        _, output = output_of(diag.print_config_summary, "/etc/nginx/conf.d/default.conf", config)
        self.assertIn('config_source="/etc/nginx/conf.d/default.conf"', output)
        self.assertIn("listen [::]:8888 ipv6only=on;", output)
        self.assertNotIn("fixture", output)
        self.assertNotIn("$SECRET", output)

    def test_nginx_T_failure_never_exposes_raw_stderr(self):
        result = subprocess.CompletedProcess(["nginx", "-T"], 1, "", "password=fixture-secret invalid directive")
        with patch.object(diag.subprocess, "run", return_value=result), patch.object(Path, "read_text", side_effect=FileNotFoundError):
            status, output = output_of(diag.capture_config, True)
        self.assertEqual(status, 1)
        self.assertIn("NGINX_EFFECTIVE_CONFIG_CAPTURED=NO", output)
        self.assertNotIn("fixture-secret", output)

    def test_valid_effective_contract_passes_and_shows_main_path(self):
        result = subprocess.CompletedProcess(["nginx", "-T"], 0, DUMP, "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok")
        with patch.object(diag.subprocess, "run", return_value=result), patch.object(Path, "read_text", return_value=SERVER):
            status, output = output_of(diag.capture_config, True)
        self.assertEqual(status, 0)
        self.assertIn("NGINX_BUILD_CONTRACT=PASS", output)
        self.assertIn('nginx_conf_path="/etc/nginx/nginx.conf"', output)

    def test_nginx_T_timeout_is_bounded_and_contract_fails(self):
        with patch.object(diag.subprocess, "run", side_effect=subprocess.TimeoutExpired("nginx", 5)), \
             patch.object(Path, "read_text", side_effect=FileNotFoundError):
            status, output = output_of(diag.capture_config, True)
        self.assertEqual(status, 1)
        self.assertIn("exception=TimeoutExpired", output)


class ListenerTests(unittest.TestCase):
    def test_proc_decodes_ipv4_and_ipv6_and_only_listen_state(self):
        ipv4 = "0100007F" if sys.byteorder == "little" else "7F000001"
        ipv6 = "00000000000000000000000001000000" if sys.byteorder == "little" else "00000000000000000000000000000001"
        self.assertEqual(diag.parse_proc_listeners(proc_table(ipv4, 8889)), {(4, "127.0.0.1", 8889)})
        self.assertEqual(diag.parse_proc_listeners(proc_table(ipv6, 8888), True), {(6, "::1", 8888)})
        self.assertEqual(diag.parse_proc_listeners(proc_table(ipv4, 8888, "01")), set())
        # A default nginx port must remain visible when :8888 is missing.
        self.assertEqual(diag.parse_proc_listeners(proc_table(ipv4, 80)), {(4, "127.0.0.1", 80)})

    def test_ss_extracts_endpoints_without_process_arguments(self):
        data = 'LISTEN 0 511 0.0.0.0:8888 0.0.0.0:* users:(("nginx",pid=42,fd=6))\nLISTEN 0 511 [::]:8888 [::]:* users:(("fixture-secret",pid=43))\nLISTEN 0 511 0.0.0.0:80 0.0.0.0:* users:(("nginx",pid=44))'
        self.assertEqual(diag.parse_ss_listeners(data), {(4, "0.0.0.0", 8888), (6, "::", 8888), (4, "0.0.0.0", 80)})

    def test_no_ss_falls_back_to_proc_and_ipv6_unavailable(self):
        def read(path, **kwargs):
            if path.as_posix() == "/proc/net/tcp":
                return proc_table("00000000", 8888)
            raise FileNotFoundError()
        with patch.object(diag.shutil, "which", return_value=None), \
             patch.object(Path, "glob", return_value=[]), \
             patch.object(Path, "read_text", autospec=True, side_effect=read):
            _, output = output_of(diag.capture_listeners)
        self.assertIn("ss_result=UNAVAILABLE fallback=proc", output)
        self.assertIn("RUNTIME_TCP_8888_IPV4=LISTENING", output)
        self.assertIn("RUNTIME_TCP_8888_IPV6=UNAVAILABLE", output)
        self.assertIn("JUPYTER_8889=NOT_LISTENING", output)

    def test_ss_is_bounded_and_raw_output_is_not_printed(self):
        result = subprocess.CompletedProcess(["ss"], 0, 'LISTEN 0 511 0.0.0.0:8888 0.0.0.0:* users:(("fixture-secret",pid=42))', "")
        with patch.object(diag.shutil, "which", return_value="/usr/bin/ss"), \
             patch.object(diag.subprocess, "run", return_value=result) as command, \
             patch.object(Path, "glob", return_value=[]), \
             patch.object(Path, "read_text", side_effect=FileNotFoundError):
            _, output = output_of(diag.capture_listeners)
        command.assert_called_once_with(["/usr/bin/ss", "-ltnp"], capture_output=True, text=True, timeout=2, check=False)
        self.assertIn("source=ss family=IPv4 address=0.0.0.0 port=8888", output)
        self.assertNotIn("fixture-secret", output)


class TcpProbeTests(unittest.TestCase):
    def test_probe_connects_to_all_expected_addresses_without_http(self):
        for name, (family, address, port) in diag.TCP_PROBES.items():
            connection = MagicMock()
            with patch.object(diag.socket, "socket", return_value=connection) as factory, patch.object(diag.socket, "has_ipv6", True):
                status, output = output_of(diag.tcp_probe, name)
            self.assertEqual(status, 0)
            factory.assert_called_once_with(family, socket.SOCK_STREAM)
            connection.__enter__.return_value.connect.assert_called_once_with((address, port))
            connection.__enter__.return_value.settimeout.assert_called_once_with(2)
            self.assertIn(f"probe={name} result=PASS exception=NONE errno=NONE", output)

    def test_refused_and_unavailable_differ_and_error_message_is_hidden(self):
        for name, number, expected, status in (
            ("nginx_tcp_ipv4", errno.ECONNREFUSED, "FAIL", 1),
            ("nginx_tcp_ipv6", errno.EAFNOSUPPORT, "UNAVAILABLE", 0),
            ("nginx_tcp_ipv6", errno.ECONNREFUSED, "FAIL", 1),
        ):
            with patch.object(diag.socket, "socket", side_effect=OSError(number, "fixture-secret")), patch.object(diag.socket, "has_ipv6", True):
                actual, output = output_of(diag.tcp_probe, name)
            self.assertEqual(actual, status)
            self.assertIn(f"result={expected}", output)
            self.assertIn(f"errno={number}", output)
            self.assertNotIn("fixture-secret", output)

    def test_no_ipv6_does_not_create_socket(self):
        with patch.object(diag.socket, "has_ipv6", False), patch.object(diag.socket, "socket") as factory:
            status, output = output_of(diag.tcp_probe, "nginx_tcp_ipv6")
        self.assertEqual(status, 0)
        self.assertIn("result=UNAVAILABLE", output)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
