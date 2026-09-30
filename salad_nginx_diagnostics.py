"""Bounded, secret-safe nginx config/listener diagnostics; no model imports."""

import argparse
import errno
import ipaddress
import json
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys


MAIN_CONFIG = "/etc/nginx/nginx.conf"
SERVER_CONFIG = "/etc/nginx/conf.d/default.conf"
CONF_D_INCLUDE = "/etc/nginx/conf.d/*.conf"
TCP_PROBES = {
    "nginx_tcp_ipv4": (socket.AF_INET, "127.0.0.1", 8888),
    "jupyter_tcp": (socket.AF_INET, "127.0.0.1", 8889),
    "asr_tcp": (socket.AF_INET, "127.0.0.1", 8765),
    "nginx_tcp_ipv6": (socket.AF_INET6, "::1", 8888),
}


def statements(config):
    """Yield directive arguments and enclosing block names, ignoring comments."""
    lexer = shlex.shlex(config, posix=True, punctuation_chars="{};")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    scope, pending = [], []
    for token in lexer:
        # shlex groups consecutive punctuation (e.g. '}}').
        tokens = list(token) if token and set(token) <= set("{};") else [token]
        for part in tokens:
            if part == "{":
                scope.append(pending[0] if pending else "")
                yield "block", tuple(pending), tuple(scope)
                pending = []
            elif part == "}":
                yield "end", (), tuple(scope)
                if scope:
                    scope.pop()
                pending = []
            elif part == ";":
                yield "directive", tuple(pending), tuple(scope)
                pending = []
            else:
                pending.append(part)


def config_facts(config):
    includes, servers, current_server = [], [], None
    for kind, args, scope in statements(config):
        if kind == "block" and args == ("server",):
            current_server = []
            servers.append(current_server)
        elif kind == "end" and scope and scope[-1] == "server":
            current_server = None
        elif kind == "directive" and args:
            if args[0] == "include" and len(args) == 2:
                includes.append(args[1])
            if args[0] == "listen" and scope and scope[-1] == "server" and current_server is not None:
                current_server.append(args[1:])
    return {"includes": includes, "servers": servers}


def split_effective_config(dump):
    markers = list(re.finditer(r"^# configuration file (.+):\s*$", dump, re.MULTILINE))
    return {
        match.group(1): dump[match.end():markers[index + 1].start() if index + 1 < len(markers) else len(dump)]
        for index, match in enumerate(markers)
    }


def effective_contract(files):
    main = config_facts(files.get(str(MAIN_CONFIG), ""))
    server = config_facts(files.get(str(SERVER_CONFIG), ""))
    ipv4 = ("0.0.0.0:8888",)
    ipv6 = ("[::]:8888", "ipv6only=on")
    return {
        "NGINX_CONF_D_INCLUDED": CONF_D_INCLUDE in main["includes"] and str(SERVER_CONFIG) in files,
        "NGINX_EFFECTIVE_IPV4_8888": any(ipv4 in listeners for listeners in server["servers"]),
        "NGINX_EFFECTIVE_IPV6_8888": any(ipv6 in listeners for listeners in server["servers"]),
        "NGINX_EFFECTIVE_SAME_SERVER": any(ipv4 in listeners and ipv6 in listeners for listeners in server["servers"]),
    }


def safe_path(value):
    # Never dump arbitrary config strings, headers, variables or command arguments.
    return value if re.fullmatch(r"/[A-Za-z0-9_./*?\[\]-]+", value) else "[REDACTED_NONSTANDARD_PATH]"


def print_config_summary(source, config, origin="nginx_T"):
    facts = config_facts(config)
    if origin == "nginx_T":
        print(f"# configuration file {safe_path(source)}:")
    print(f"config_origin={origin} config_source={json.dumps(safe_path(source))}")
    for include in facts["includes"]:
        print(f"  include {json.dumps(safe_path(include))};")
    for listeners in facts["servers"]:
        print("  server {")
        for args in listeners:
            # Only numeric TCP listen addresses/options are safe to report.
            if args and re.fullmatch(r"[0-9a-fA-F.:\[\]]+", args[0]):
                options = [option for option in args[1:] if option in ("ipv6only=on", "ipv6only=off", "default_server", "ssl")]
                print("    listen " + " ".join((args[0], *options)) + ";")
        print("  }")


def capture_config(strict=False):
    try:
        result = subprocess.run(["nginx", "-T"], capture_output=True, text=True, timeout=5, check=False)
        files = split_effective_config(result.stdout)
        captured = result.returncode == 0 and bool(files)
        print(f"NGINX_EFFECTIVE_CONFIG_CAPTURED={'YES' if captured else 'NO'}")
        print(f"nginx_T_exit_code={result.returncode}")
        # nginx's success message identifies its main configuration path.
        for path in re.findall(r"configuration file (\S+) (?:syntax is ok|test is successful)", result.stderr):
            print(f"nginx_conf_path={json.dumps(safe_path(path))}")
        for path, config in files.items():
            print_config_summary(path, config)
        facts = effective_contract(files) if captured else dict.fromkeys(effective_contract({}), False)
        for key, value in facts.items():
            print(f"{key}={'YES' if value else 'NO'}")
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"NGINX_EFFECTIVE_CONFIG_CAPTURED=NO exception={type(error).__name__}")
        captured, facts = False, {}

    # These are disk-file facts, deliberately separate from the effective dump.
    for path, prefix in ((MAIN_CONFIG, "NGINX_DISK_MAIN"), (SERVER_CONFIG, "NGINX_DISK_DEFAULT")):
        try:
            config = Path(path).read_text(encoding="utf-8")
            print_config_summary(str(path), config, origin="disk")
            disk_facts = config_facts(config)
            if path == MAIN_CONFIG:
                print(f"{prefix}_CONF_D_INCLUDED={'YES' if CONF_D_INCLUDE in disk_facts['includes'] else 'NO'}")
            else:
                for key, value in effective_contract({str(path): config}).items():
                    if "IPV" in key:
                        print(f"{prefix}_{key.removeprefix('NGINX_EFFECTIVE_')}={'YES' if value else 'NO'}")
        except (OSError, ValueError) as error:
            print(f"{prefix}_READ=FAIL exception={type(error).__name__}")
    if strict and (not captured or not facts or not all(facts.values())):
        print("NGINX_BUILD_CONTRACT=FAIL")
        return 1
    if strict:
        print("NGINX_BUILD_CONTRACT=PASS")
    return 0


def parse_proc_listeners(data, ipv6=False):
    listeners = set()
    for line in data.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 4 or fields[3] != "0A":
            continue
        address_hex, port_hex = fields[1].split(":")
        port = int(port_hex, 16)
        raw = bytes.fromhex(address_hex)
        # /proc formats each native-endian 32-bit word separately.
        if sys.byteorder == "little":
            raw = b"".join(raw[offset:offset + 4][::-1] for offset in range(0, len(raw), 4))
        address = str(ipaddress.ip_address(raw))
        listeners.add((6 if ipv6 else 4, address, port))
    return listeners


def parse_ss_listeners(data):
    listeners = set()
    for line in data.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0] != "LISTEN":
            continue
        address, _, port = fields[3].rpartition(":")
        if not port.isdigit():
            continue
        address = address.strip("[]")
        # '*' is family-ambiguous; proc tables provide the precise family.
        if address == "*":
            continue
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        listeners.add((ip.version, str(ip), int(port)))
    return listeners


def capture_listeners():
    nginx_running = False
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            if path.read_text().strip() == "nginx":
                nginx_running = True
                command = (path.parent / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
                # Report only an explicit config override, never the raw command line.
                override = re.search(r"(?:^|\s)-c\s+(\S+)", command)
                config_path = safe_path(override.group(1)) if override else "DEFAULT"
                print(f"nginx_process_pid={path.parent.name} config_override={json.dumps(config_path)}")
        except OSError:
            pass
    print(f"NGINX_PROCESS_RUNNING={'YES' if nginx_running else 'NO'}")
    ss = shutil.which("ss")
    if ss:
        try:
            result = subprocess.run([ss, "-ltnp"], capture_output=True, text=True, timeout=2, check=False)
            print(f"ss_exit_code={result.returncode}")
            if result.returncode == 0:
                for family, address, port in sorted(parse_ss_listeners(result.stdout)):
                    print(f"listener source=ss family=IPv{family} address={address} port={port} state=LISTEN")
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"ss_result=UNAVAILABLE exception={type(error).__name__}")
    else:
        print("ss_result=UNAVAILABLE fallback=proc")
    # Proc also resolves ambiguous wildcard addresses and produces acceptance flags.
    listeners, available = set(), set()
    for family, path in ((4, "/proc/net/tcp"), (6, "/proc/net/tcp6")):
        try:
            listeners.update(parse_proc_listeners(Path(path).read_text(), ipv6=family == 6))
            available.add(family)
        except (OSError, ValueError) as error:
            print(f"listener_source={path} result=UNAVAILABLE exception={type(error).__name__}")
    for family, address, port in sorted(listeners):
        print(f"listener source=proc family=IPv{family} address={address} port={port} state=LISTEN")
    for key, family, port, addresses in (
        ("RUNTIME_TCP_8888_IPV4", 4, 8888, {"0.0.0.0", "127.0.0.1"}),
        ("RUNTIME_TCP_8888_IPV6", 6, 8888, {"::", "::1"}),
        ("JUPYTER_8889", 4, 8889, {"0.0.0.0", "127.0.0.1"}),
        ("ASR_8765", 4, 8765, {"0.0.0.0", "127.0.0.1"}),
    ):
        status = "UNAVAILABLE" if family not in available else (
            "LISTENING" if any((family, address, port) in listeners for address in addresses) else "NOT_LISTENING")
        print(f"{key}={status}")
    return 0


def tcp_probe(name):
    family, address, port = TCP_PROBES[name]
    if family == socket.AF_INET6 and not socket.has_ipv6:
        print(f"probe={name} result=UNAVAILABLE exception=IPv6Unavailable errno=NONE")
        return 0
    try:
        with socket.socket(family, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect((address, port))
    except OSError as error:
        unavailable = family == socket.AF_INET6 and error.errno in (
            errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT, errno.EADDRNOTAVAIL, errno.ENETUNREACH)
        status = "UNAVAILABLE" if unavailable else "FAIL"
        print(f"probe={name} result={status} exception={type(error).__name__} errno={error.errno if error.errno is not None else 'NONE'}")
        return 0 if unavailable else 1
    print(f"probe={name} result=PASS exception=NONE errno=NONE")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "config", "listeners", "tcp"))
    parser.add_argument("probe", nargs="?", choices=tuple(TCP_PROBES))
    args = parser.parse_args()
    if args.mode == "tcp":
        if args.probe is None:
            parser.error("tcp requires a probe name")
        return tcp_probe(args.probe)
    if args.mode == "listeners":
        return capture_listeners()
    return capture_config(strict=args.mode == "contract")


if __name__ == "__main__":
    raise SystemExit(main())
