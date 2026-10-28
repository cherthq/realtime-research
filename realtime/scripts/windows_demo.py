"""Run the local diagnostic demo in Windows Chrome/Edge, with services in WSL.

Some Windows hosts stall after the first write on loopback TCP connections.
Buffer HTTP replies into one write, use polling for control, and send WebRTC
directly to WSL's private IP. No browser flags or firewall changes are needed.
This is a local development launcher, not a public deployment proxy.
"""

from __future__ import annotations

import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PORT = 18800
API_PORT, SIGNAL_PORT, MEDIA_PORT = 18801, 18802, 18803
MAX_BODY = 2 * 1024 * 1024
MAX_RESPONSE = 32 * 1024 * 1024
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


class Bridge(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, upstream):
        self.upstream = upstream
        super().__init__(address, BridgeRequest)


class BridgeRequest(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_):
        pass  # Do not log session keys, cookies or request URLs.

    def reply(self, status, body, headers=()):
        reason = http.client.responses.get(status, "Response")
        lines = [f"HTTP/1.1 {status} {reason}"]
        lines += [
            f"{key}: {value}"
            for key, value in headers
            if key.lower() not in HOP_HEADERS
        ]
        lines += [f"Content-Length: {len(body)}", "Connection: close", "", ""]
        self.close_connection = True
        # One write is deliberate: separately sending headers and body stalls
        # on the affected Windows loopback stack, even for a 420-byte document.
        self.connection.sendall("\r\n".join(lines).encode("latin-1") + body)

    def send_error(self, code, message=None, explain=None):
        self.reply(
            code,
            json.dumps(
                {"detail": message or http.client.responses.get(code, "Request failed")}
            ).encode(),
            [("Content-Type", "application/json"), ("Cache-Control", "no-store")],
        )

    def forward(self):
        host = self.headers.get("Host", "")
        allowed = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if host not in allowed or (
            self.headers.get("Origin") not in (None, f"http://{host}")
        ):
            return self.send_error(403, "Local same-origin requests only")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return self.send_error(403, "Cross-site access denied")
        parsed = urlsplit(self.path)
        if (
            parsed.scheme
            or parsed.netloc
            or not self.path.startswith("/")
            or self.path.startswith("//")
        ):
            return self.send_error(400, "Relative request path required")
        if parsed.path.endswith("/events"):
            return self.send_error(
                400, "This local launcher uses polling for session updates"
            )
        if self.headers.get("Transfer-Encoding") or self.headers.get("Upgrade"):
            return self.send_error(400, "Unsupported request encoding")
        lengths = self.headers.get_all("Content-Length", [])
        try:
            length = int(lengths[0]) if lengths else 0
            if len(lengths) > 1 or not 0 <= length <= MAX_BODY:
                raise ValueError
        except ValueError:
            return self.send_error(413, "Invalid or oversized request body")
        upstream = http.client.HTTPConnection(*self.server.upstream, timeout=350)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                return self.send_error(400, "Incomplete request body")
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS
            }
            headers["Connection"] = "close"
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            data = response.read(MAX_RESPONSE + 1)
            if len(data) > MAX_RESPONSE:
                return self.send_error(
                    502, "Upstream response exceeds the local bridge limit"
                )
            self.reply(response.status, data, response.getheaders())
        except (OSError, http.client.HTTPException):
            self.send_error(
                502, "Local demo backend is unavailable; restart the launcher"
            )
        finally:
            upstream.close()

    do_GET = do_POST = do_DELETE = do_HEAD = forward


def private_ip(value):
    address = ipaddress.ip_address(value)
    if (
        address.version != 4
        or not address.is_private
        or address.is_loopback
        or address.is_unspecified
    ):
        raise ValueError("Expected a private WSL IPv4 address")
    return str(address)


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def linux_services(address):
    from realtime.scripts.livekit_local import install

    address = private_ip(address)
    for port in (API_PORT, SIGNAL_PORT, MEDIA_PORT):
        with socket.socket() as probe:
            probe.bind((address, port))
    directory = Path.home() / ".cache/chert/windows-demo"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    key, secret = "chert_" + secrets.token_hex(12), secrets.token_urlsafe(48)
    configuration = {
        "port": SIGNAL_PORT,
        "bind_addresses": [address],
        "rtc": {
            "tcp_port": MEDIA_PORT,
            "force_tcp": True,
            "use_external_ip": False,
            "node_ip": address,
        },
        "keys": {key: secret},
        "logging": {"level": "warn"},
    }
    path = directory / "livekit.json"
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
    ) as output:
        json.dump(configuration, output)
    environment = {
        **os.environ,
        "LIVEKIT_URL": f"ws://{address}:{SIGNAL_PORT}",
        "LIVEKIT_API_KEY": key,
        "LIVEKIT_API_SECRET": secret,
        "CONTROL_TRANSPORT": "poll",
        "PYTHONUNBUFFERED": "1",
    }
    ending = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: ending.set())

    # Windows owns stdin. EOF also cleans up if the launcher window is closed.
    def watch_parent():
        sys.stdin.buffer.read()
        ending.set()

    threading.Thread(target=watch_parent, daemon=True).start()
    children = []
    try:
        with (
            (directory / "livekit.log").open("ab") as media_log,
            (directory / "app.log").open("ab") as api_log,
        ):
            children.append(
                subprocess.Popen(
                    [str(install()), "--config", str(path)],
                    stdout=media_log,
                    stderr=subprocess.STDOUT,
                )
            )
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "realtime.app:app",
                        "--host",
                        address,
                        "--port",
                        str(API_PORT),
                        "--no-access-log",
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=api_log,
                    stderr=subprocess.STDOUT,
                )
            )
            while not ending.wait(0.2):
                if any(child.poll() is not None for child in children):
                    raise RuntimeError(f"A service exited. Inspect logs in {directory}")
    finally:
        for child in reversed(children):
            stop(child)


def windows_demo(distro, python):
    if os.name != "nt":
        raise RuntimeError(
            "Run this launcher in Windows; use --services only inside WSL"
        )
    if not (ROOT / "realtime/web/dist/index.html").is_file():
        raise RuntimeError(
            "Build the frontend first: cd realtime/web && npm ci && npm run build"
        )
    command = ["wsl", "-d", distro, "--exec"]
    addresses = subprocess.check_output(command + ["hostname", "-I"], text=True).split()
    address = next((private_ip(value) for value in addresses if ":" not in value), None)
    if not address:
        raise RuntimeError("WSL has no private IPv4 address")
    root = subprocess.check_output(
        command + ["wslpath", "-a", str(ROOT)], text=True
    ).strip()
    if not python:
        home = subprocess.check_output(
            command + ["printenv", "HOME"], text=True
        ).strip()
        python = f"{home}/.cache/chert/realtime-venv/bin/python"
    # Bind before launching any child so a second launch never replaces services.
    with Bridge(("127.0.0.1", PORT), (address, API_PORT)) as server:
        child = subprocess.Popen(
            [
                "wsl",
                "-d",
                distro,
                "--cd",
                root,
                "--exec",
                python,
                "-m",
                "realtime.scripts.windows_demo",
                "--services",
                address,
            ],
            stdin=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 45
            while True:
                if child.poll() is not None:
                    raise RuntimeError("WSL services exited during startup")
                connection = http.client.HTTPConnection(address, API_PORT, timeout=2)
                try:
                    connection.request(
                        "GET", "/api/access", headers={"Host": f"127.0.0.1:{PORT}"}
                    )
                    response = connection.getresponse()
                    if response.status == 200:
                        response.read()
                        break
                except OSError:
                    pass
                finally:
                    connection.close()
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "WSL services did not start; inspect ~/.cache/chert/windows-demo logs in WSL"
                    )
                time.sleep(0.2)
            print(
                f"Open http://127.0.0.1:{PORT} in Windows Chrome or Edge. Ctrl+C stops the local demo.",
                flush=True,
            )
            server.timeout = 0.25
            while child.poll() is None:
                server.handle_request()
            raise RuntimeError("WSL services stopped; restart the launcher")
        except KeyboardInterrupt:
            pass
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=55)
            except subprocess.TimeoutExpired:
                stop(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument(
        "--wsl-python",
        help="Absolute Python path inside WSL with realtime dependencies installed",
    )
    parser.add_argument("--services", help=argparse.SUPPRESS)
    parser.add_argument(
        "--upstream",
        help="Private WSL IP:port of an existing SSH-forwarded API; start only the HTTP bridge",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="HTTP bridge port when --upstream is supplied",
    )
    args = parser.parse_args()
    if args.upstream:
        address, port = args.upstream.rsplit(":", 1)
        address, port = private_ip(address), int(port)
        if not (1024 <= args.port <= 65535 and 1024 <= port <= 65535):
            parser.error("Use ports between 1024 and 65535")
        with Bridge(("127.0.0.1", args.port), (address, port)) as server:
            print(
                f"Open http://127.0.0.1:{args.port}; forwarding to the private SSH route.",
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    elif args.services:
        linux_services(args.services)
    else:
        windows_demo(args.distro, args.wsl_python)


if __name__ == "__main__":
    main()
