"""Prepare and supervise a private single-host demo, or forward it over SSH.

This does not launch/terminate a cloud instance or alter its firewall. Keep the
three service ports closed to public ingress; only SSH is needed for team use.
LiveKit's TCP media listener binds all interfaces even with HTTP bound locally.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import ipaddress
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from realtime.config import ROOT
from realtime.scripts.livekit_local import install

DEFAULT = ROOT / ".cache/realtime-host"


def ports(api: int, signaling: int, media: int) -> tuple[int, int, int]:
    values = (api, signaling, media)
    if len(set(values)) != 3 or any(not 1024 <= p <= 65535 for p in values):
        raise ValueError("Choose three distinct ports between 1024 and 65535")
    return values


def write_private(path: Path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def livekit_config(profile: dict) -> dict:
    config = {
        "port": profile["signal_port"],
        "bind_addresses": ["127.0.0.1"],
        "rtc": {
            "tcp_port": profile["media_port"],
            "force_tcp": True,
            "use_external_ip": False,
            "node_ip": "127.0.0.1",
            "enable_loopback_candidate": True,
        },
        "keys": {profile["livekit_key"]: profile["livekit_secret"]},
        "logging": {"level": "warn"},
    }
    if profile.get("client_address"):
        # The Windows browser reaches SSH forwards on WSL's private interface.
        # Retain host candidates so the remote publisher can connect locally.
        config["rtc"].update(
            node_ip=private_client_address(profile["client_address"]),
            advertise_internal_ip=True,
        )
    return config


def private_client_address(value: str) -> str:
    address = ipaddress.ip_address(value)
    if (
        address.version != 4
        or not address.is_private
        or address.is_unspecified
        or address.is_loopback
    ):
        raise ValueError(
            "Client address must be the private IPv4 address of the SSH tunnel host"
        )
    return str(address)


def prepare(directory: Path, api=8080, signaling=7880, media=7881, client_address=None):
    ports(api, signaling, media)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    profile = {
        "schema_version": 1,
        "transport": "ssh_tcp",
        "api_port": api,
        "signal_port": signaling,
        "media_port": media,
        "livekit_key": "chert_" + secrets.token_hex(12),
        "livekit_secret": secrets.token_urlsafe(48),
        "access_code": secrets.token_urlsafe(32),
    }
    if client_address:
        profile["client_address"] = private_client_address(client_address)
    # Never replace an existing profile or rotate an operator's credentials.
    if (directory / "profile.json").exists() or (directory / "livekit.yaml").exists():
        raise FileExistsError(
            "Profile already exists; reuse it or choose a fresh directory"
        )
    write_private(directory / "profile.json", profile)
    write_private(
        directory / "livekit.yaml", livekit_config(profile)
    )  # JSON is valid YAML.
    return profile


def load(directory: Path):
    directory = directory.resolve()
    if os.name == "posix":
        for path in (directory / "profile.json", directory / "livekit.yaml"):
            if path.stat().st_mode & 0o077:
                raise ValueError(
                    "Private profile permissions must be 600 on Linux (use chmod 600)"
                )
    profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("transport") != "ssh_tcp":
        raise ValueError("Unsupported demo profile")
    ports(profile["api_port"], profile["signal_port"], profile["media_port"])
    for name in ("livekit_key", "livekit_secret", "access_code"):
        if not isinstance(profile.get(name), str) or len(profile[name]) < 24:
            raise ValueError("Private credentials are missing or too short")
    if json.loads(
        (directory / "livekit.yaml").read_text(encoding="utf-8")
    ) != livekit_config(profile):
        raise ValueError("LiveKit configuration differs from this SSH-only profile")
    return profile


def environment(profile: dict):
    env = {
        **os.environ,
        "LIVEKIT_URL": f"ws://127.0.0.1:{profile['signal_port']}",
        "LIVEKIT_API_KEY": profile["livekit_key"],
        "LIVEKIT_API_SECRET": profile["livekit_secret"],
        "DEMO_ACCESS_TOKEN": profile["access_code"],
        "PYTHONUNBUFFERED": "1",
    }
    if profile.get("client_address"):
        env.update(
            LIVEKIT_BROWSER_URL=f"ws://{private_client_address(profile['client_address'])}:{profile['signal_port']}",
            CONTROL_TRANSPORT="poll",
        )
    return env


def tunnel_command(
    host: str, identity: Path | None, values=(8080, 7880, 7881), ssh_port=22
):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("Use an explicit user@hostname or user@IPv4 address")
    ports(*values)
    command = [
        "ssh",
        "-N",
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(ssh_port),
    ]
    if identity:
        command += ["-i", str(identity.expanduser())]
    for port in values:
        command += ["-L", f"127.0.0.1:{port}:127.0.0.1:{port}"]
    return command + [host]


def wait_http(port, path, child, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if child.poll() is not None:
            raise RuntimeError("Service exited during startup; inspect its local log")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=1
            ) as reply:
                if reply.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise TimeoutError("Service did not become ready; inspect its local log")


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def serve(directory: Path):
    profile = load(directory)
    if not (ROOT / "realtime/web/dist/index.html").is_file():
        raise RuntimeError(
            "Build the web application with npm ci and npm run build first"
        )
    # Bind checks happen before starting either service. Never stop an existing
    # server to reclaim a port, and never interpret a stale PID file as ownership.
    with ExitStack() as stack:
        for port in ports(
            profile["api_port"], profile["signal_port"], profile["media_port"]
        ):
            probe = stack.enter_context(socket.socket())
            # Linux TIME_WAIT sockets from a clean shutdown must not prevent a
            # restart. This still rejects a listener already owning the port.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    binary = install()
    print(
        "Private demo: keep API, signaling and media ports blocked to public ingress; forward them over SSH.",
        flush=True,
    )
    print(
        f"Logs: {directory.resolve()} (Ctrl+C stops these services; it does not terminate Lambda)",
        flush=True,
    )
    children = []
    running = True

    def signal_stop(*_):
        nonlocal running
        running = False

    previous = {
        sig: signal.signal(sig, signal_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with ExitStack() as stack:
            media_log = stack.enter_context((directory / "livekit.log").open("ab"))
            api_log = stack.enter_context((directory / "app.log").open("ab"))
            media = subprocess.Popen(
                [str(binary), "--config", str((directory / "livekit.yaml").resolve())],
                stdout=media_log,
                stderr=subprocess.STDOUT,
            )
            children.append(media)
            wait_http(profile["signal_port"], "/", media)
            app = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "realtime.app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(profile["api_port"]),
                    "--workers",
                    "1",
                    "--no-access-log",
                ],
                cwd=ROOT,
                env=environment(profile),
                stdout=api_log,
                stderr=subprocess.STDOUT,
            )
            children.append(app)
            wait_http(profile["api_port"], "/api/access", app)
            print(
                f"Ready: http://127.0.0.1:{profile['api_port']} — access_code is in the private profile.json",
                flush=True,
            )
            while running:
                if any(child.poll() is not None for child in children):
                    raise RuntimeError(
                        "A service exited; inspect app.log and livekit.log"
                    )
                time.sleep(0.2)
    finally:
        for child in reversed(children):
            stop(child)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def warm(directory: Path, portrait="ref01", timeout=900):
    """Load/compile an idle host's model without recording or starting voice."""
    if (
        portrait not in {"ref01", "ref02", "ref03"}
        or not math.isfinite(timeout)
        or not 0 < timeout <= 1800
    ):
        raise ValueError(
            "Choose a prepared portrait and a timeout in (0, 1800] seconds"
        )
    profile = load(directory)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    base = f"http://127.0.0.1:{profile['api_port']}"

    def request(path, method="GET", body=None, key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-Session-Key"] = key
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
            method=method,
        )
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    request("/api/access", "POST", {"token": profile["access_code"]})
    if request("/api/health")["active_session"]:
        raise RuntimeError("A call is active; warm-up did not start")
    started = time.monotonic()
    session = request(
        "/api/sessions",
        "POST",
        {"avatar_id": portrait, "model_id": "avatarforever", "mode": "diagnostic"},
    )
    try:
        # Validate readiness while holding our lease, then always release it.
        ready = request("/api/health")["avatarforever"]["loaded"]
        if not ready:
            raise RuntimeError("Model did not report readiness")
    finally:
        request(f"/api/sessions/{session['id']}", "DELETE", key=session["session_key"])
    return {
        "loaded": True,
        "warmup_seconds": time.monotonic() - started,
        "portrait": portrait,
        "note": "Model preparation only; no voice call or generated media qualification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "serve", "check", "warm"):
        p = sub.add_parser(name)
        p.add_argument("--directory", type=Path, default=DEFAULT)
        if name == "warm":
            p.add_argument(
                "--portrait", choices=["ref01", "ref02", "ref03"], default="ref01"
            )
            p.add_argument("--timeout", type=float, default=900)
        if name == "prepare":
            p.add_argument(
                "--client-address",
                help="Private WSL address for an SSH-forwarded Windows browser",
            )
            for flag, default in (("api", 8080), ("signal", 7880), ("media", 7881)):
                p.add_argument(f"--{flag}-port", type=int, default=default)
    p = sub.add_parser("tunnel")
    p.add_argument("--host", required=True)
    p.add_argument("--identity", type=Path)
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--dry-run", action="store_true")
    for flag, default in (("api", 8080), ("signal", 7880), ("media", 7881)):
        p.add_argument(f"--{flag}-port", type=int, default=default)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.directory,
            args.api_port,
            args.signal_port,
            args.media_port,
            args.client_address,
        )
        print(
            f"Created private profile in {args.directory.resolve()}. Keep it out of Git; no services started."
        )
    elif args.command == "check":
        profile = load(args.directory)
        print(
            json.dumps(
                {
                    key: profile[key]
                    for key in (
                        "schema_version",
                        "transport",
                        "api_port",
                        "signal_port",
                        "media_port",
                    )
                },
                indent=2,
            )
        )
        print(
            "Profile valid. This does not verify the cloud firewall, GPU, OpenAI access, or media delivery."
        )
    elif args.command == "serve":
        serve(args.directory)
    elif args.command == "warm":
        print(json.dumps(warm(args.directory, args.portrait, args.timeout), indent=2))
    else:
        command = tunnel_command(
            args.host,
            args.identity,
            (args.api_port, args.signal_port, args.media_port),
            args.ssh_port,
        )
        if args.dry_run:
            print(
                subprocess.list2cmdline(command)
                if os.name == "nt"
                else shlex.join(command)
            )
        else:
            print(
                f"Forwarding the demo to http://127.0.0.1:{args.api_port}; Ctrl+C closes the tunnel.",
                flush=True,
            )
            raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
