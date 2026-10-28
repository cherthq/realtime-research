"""Install a pinned, checksum-verified local LiveKit server. No cloud account."""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.13.11"
HASHES = {
    "windows_amd64": "5d071ea43339f0140d24a66f0b2ff8a7bf6c4d0dceb19b30abca835670213667",
    "windows_arm64": "47ebfaa1ef6de546bf17a6699d4d9be59693bd5c3ffbec14d0749bd396ea3875",
    "linux_amd64": "1e9ee90ca301e1842b2887fc77cb122759756a9c8aeb4334ee91de3325ec43fb",
    "linux_arm64": "2af6ce091d6b3c79d1677f8ff1bbb91a8007f40b2325c059b52d38383c0830b6",
}


def install() -> Path:
    system = platform.system().lower()
    arch = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine().lower())
    target = f"{system}_{arch}"
    if target not in HASHES:
        raise SystemExit(
            "Installer supports Windows/Linux x64 and ARM64. Install LiveKit manually on other platforms."
        )
    extension = "zip" if system == "windows" else "tar.gz"
    name = f"livekit_{VERSION}_{target}.{extension}"
    cache = ROOT / ".cache" / "livekit" / VERSION / target
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / name
    if not archive.exists():
        url = f"https://github.com/livekit/livekit/releases/download/v{VERSION}/{name}"
        print(f"Downloading LiveKit {VERSION} for {target}", flush=True)
        temporary = archive.with_suffix(".download")
        with (
            urllib.request.urlopen(url, timeout=90) as source,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(source, output)
        temporary.replace(archive)
    blob = archive.read_bytes()
    if hashlib.sha256(blob).hexdigest() != HASHES[target]:
        raise SystemExit(f"Checksum mismatch: {archive}. Refusing to execute.")
    binary_name = "livekit-server.exe" if system == "windows" else "livekit-server"
    binary = cache / binary_name
    # Extract only the expected executable: never trust archive directory paths.
    if extension == "zip":
        with zipfile.ZipFile(io.BytesIO(blob)) as bundle:
            members = [n for n in bundle.namelist() if Path(n).name == binary_name]
            if len(members) != 1:
                raise SystemExit("Unexpected LiveKit archive contents")
            payload = bundle.read(members[0])
    else:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as bundle:
            members = [
                m
                for m in bundle.getmembers()
                if m.isfile() and Path(m.name).name == binary_name
            ]
            if len(members) != 1:
                raise SystemExit("Unexpected LiveKit archive contents")
            stream = bundle.extractfile(members[0])
            assert stream is not None
            payload = stream.read()
    if (
        not binary.exists()
        or hashlib.sha256(binary.read_bytes()).digest()
        != hashlib.sha256(payload).digest()
    ):
        binary.write_bytes(payload)
        binary.chmod(0o755)
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "run"])
    args = parser.parse_args()
    binary = install()
    if args.command == "install":
        print(binary)
        return
    print(
        "Development LiveKit at ws://127.0.0.1:7880 (devkey / secret). Ctrl+C stops it.",
        flush=True,
    )
    print(
        "HTTP binds loopback; media listeners may bind other interfaces. Keep media ports blocked to public ingress.",
        flush=True,
    )
    log_path = ROOT / ".cache" / "livekit" / "local-server.log"
    print(f"Server log: {log_path}", flush=True)
    log_file = log_path.open("w", encoding="utf-8")
    child = subprocess.Popen(
        [
            str(binary),
            "--dev",
            "--bind",
            "127.0.0.1",
            "--node-ip",
            "127.0.0.1",
            "--rtc.enable_loopback_candidate",
            "--logging.level",
            "warn",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        raise SystemExit(child.wait())
    except KeyboardInterrupt:
        pass
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        log_file.close()


if __name__ == "__main__":
    main()
