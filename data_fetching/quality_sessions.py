"""Scoped persistent local workers; file responses retain the existing audited protocol.

Only OCR/VAD are pooled. One request is active per child, with no ports or remote
services. A timeout/crash invalidates the child rather than reusing its model state.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

_ACTIVE = ContextVar("quality_worker_session", default=None)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)


class WorkerSession:
    def __init__(self):
        self.children = {}
        self.temporary = tempfile.TemporaryDirectory(prefix="chert-qc-workers-")

    def discard(self, key):
        child = self.children.pop(key, None)
        if child is not None:
            stop_process(child["process"])
            if child["process"].stdin is not None:
                child["process"].stdin.close()
            child["log"].close()

    def invoke(self, name, entry, request, response, log):
        key = (name, entry["fingerprint"])
        child = self.children.get(key)
        if child is not None and child["process"].poll() is not None:
            self.discard(key)
            child = None
        if child is None:
            # Registry changes must not keep stale model copies resident.
            for old in list(self.children):
                if old[0] == name:
                    self.discard(old)
            shared = Path(self.temporary.name) / f"{name}.log"
            handle = shared.open("w+b")
            try:
                process = subprocess.Popen(
                    [entry["python"], "-u", "-B", "-m", "data_fetching.quality_worker", "--serve"],
                    cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                    stdout=handle, stderr=handle, start_new_session=os.name != "nt",
                )
            except BaseException:
                handle.close()
                raise
            child = {"process": process, "log": handle, "path": shared}
            self.children[key] = child
        process, handle = child["process"], child["log"]
        handle.seek(0, os.SEEK_END)
        offset = handle.tell()
        deadline = time.monotonic() + entry.get("timeout_seconds", 1800)
        try:
            process.stdin.write((json.dumps({"request": str(request), "response": str(response)}) + "\n").encode())
            process.stdin.flush()
            while not response.exists():
                if process.poll() is not None:
                    raise subprocess.CalledProcessError(process.returncode, "persistent QC worker")
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("persistent QC worker", entry.get("timeout_seconds", 1800))
                time.sleep(0.02)
        except BaseException:
            # Stop first so stderr and owned FFmpeg descendants finish before copying evidence.
            stop_process(process)
            raise
        finally:
            with child["path"].open("rb") as reader:
                reader.seek(offset)
                with Path(log).open("wb") as output:
                    output.write(f"Persistent {name} worker pid={process.pid}\n".encode())
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        output.write(block)
            if process.poll() is not None:
                self.discard(key)

    def close(self):
        for key in list(self.children):
            child = self.children[key]
            process = child["process"]
            if process.poll() is None:
                try:
                    process.stdin.close()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    stop_process(process)
            child["log"].close()
            if process.stdin is not None:
                process.stdin.close()
            del self.children[key]
        self.temporary.cleanup()


@contextmanager
def worker_session(enabled=True):
    if not enabled or _ACTIVE.get() is not None:
        yield
        return
    session = WorkerSession()
    token = _ACTIVE.set(session)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        session.close()


def active_session():
    return _ACTIVE.get()
