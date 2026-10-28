"""One resident avatar per process. stdin/stdout are private framed IPC only."""

from __future__ import annotations

import argparse
import contextlib
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from realtime.avatarforever import verify_installation
from realtime.contracts import AudioChunk, FRAME_SAMPLES
from realtime.gpu.wire import encode, read
from realtime.streaming import chunk_frames


def serve(backend, incoming, outgoing):
    jobs = queue.Queue(maxsize=1)
    lock = threading.Lock()
    write_lock = threading.Lock()
    stop = threading.Event()
    state = {"minimum_epoch": 0, "active": None, "busy": False, "session_id": None}

    def send(header, body=b""):
        with write_lock:
            outgoing.write(encode(header, body))
            outgoing.flush()

    def generate():
        current_epoch = None
        while not stop.is_set():
            try:
                header, body, cancelled = jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            opening = header.get("op") == "open_session"
            closing = header.get("op") == "close_session"
            identity = {"session_id": header.get("session_id")}
            if not opening and not closing:
                identity.update(
                    epoch=header["epoch"],
                    response_id=header["response_id"],
                    sample_start=header["sample_start"],
                )
            reply, payload = {"type": "cancelled", **identity}, b""
            try:
                if opening:
                    backend.open_session(Path(header["image"]))
                    current_epoch = None
                    with lock:
                        state["minimum_epoch"] = 0
                        state["session_id"] = header["session_id"]
                    reply = {"type": "session_ready", **identity}
                elif closing:
                    backend.close_session()
                    current_epoch = None
                    with lock:
                        state["session_id"] = None
                    reply = {"type": "session_closed", **identity}
                elif cancelled.is_set():
                    pass
                else:
                    if current_epoch != header["epoch"]:
                        backend.reset()
                        current_epoch = header["epoch"]
                    audio = AudioChunk(
                        header["response_id"],
                        header["epoch"],
                        header["sample_start"],
                        np.frombuffer(body, dtype="<i2").copy(),
                        header["valid_samples"],
                    )
                    result = backend.generate(audio, cancelled)
                    if not cancelled.is_set():
                        reply = {
                            "type": "frames",
                            **identity,
                            "frame_count": len(result.frames),
                            "compute_seconds": result.compute_seconds,
                        }
                        if hasattr(backend, "metrics"):
                            reply["memory"] = backend.metrics()
                        payload = result.frames.tobytes()
            except Exception as exc:
                if cancelled.is_set():
                    pass
                else:
                    # Full diagnostic on stderr; no input audio/transcript in the error.
                    import traceback

                    traceback.print_exc(file=sys.stderr)
                    reply = {
                        "type": "error",
                        **identity,
                        "error_type": type(exc).__name__,
                    }
                current_epoch = None
            finally:
                with lock:
                    state["active"], state["busy"] = None, False
                jobs.task_done()
            send(reply, payload)

    thread = threading.Thread(target=generate, name="avatar-gpu", daemon=True)
    thread.start()
    send({"type": "ready", **backend.info()})
    try:
        while True:
            header, body = read(incoming)
            if header.get("op") == "close":
                break
            if header.get("op") in {"open_session", "close_session"}:
                sid, image = header.get("session_id"), header.get("image")
                if not isinstance(sid, str) or not 1 <= len(sid) <= 100:
                    raise ValueError("Invalid session preparation")
                if header["op"] == "open_session" and (
                    not isinstance(image, str) or not Path(image).is_file()
                ):
                    raise ValueError("Missing character reference")
                with lock:
                    if state["busy"]:
                        raise ValueError("Cannot replace a character during generation")
                    if header["op"] == "close_session" and sid != state["session_id"]:
                        raise ValueError("Wrong session to close")
                    cancelled = threading.Event()
                    state["active"], state["busy"] = cancelled, True
                    jobs.put_nowait((header, body, cancelled))
                continue
            epoch = header.get("epoch")
            if type(epoch) is not int or epoch < 0:
                raise ValueError("Invalid response epoch")
            with lock:
                if header.get("op") == "cancel":
                    state["minimum_epoch"] = max(state["minimum_epoch"], epoch)
                    if state["active"]:
                        state["active"].set()
                elif header.get("op") == "generate":
                    if (
                        state["session_id"] is None
                        or header.get("session_id") != state["session_id"]
                    ):
                        raise ValueError("GPU session identity changed")
                    if state["busy"]:
                        raise ValueError("Concurrent GPU jobs are not supported")
                    position = header.get("sample_start")
                    valid = header.get("valid_samples")
                    response = header.get("response_id")
                    if (
                        type(position) is not int
                        or position < 0
                        or type(valid) is not int
                        or not isinstance(response, str)
                        or not 1 <= len(response) <= 200
                    ):
                        raise ValueError("Invalid chunk identity")
                    samples = chunk_frames(position) * FRAME_SAMPLES
                    if len(body) != samples * 2 or not 0 < valid <= samples:
                        raise ValueError("Invalid chunk coverage")
                    cancelled = threading.Event()
                    if epoch < state["minimum_epoch"]:
                        cancelled.set()
                    state["active"], state["busy"] = cancelled, True
                    jobs.put_nowait((header, body, cancelled))
                else:
                    raise ValueError("Unsupported GPU command")
    except EOFError:
        pass
    finally:
        stop.set()
        with lock:
            if state["active"]:
                state["active"].set()
        thread.join(timeout=5)
        if not thread.is_alive():
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--compile-blocks",
        action="store_true",
        help="Compile and warm repeated transformer blocks before readiness",
    )
    args = parser.parse_args()
    # Preserve binary stdout before redirecting third-party progress messages.
    outgoing = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        verify_installation(args.weights)
        from realtime.gpu.avatarforever import AvatarForeverBackend

        started = time.monotonic()
        backend = AvatarForeverBackend(
            args.weights, args.image, compile_blocks=args.compile_blocks
        )
        print(
            f"Avatar loaded in {time.monotonic() - started:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        serve(backend, sys.stdin.buffer, outgoing)


if __name__ == "__main__":
    main()
