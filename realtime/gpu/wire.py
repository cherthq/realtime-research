"""Length-bounded JSON + binary IPC; never deserialize pickle from a subprocess."""

from __future__ import annotations

import json
import struct

MAX_HEADER = 4096
MAX_BODY = 32 * 1024 * 1024


def _prefix(header: dict, body_size: int) -> bytes:
    if body_size > MAX_BODY:
        raise ValueError("GPU payload exceeds bound")
    raw = json.dumps({**header, "body_bytes": body_size}, allow_nan=False).encode()
    if len(raw) > MAX_HEADER:
        raise ValueError("GPU header exceeds bound")
    return struct.pack("!I", len(raw)) + raw


def encode(header: dict, body: bytes = b"") -> bytes:
    return _prefix(header, len(body)) + body


def write(stream, header: dict, body=b""):
    """Write a bounded packet without copying a full RGB batch into bytes twice.

    The caller owns the write lock and must retain the contiguous body until this
    synchronous write finishes. The wire format is identical to ``encode``.
    """
    payload = memoryview(body).cast("B")
    prefix = _prefix(header, payload.nbytes)
    for part in (memoryview(prefix), payload):
        while part:
            count = stream.write(part)
            if count is None or count <= 0:
                raise EOFError("GPU IPC closed during write")
            part = part[count:]
    stream.flush()


def parse_header(raw: bytes) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Invalid GPU header")
    size = value.get("body_bytes")
    if type(size) is not int or not 0 <= size <= MAX_BODY:
        raise ValueError("Invalid GPU payload length")
    return value


def read(stream):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            part = stream.read(size - len(result))
            if not part:
                raise EOFError("GPU IPC closed")
            result.extend(part)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Invalid GPU header length")
    header = parse_header(exact(size))
    return header, exact(header["body_bytes"])


async def read_async(stream):
    size = struct.unpack("!I", await stream.readexactly(4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Invalid GPU header length")
    header = parse_header(await stream.readexactly(size))
    return header, await stream.readexactly(header["body_bytes"])
