"""Bounded, process-local receipts for call-start response recovery.

Keys are caller secrets carried only in headers. Recovery never allocates a
worker, even after expiry or a process restart. This is not durable storage.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import re
import time

from fastapi import HTTPException


@dataclass
class Receipt:
    fingerprint: str
    task: asyncio.Task
    completed: float | None = None


class StartReceipts:
    def __init__(self, *, clock=time.monotonic, capacity=128, retention=1800):
        self.clock = clock
        self.capacity = capacity
        self.retention = retention
        self.entries: dict[str, Receipt] = {}
        self.lock = asyncio.Lock()

    async def run(self, key, payload, create=None):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise HTTPException(428, "A valid call-start key is required.")
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        async with self.lock:
            now = self.clock()
            for old_key, entry in list(self.entries.items()):
                if entry.completed is not None and now - entry.completed >= self.retention:
                    del self.entries[old_key]
            receipt = self.entries.get(key)
            if receipt is not None:
                if receipt.fingerprint != fingerprint:
                    raise HTTPException(409, "The call-start request has changed.")
            else:
                if create is None:
                    raise HTTPException(410, "The call-start result is unavailable or expired.")
                if len(self.entries) >= self.capacity:
                    raise HTTPException(503, "Call-start recovery is full. Try again later.")
                receipt = Receipt(fingerprint, asyncio.create_task(create()))
                self.entries[key] = receipt

                def finished(task):
                    receipt.completed = self.clock()
                    if not task.cancelled():
                        task.exception()  # Observe failures when the HTTP waiter left.

                receipt.task.add_done_callback(finished)
        if receipt.task.cancelled():
            raise HTTPException(503, "The call start was cancelled. Reconnect the call.")
        return await asyncio.shield(receipt.task)

    async def close(self):
        # Startup has its own deadline; shutdown must not race it with teardown.
        await asyncio.gather(*(r.task for r in self.entries.values()), return_exceptions=True)
        self.entries.clear()
