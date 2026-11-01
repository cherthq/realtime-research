"""FIFO admission for the single-process public demo gateway.

Tickets are bearer credentials supplied in a header, never in a URL. This queue
does not create GPU sessions; the application's session lock remains the final
authority and holds its lease until teardown completes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import re
import time
from typing import Awaitable, Callable

from fastapi import HTTPException


@dataclass
class Ticket:
    seen: float
    offered_until: float | None = None


class CallQueue:
    def __init__(
        self,
        health: Callable[[], Awaitable[dict]],
        *,
        clock=time.monotonic,
        heartbeat_seconds=90,
        claim_seconds=30,
        capacity=100,
    ):
        self.health = health
        self.clock = clock
        self.heartbeat_seconds = heartbeat_seconds
        self.claim_seconds = claim_seconds
        self.capacity = capacity
        self.lock = asyncio.Lock()
        self.tickets: dict[str, Ticket] = {}
        self.starting: str | None = None
        self.ready = False
        self.busy = True
        self.checked = float("-inf")

    @staticmethod
    def validate(key: str):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise HTTPException(428, "Join the waiting queue before starting a call.")

    async def _refresh(self, *, fresh=False):
        # All observations and admission decisions share this lock. A stale
        # health response cannot promote another visitor during an in-flight start.
        now = self.clock()
        if not self.starting and (fresh or now - self.checked >= 2):
            try:
                state = await self.health()
                self.ready = state.get("avatarforever_ready") is True
                self.busy = state.get("active_session") is not False
            except Exception:
                self.ready, self.busy = False, True  # Fail closed on outages.
            self.checked = self.clock()
        now = self.clock()
        for key, ticket in list(self.tickets.items()):
            if key == self.starting:
                continue
            if now - ticket.seen >= self.heartbeat_seconds or (
                ticket.offered_until is not None and now >= ticket.offered_until
            ):
                del self.tickets[key]
        if self.busy or not self.ready or self.starting:
            for ticket in self.tickets.values():
                ticket.offered_until = None
        elif self.tickets:
            first = next(iter(self.tickets.values()))
            if first.offered_until is None:
                first.offered_until = now + self.claim_seconds

    def _status(self, key):
        ticket = self.tickets.get(key)
        if ticket is None:
            raise HTTPException(
                410, "Your queue place expired. Join again to take a new place."
            )
        position = list(self.tickets).index(key) + 1
        state = (
            "starting"
            if key == self.starting
            else ("ready" if ticket.offered_until is not None else "waiting")
        )
        return {
            "state": state,
            "position": position,
            "waiting": len(self.tickets),
            "turn_expires_in": max(0, math.ceil(ticket.offered_until - self.clock()))
            if ticket.offered_until is not None
            else None,
            "backend_ready": self.ready,
            "poll_after_seconds": 3,
        }

    async def visit(self, key, *, join=False):
        self.validate(key)
        async with self.lock:
            await self._refresh()
            if key not in self.tickets and join:
                if len(self.tickets) >= self.capacity:
                    raise HTTPException(
                        429, "The waiting queue is full. Please try again later."
                    )
                self.tickets[key] = Ticket(self.clock())
                await self._refresh()
            result = self._status(key)
            self.tickets[key].seen = self.clock()
            return result

    async def leave(self, key):
        self.validate(key)
        async with self.lock:
            if key == self.starting:
                raise HTTPException(409, "Your call is already connecting.")
            self.tickets.pop(key, None)
            await self._refresh()
            return {"left": True}

    async def claim(self, key):
        self.validate(key)
        async with self.lock:
            await self._refresh(fresh=True)
            status = self._status(key)
            if self.starting or status["state"] != "ready":
                raise HTTPException(409, "Please wait for your turn in the queue.")
            self.starting = key
            self.tickets[key].offered_until = None

    async def finish(self, key):
        async with self.lock:
            if self.starting == key:
                self.starting = None
                self.tickets.pop(key, None)
                # Never reuse the pre-start idle observation after success/error.
                self.checked = float("-inf")
