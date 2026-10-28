"""Optional team access: opaque, revocable browser cookies; secrets stay server-side."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

COOKIE = "chert_demo_access"
TTL = 12 * 60 * 60


class DemoAccess:
    def __init__(self, secret: str):
        self.secret = secret
        self.tickets: dict[str, float] = {}

    @staticmethod
    def key(ticket: str) -> str:
        return hashlib.sha256(ticket.encode()).hexdigest()

    def authenticated(self, cookie: str = "", authorization: str = "") -> bool:
        if not self.secret:
            return True
        if authorization.startswith("Bearer ") and hmac.compare_digest(
            authorization[7:].encode(), self.secret.encode()
        ):
            return True
        return self.tickets.get(self.key(cookie), 0) > time.monotonic()

    def login(self, supplied: str) -> str | None:
        if not self.secret or not hmac.compare_digest(
            supplied.encode(), self.secret.encode()
        ):
            return None
        now = time.monotonic()
        self.tickets = {
            key: expiry for key, expiry in self.tickets.items() if expiry > now
        }
        # The single-GPU team demo does not need an unbounded authentication store.
        while len(self.tickets) >= 32:
            del self.tickets[next(iter(self.tickets))]
        ticket = secrets.token_urlsafe(32)
        self.tickets[self.key(ticket)] = now + TTL
        return ticket

    def logout(self, ticket: str):
        self.tickets.pop(self.key(ticket), None)
