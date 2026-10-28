import importlib
import time

from fastapi.testclient import TestClient
from pydantic import SecretStr

from realtime.access import COOKIE, DemoAccess
from realtime.config import Settings


def test_private_api_login_cookie_logout_and_cross_origin():
    module = importlib.import_module("realtime.app")
    config = Settings(_env_file=None, demo_access_token=SecretStr("private-test-code"))
    with TestClient(module.create_app(config)) as client:
        assert client.get("/api/access").json() == {
            "required": True,
            "authenticated": False,
        }
        assert client.get("/api/avatars").status_code == 401
        assert client.post("/api/access", json={"token": "wrong"}).status_code == 401
        assert (
            client.post(
                "/api/access",
                json={"token": "private-test-code"},
                headers={"Origin": "http://localhost:9000"},
            ).status_code
            == 403
        )
        reply = client.post("/api/access", json={"token": "private-test-code"})
        assert reply.status_code == 200
        assert (
            "HttpOnly" in reply.headers["set-cookie"]
            and "SameSite=strict" in reply.headers["set-cookie"]
        )
        assert "private-test-code" not in reply.text + reply.headers["set-cookie"]
        ticket = client.cookies.get(COOKIE)
        assert client.get("/api/access").json()["authenticated"]
        assert client.get("/api/avatars").status_code == 200
        # Normal image/audio element requests do not carry custom auth headers.
        assert client.get("/api/avatars/ref01/image").status_code == 200
        assert client.delete("/api/access").status_code == 200
        assert (
            client.get(
                "/api/avatars", headers={"Cookie": f"{COOKIE}={ticket}"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/api/avatars", headers={"Authorization": "Bearer private-test-code"}
            ).status_code
            == 200
        )


def test_ticket_expiry_invalid_tokens_and_bounded_store(monkeypatch):
    access = DemoAccess("valid-code")
    mark = time.monotonic()
    monkeypatch.setattr("realtime.access.time.monotonic", lambda: mark)
    first = access.login("valid-code")
    assert access.authenticated(first) and not access.authenticated("made-up")
    assert access.login("🌻") is None
    for _ in range(32):
        access.login("valid-code")
    assert len(access.tickets) == 32 and not access.authenticated(first)
    last = access.login("valid-code")
    mark += 12 * 60 * 60 + 1
    assert not access.authenticated(last)
