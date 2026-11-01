import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from realtime.app import create_app
from realtime.catalog import Catalog
from realtime.config import Settings
from realtime.public_gateway import PUBLIC_IDS, StartLimiter, create_gateway
from realtime.voices import VoicePreparationError, VoiceService
import pytest


def test_clean_checkout_uses_only_ten_bundled_portraits(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    # Even a malformed private library is neither loaded nor exposed.
    (state / "avatar-library.json").write_text("PRIVATE CLASSROOM INVALID JSON")
    settings = Settings(
        _env_file=None,
        avatar_catalog="demo",
        state_dir=state,
        assets_dir=tmp_path / "missing-assets",
        tts_provider="elevenlabs",
        elevenlabs_voice_id="privateClone",
    )
    catalog = Catalog(settings)
    assert set(catalog.portraits) == PUBLIC_IDS
    assert all(a["image_available"] for a in catalog.avatars())
    voices = VoiceService(settings, catalog)
    assert "privateClone" not in {p["id"] for p in voices.profiles()}
    for ident in PUBLIC_IDS:
        selected = asyncio.run(voices.resolve_for_call(ident))
        assert selected["mode"] == "curated_stock_voice"
    with pytest.raises(VoicePreparationError):
        voices.profile("privateClone")
    with TestClient(create_app(settings)) as client:
        assert len(client.get("/api/avatars").json()) == 10
        for name in ("ref01", "ref02", "ref03", "call11"):
            assert client.get(f"/api/avatars/{name}/image").status_code == 404
        assert client.post("/api/avatars/call01/cast-voice", json={}).status_code == 403


def test_public_gateway_keeps_server_secret_and_blocks_admin_routes():
    received = []
    secret = "private-server-only-code"

    def upstream(request):
        received.append(request)
        assert request.headers["authorization"] == f"Bearer {secret}"
        assert "cookie" not in request.headers
        return httpx.Response(
            200, json={"status": "ok"}, headers={"Set-Cookie": "private=secret"}
        )

    origin = "https://demo.example.com"
    app = create_gateway(
        origin, "http://127.0.0.1:8080", secret, transport=httpx.MockTransport(upstream)
    )
    with TestClient(app, base_url=origin) as client:
        response = client.get(
            "/api/access", headers={"authorization": "evil", "cookie": "private=evil"}
        )
        assert response.json() == {
            "required": False,
            "authenticated": True,
            "queue_required": True,
        }
        assert secret not in response.text and "set-cookie" not in response.headers
        assert client.get("/api/avatars/call11/image").status_code == 404
        assert client.get("/api/avatars/ref01/image").status_code == 404
        assert client.get("/realtime/.env").status_code == 404
        assert (
            client.get("/api/health", headers={"Host": "evil.example"}).status_code
            == 400
        )
        assert client.post("/api/sessions", json={}).status_code == 403
        assert (
            client.post(
                "/api/sessions", headers={"Origin": "https://evil.example"}, json={}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/avatars/call01/cast-voice", headers={"Origin": origin}, json={}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/sessions",
                headers={"Origin": origin},
                json={"avatar_id": "call11"},
            ).status_code
            == 422
        )
        assert len(received) == 1


def test_gateway_forwards_scoped_session_key_and_blocks_private_catalog():
    secret = "private-code"

    def upstream(request):
        if request.url.path == "/api/avatars":
            return httpx.Response(200, json=[{"id": "call11"}])
        assert request.headers["x-session-key"] == "session-scoped-key"
        return httpx.Response(200, json={"accepted": True})

    origin = "https://demo.example.com"
    app = create_gateway(
        origin, "http://127.0.0.1:8080", secret, transport=httpx.MockTransport(upstream)
    )
    with TestClient(app, base_url=origin) as client:
        assert client.get("/api/avatars").status_code == 503
        response = client.post(
            "/api/sessions/" + "a" * 24 + "/playback",
            headers={"Origin": origin, "X-Session-Key": "session-scoped-key"},
            json={"frame_id": 1},
        )
        assert response.json() == {"accepted": True}
        assert secret not in json.dumps(dict(response.headers))


def test_public_starts_are_bounded_globally():
    limiter = StartLimiter()
    assert all(limiter.admit(t) for t in (0, 1, 2))
    assert not limiter.admit(3)
    for minute in range(1, 10):
        assert all(limiter.admit(minute * 61 + t) for t in (0, 1, 2))
    assert not limiter.admit(700)
    assert limiter.admit(3601)


def test_readiness_shares_only_status_with_website():
    origin = "https://demo.example.com"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "avatarforever_ready": True,
                "active_session": False,
                "internal": "not-public",
            },
        )
    )
    with TestClient(
        create_gateway(
            origin, "http://127.0.0.1:8080", "server-only", transport=transport
        ),
        base_url=origin,
    ) as client:
        response = client.get(
            "/ready",
            headers={"Origin": "https://dunlichan.com", "Sec-Fetch-Site": "cross-site"},
        )
        assert response.json() == {"ready": True, "busy": False}
        assert (
            response.headers["access-control-allow-origin"] == "https://dunlichan.com"
        )
        assert "not-public" not in response.text
        assert (
            client.get("/ready", headers={"Origin": "https://evil.example"}).status_code
            == 403
        )
