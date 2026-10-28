import json
import os
import io

import pytest

from realtime.scripts.demo_host import (
    environment,
    livekit_config,
    load,
    ports,
    prepare,
    tunnel_command,
)
from realtime.scripts.demo_host import warm


def test_private_profile_is_stable_and_tunnels_all_three_ports(tmp_path):
    profile = prepare(tmp_path)
    assert load(tmp_path) == profile
    with pytest.raises(FileExistsError):
        prepare(tmp_path)
    assert load(tmp_path) == profile
    assert len(profile["access_code"]) >= 32
    assert livekit_config(profile)["rtc"]["force_tcp"]
    assert not livekit_config(profile)["rtc"]["use_external_ip"]
    env = environment(profile)
    assert env["DEMO_ACCESS_TOKEN"] == profile["access_code"]
    assert env["LIVEKIT_API_SECRET"] == profile["livekit_secret"]
    command = tunnel_command("ubuntu@host.example", tmp_path / "key with spaces")
    assert command.count("-L") == 3
    assert all(
        f"127.0.0.1:{port}:127.0.0.1:{port}" in command for port in (8080, 7880, 7881)
    )
    assert (
        command[-1] == "ubuntu@host.example"
        and "StrictHostKeyChecking=no" not in command
    )


def test_invalid_or_public_profiles_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        ports(8080, 8080, 7881)
    with pytest.raises(ValueError):
        tunnel_command("-oProxyCommand=bad", None)
    prepare(tmp_path)
    path = tmp_path / "livekit.yaml"
    modified = json.loads(path.read_text())
    modified["bind_addresses"] = ["0.0.0.0"]
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="differs"):
        load(tmp_path)
    if os.name == "posix":
        path.chmod(0o644)
        with pytest.raises(ValueError, match="permissions"):
            load(tmp_path)


def test_private_wsl_browser_route_preserves_server_loopback(tmp_path):
    profile = prepare(tmp_path, 19881, 19882, 19883, "10.77.0.2")
    assert load(tmp_path) == profile
    rtc = livekit_config(profile)["rtc"]
    assert rtc["node_ip"] == "10.77.0.2" and rtc["advertise_internal_ip"]
    env = environment(profile)
    assert env["LIVEKIT_URL"] == "ws://127.0.0.1:19882"
    assert env["LIVEKIT_BROWSER_URL"] == "ws://10.77.0.2:19882"
    assert env["CONTROL_TRANSPORT"] == "poll"


@pytest.mark.parametrize(
    "address", ["8.8.8.8", "0.0.0.0", "127.0.0.1", "::1", "host.example"]
)
def test_browser_route_rejects_public_or_invalid_addresses(tmp_path, address):
    with pytest.raises(ValueError):
        prepare(tmp_path, client_address=address)


def test_warm_releases_its_lease_when_readiness_check_fails(tmp_path, monkeypatch):
    prepare(tmp_path)
    calls = []
    replies = iter(
        [
            {},
            {"active_session": False},
            {"id": "prepared", "session_key": "private-key"},
            {"avatarforever": {"loaded": False}},
            {},
        ]
    )

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            return io.BytesIO(json.dumps(next(replies)).encode())

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError, match="readiness"):
        warm(tmp_path)
    assert calls[-1].get_method() == "DELETE"
    assert calls[-1].full_url.endswith("/api/sessions/prepared")
    assert calls[-1].get_header("X-session-key") == "private-key"


def test_warm_does_not_interfere_with_an_existing_call(tmp_path, monkeypatch):
    prepare(tmp_path)
    calls = []
    replies = iter([{}, {"active_session": True}])

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            return io.BytesIO(json.dumps(next(replies)).encode())

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError, match="call is active"):
        warm(tmp_path)
    assert not any("/sessions" in request.full_url for request in calls)
