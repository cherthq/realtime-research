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
from realtime.scripts import demo_host


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


def test_cloud_settings_replace_local_profile_for_both_worker_and_browser(
    tmp_path, monkeypatch
):
    profile = prepare(tmp_path, client_address="10.77.0.2")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "cloud-test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "cloud-test-secret")
    monkeypatch.setenv("LIVEKIT_BROWSER_URL", "ws://127.0.0.1:9999")
    env = environment(profile, livekit_cloud=True)
    assert (
        env["LIVEKIT_URL"]
        == env["LIVEKIT_BROWSER_URL"]
        == "wss://example.livekit.cloud"
    )
    assert env["LIVEKIT_API_KEY"] == "cloud-test-key"
    assert env["LIVEKIT_API_SECRET"] == "cloud-test-secret"
    assert env["DEMO_ACCESS_TOKEN"] == profile["access_code"]
    assert env["CONTROL_TRANSPORT"] == "poll"
    assert "cloud-test-secret" not in (tmp_path / "profile.json").read_text()
    assert environment(profile)["LIVEKIT_API_SECRET"] == profile["livekit_secret"]


@pytest.mark.parametrize(
    "url",
    [
        "ws://127.0.0.1:7880",
        "wss://livekit.cloud.evil.example",
        "wss://user:secret@example.livekit.cloud",
        "wss://example.livekit.cloud?token=private",
    ],
)
def test_cloud_mode_rejects_invalid_project_urls_without_echoing_values(
    tmp_path, monkeypatch, url
):
    profile = prepare(tmp_path)
    monkeypatch.setenv("LIVEKIT_URL", url)
    monkeypatch.setenv("LIVEKIT_API_KEY", "cloud-test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "cloud-test-secret")
    with pytest.raises(ValueError, match="Configure LIVEKIT_URL") as error:
        environment(profile, livekit_cloud=True)
    assert url not in str(error.value) and "cloud-test-secret" not in str(error.value)


def test_cloud_serve_only_launches_api_and_never_installs_local_livekit(
    tmp_path, monkeypatch
):
    directory = tmp_path / "host"
    profile = prepare(directory)
    build = tmp_path / "realtime/web/dist/index.html"
    build.parent.mkdir(parents=True)
    build.write_text("test")
    monkeypatch.setattr(demo_host, "ROOT", tmp_path)
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "cloud-test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "cloud-test-secret")
    monkeypatch.setattr(
        demo_host, "install", lambda: pytest.fail("local LiveKit installed")
    )
    calls, bound = [], []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def setsockopt(self, *_):
            pass

        def bind(self, address):
            bound.append(address)

    class Child:
        def poll(self):
            return 1  # simulate a service exiting after the mocked readiness check

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setattr(demo_host.socket, "socket", Probe)
    monkeypatch.setattr(demo_host.subprocess, "Popen", spawn)
    monkeypatch.setattr(demo_host, "wait_http", lambda *_: None)
    with pytest.raises(RuntimeError, match="service exited"):
        demo_host.serve(directory, livekit_cloud=True)
    assert bound == [("127.0.0.1", profile["api_port"])]
    assert len(calls) == 1 and "uvicorn" in calls[0][0]
    assert calls[0][1]["env"]["LIVEKIT_URL"] == "wss://example.livekit.cloud"
    assert not (directory / "livekit.log").exists()


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
