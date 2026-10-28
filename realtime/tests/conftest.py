"""Tests use generated fixtures, not ignored user media or deployment secrets."""

import json
import wave

from PIL import Image
import pytest

from realtime.config import ROOT, Settings


@pytest.fixture(scope="session")
def demo_assets(tmp_path_factory):
    assets = tmp_path_factory.mktemp("demo-assets")
    (assets / "images").mkdir()
    (assets / "audio").mkdir()
    portraits = json.loads(
        (ROOT / "configs/zero_shot_portraits.json").read_text(encoding="utf-8")
    )["portraits"]
    for i, portrait in enumerate(portraits):
        Image.new("RGB", (640, 480), (50 + i * 40, 90, 120)).save(
            assets / portrait["path"]
        )
    suite = json.loads(
        (ROOT / "configs/zero_shot_suite.json").read_text(encoding="utf-8")
    )["scenarios"]
    for scenario in suite:
        with wave.open(str(assets / "audio" / f"{scenario['id']}.wav"), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(48000 * 2))
    return assets


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path, demo_assets):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    # Keep explicit LiveKit endpoint overrides for opt-in local integration tests.
    for field in Settings.model_fields:
        if field not in {"livekit_url", "livekit_api_key", "livekit_api_secret"}:
            monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.setenv("ASSETS_DIR", str(demo_assets))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
