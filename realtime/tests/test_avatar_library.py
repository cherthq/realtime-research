import asyncio
import json

import pytest
from PIL import Image

from realtime.catalog import Catalog
from realtime.config import ROOT, Settings
from realtime.voices import VoicePreparationError, VoiceService, digest, save_json


@pytest.fixture
def library(tmp_path):
    settings = Settings(
        avatar_catalog="research",
        _env_file=None,
        assets_dir=tmp_path / "assets",
        state_dir=tmp_path / "state",
        tts_provider="elevenlabs",
        elevenlabs_voice_id="hostVoice",
    )
    image = settings.assets_dir / "images" / "call01.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (640, 480), "navy").save(image)
    portrait = {
        "id": "call01",
        "name": "Desk call",
        "path": "images/call01.png",
        "synthetic": False,
        "image_sha256": digest(image.read_bytes()),
        "elevenlabs_voice_id": "stockVoice",
        "voice_note": "Warm conversational delivery.",
    }
    save_json(settings.state_dir / "avatar-library.json", {"portraits": [portrait]})
    preview = settings.state_dir / "voices/previews/stockVoice.wav"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"fixture audio")
    save_json(
        settings.state_dir / "voices/profiles/stockVoice.json",
        {
            "id": "stockVoice",
            "voice": "stockVoice",
            "provider": "elevenlabs",
            "preview_available": True,
            "preview_sha256": digest(preview.read_bytes()),
            "review": {
                "presentation": "neutral",
                "description": "A warm conversational stock voice.",
            },
        },
    )
    return settings


def test_private_library_extends_catalog_without_changing_benchmark(library):
    frozen = json.loads((ROOT / "configs/zero_shot_portraits.json").read_text())[
        "portraits"
    ]
    catalog = Catalog(library)
    assert len(catalog.avatars()) == len(frozen) + 1
    assert [catalog.portraits[p["id"]] for p in frozen] == frozen
    avatar = catalog.avatars()[-1]
    assert avatar["name"] == "Desk call" and avatar["image_available"]
    assert avatar["synthetic"] is False


@pytest.mark.parametrize(
    "change", [{"id": "ref01"}, {"id": "../escape"}, {"path": "../../escape.png"}]
)
def test_library_rejects_reserved_ids_and_path_escape(library, change):
    path = library.state_dir / "avatar-library.json"
    data = json.loads(path.read_text())
    data["portraits"][0].update(change)
    save_json(path, data)
    with pytest.raises(ValueError):
        Catalog(library)


def test_portrait_voice_override_and_changed_image(library):
    service = VoiceService(library, Catalog(library))

    async def run():
        matched = await service.resolve_for_call("call01")
        assert matched["mode"] == "curated_stock_voice"
        assert matched["profile"]["voice"] == "stockVoice"
        explicit = await service.resolve_for_call("call01", "hostVoice")
        assert explicit["mode"] == "explicit_override"
        assert explicit["profile"]["voice"] == "hostVoice"
        assert (await service.resolve_for_call("ref01"))["profile"][
            "voice"
        ] == "hostVoice"
        assert library.elevenlabs_voice_id == "hostVoice"
        with pytest.raises(VoicePreparationError):
            await service.resolve_for_call("call01", "unpreparedVoice")
        service.catalog.image("call01").write_bytes(b"changed image")
        with pytest.raises(VoicePreparationError, match="Portrait changed"):
            await service.resolve_for_call("call01")
        assert (await service.resolve_for_call("call01", "stockVoice"))[
            "mode"
        ] == "explicit_override"

    asyncio.run(run())
    service.preview("stockVoice").write_bytes(b"changed audio")
    with pytest.raises(VoicePreparationError, match="preview missing or changed"):
        service.profiles()


def test_text_agent_passes_selected_voice_to_speech_client(library):
    from realtime.text_voice_agent import TextVoiceAgent
    from realtime.tests.test_voice_stream import Sink

    agent = TextVoiceAgent(library, {"voice": "stockVoice"}, Sink())
    assert agent.tts.voice_id == "stockVoice"
    assert library.elevenlabs_voice_id == "hostVoice"
