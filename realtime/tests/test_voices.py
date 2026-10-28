import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from realtime.catalog import Catalog
from realtime.config import Settings
from realtime.voices import (
    CastDecision,
    VoicePreparationError,
    VoiceService,
    digest,
    save_json,
    parse_voice_review,
)


def test_review_allows_wrappers_but_rejects_ambiguous_or_invalid_content():
    valid = '{"presentation":"neutral","description":"Clear mid-range vocal delivery."}'
    assert (
        parse_voice_review(
            "Here is the JSON:\n```json\n" + valid + "\n```"
        ).presentation
        == "neutral"
    )
    for invalid in [
        "A nice voice",
        valid + valid,
        valid.replace("neutral", "unknown"),
        valid.replace("delivery.", 'delivery.","extra":"no'),
    ]:
        with pytest.raises(ValueError):
            parse_voice_review(invalid)


@pytest.fixture
def voice_service(tmp_path):
    settings = Settings(
        _env_file=None, assets_dir=tmp_path / "assets", state_dir=tmp_path / "state"
    )
    catalog = Catalog(settings)
    settings.assets_dir.joinpath("images").mkdir(parents=True)
    for avatar_id in catalog.portraits:
        Image.new("RGB", (64, 48), "navy").save(catalog.image(avatar_id))
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=CastDecision(
                status="matched",
                voice_profile_id="alloy",
                description="A relaxed proposed performance.",
            ),
            model="resolved-caster-revision",
            id="response-test",
        )

    service = VoiceService(
        settings, catalog, SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    for voice in ["alloy", "cedar"]:
        preview = service.root / "previews" / f"{voice}.wav"
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(b"test-preview-fixture")
        save_json(
            service.root / "profiles" / f"{voice}.json",
            {
                "id": voice,
                "voice": voice,
                "review": {
                    "presentation": "neutral",
                    "description": "Test fixture voice description.",
                },
                "preview_sha256": digest(preview.read_bytes()),
                "profile_sha256": voice,
                "speech_model": "test-speech-model",
                "instructions": "Test delivery.",
            },
        )
    return service, calls


def test_actual_image_is_sent_and_cast_is_cached_without_hardcoded_voice(voice_service):
    service, calls = voice_service

    async def run():
        record = await service.cast("ref01")
        assert record["voice_profile_id"] == "alloy"
        assert record["resolved_model"] == "resolved-caster-revision"
        image_part = calls[0]["input"][0]["content"][1]
        assert (
            base64.b64decode(image_part["image_url"].split(",", 1)[1])
            == service.catalog.image("ref01").read_bytes()
        )
        assert await service.cast("ref01") == record
        assert len(calls) == 1
        assert service.resolve("ref01", "cedar")["profile"]["id"] == "cedar"
        assert service.resolve("ref01")["profile"]["id"] == "alloy"

    asyncio.run(run())


def test_changed_image_requires_explicit_recast(voice_service):
    service, calls = voice_service

    async def run():
        await service.cast("ref01")
        Image.new("RGB", (64, 48), "red").save(service.catalog.image("ref01"))
        assert (await service.cast("ref01"))["status"] == "needs_recast"
        assert len(calls) == 1
        with pytest.raises(VoicePreparationError):
            service.resolve("ref01")
        assert (await service.cast("ref01", recast=True))["status"] == "matched"
        assert len(calls) == 2

    asyncio.run(run())


def test_first_call_matches_once_override_skips_cast_and_changed_inputs_require_recast(
    voice_service,
):
    service, calls = voice_service

    async def run():
        explicit = await service.resolve_for_call("ref02", "cedar")
        assert explicit["mode"] == "explicit_override" and not calls
        first = await service.resolve_for_call("ref01")
        assert first["profile"]["id"] == "alloy" and first["preparation_ms"] >= 0
        assert len(calls) == 1
        second = await service.resolve_for_call("ref01")
        assert second["casting_sha256"] == first["casting_sha256"] and len(calls) == 1
        Image.new("RGB", (64, 48), "red").save(service.catalog.image("ref01"))
        with pytest.raises(VoicePreparationError):
            await service.resolve_for_call("ref01")
        service.catalog.portraits["ref03"]["synthetic"] = False
        with pytest.raises(VoicePreparationError):
            await service.resolve_for_call("ref03")
        assert len(calls) == 1

    asyncio.run(run())


def test_unknown_provenance_needs_preference_and_never_sends_image(voice_service):
    service, calls = voice_service
    service.catalog.portraits["ref01"]["synthetic"] = False

    async def run():
        assert (await service.cast("ref01"))["status"] == "needs_preference"
        assert not calls
        await service.cast("ref01", preference="A softly spoken neutral voice")
        assert all(
            part["type"] == "input_text" for part in calls[0]["input"][0]["content"]
        )
        context = json.loads(calls[0]["input"][0]["content"][0]["text"])
        assert "authored_character" not in context

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["unknown", "refusal", "timeout"])
def test_provider_failure_never_saves_or_selects_random_voice(voice_service, kind):
    service, _ = voice_service

    async def fail(**kwargs):
        if kind == "timeout":
            raise TimeoutError("fixture")
        decision = (
            None
            if kind == "refusal"
            else CastDecision(
                status="matched",
                voice_profile_id="invented",
                description="Invalid fixture.",
            )
        )
        return SimpleNamespace(output_parsed=decision, model="test", id="test")

    service.client.responses.parse = fail
    with pytest.raises((VoicePreparationError, TimeoutError)):
        asyncio.run(service.cast("ref01"))
    assert service.saved("ref01") is None
    with pytest.raises(VoicePreparationError):
        service.resolve("ref01", "invented")


def test_preview_tampering_rejects_catalog(voice_service):
    service, _ = voice_service
    service.preview("alloy").write_bytes(b"changed")
    with pytest.raises(VoicePreparationError, match="Preview missing or changed"):
        service.profiles()


def test_frozen_pack_shares_waveforms_preserves_v1_and_resumes_without_provider(
    voice_service, monkeypatch, tmp_path
):
    import wave
    import numpy as np
    from realtime import replay

    service, _ = voice_service
    speech_calls = []

    async def speech(client, model, voice, text, instructions, speed):
        speech_calls.append(text)
        return {
            "pcm": np.full(2400, 1000, dtype=np.int16).tobytes(),
            "transcript": text,
            "resolved_model": model,
        }

    monkeypatch.setattr(replay, "synthesize_script", speech)
    before = {
        a: service.catalog.image(a).read_bytes() for a in service.catalog.portraits
    }

    async def run():
        for avatar in service.catalog.portraits:
            await service.cast(avatar)
        output = tmp_path / "v2"
        result = await replay.freeze_pack(service, output)
        assert result["status"] == "frozen" and len(result["audio"]) == 6
        assert len(speech_calls) == 9
        assert sum(len(p["audio_ids"]) for p in result["portraits"].values()) == 18
        pause = next(
            r
            for r in result["audio"].values()
            if r["scenario_id"] == "s05_pause_resume"
        )
        with wave.open(str(output / pause["path"]), "rb") as stream:
            assert (
                stream.getframerate(),
                stream.getnchannels(),
                stream.getsampwidth(),
            ) == (48000, 1, 2)
            samples = np.frombuffer(
                stream.readframes(stream.getnframes()), dtype=np.int16
            )
        assert len(samples) == pause["samples"]
        assert any(
            s["end_sample"] - s["start_sample"] == 96000
            for s in pause["segments"]
            if s["kind"] == "silence"
        )
        for segment in pause["segments"]:
            if segment["kind"] == "silence":
                assert not np.any(
                    samples[segment["start_sample"] : segment["end_sample"]]
                )
        await replay.freeze_pack(service, output)
        assert len(speech_calls) == 9
        (output / pause["path"]).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="Frozen waveform"):
            await replay.freeze_pack(service, output)
        with pytest.raises(ValueError, match="separate"):
            await replay.freeze_pack(service, service.settings.assets_dir)

    asyncio.run(run())
    assert before == {
        a: service.catalog.image(a).read_bytes() for a in service.catalog.portraits
    }
