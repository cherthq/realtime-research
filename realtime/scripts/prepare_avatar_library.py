"""Import a private demo portrait manifest and prepare stock ElevenLabs previews.

Run explicitly: python -m realtime.scripts.prepare_avatar_library manifest.json
The manifest and images are local data, separate from the frozen eval suite.
"""

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from realtime.config import Settings
from realtime.voices import DELIVERY, digest, save_json


async def prepare(manifest: Path, settings: Settings):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("Install ffmpeg to prepare stock voice samples")
    entries = json.loads(manifest.read_text(encoding="utf-8"))["portraits"]
    if not entries or len(entries) > 100:
        raise ValueError("Supply 1–100 portraits")
    library_path = settings.state_dir / "avatar-library.json"
    existing = (
        json.loads(library_path.read_text(encoding="utf-8"))["portraits"]
        if library_path.is_file()
        else []
    )
    reserved = {"ref01", "ref02", "ref03"}
    ids = set()
    for entry in entries:
        ident = entry["id"]
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ident)
            or ident in ids
            or ident in reserved
        ):
            raise ValueError("Invalid, duplicate or reserved avatar ID")
        ids.add(ident)
        with Image.open(manifest.parent / entry["image"]) as image:
            oriented = ImageOps.exif_transpose(image)
            if oriented.width * 3 != oriented.height * 4:
                raise ValueError(
                    "Supply a 4:3 portrait; no automatic cropping is applied"
                )
        with Image.open(manifest.parent / entry["image"]) as image:
            image.verify()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", entry["voice_id"]):
            raise ValueError("Invalid voice ID")

    voices = {}
    async with httpx.AsyncClient(timeout=30) as client:
        params = {"page_size": 100, "category": "premade"}
        while True:
            response = await client.get(
                "https://api.elevenlabs.io/v2/voices",
                params=params,
                headers={"xi-api-key": settings.elevenlabs_api_key.get_secret_value()},
            )
            response.raise_for_status()
            data = response.json()
            voices.update(
                {
                    v["voice_id"]: v
                    for v in data["voices"]
                    if v.get("category") == "premade"
                }
            )
            if not data.get("has_more"):
                break
            params["next_page_token"] = data["next_page_token"]
    if any(entry["voice_id"] not in voices for entry in entries):
        raise ValueError("An assigned stock voice is unavailable to this account")

    root = settings.state_dir / "voices"
    for ident in sorted({entry["voice_id"] for entry in entries}):
        voice = voices[ident]
        profile_path = root / "profiles" / f"{ident}.json"
        preview = root / "previews" / f"{ident}.wav"
        previous = (
            json.loads(profile_path.read_text(encoding="utf-8"))
            if profile_path.is_file()
            else {}
        )
        if not (
            preview.is_file()
            and previous.get("preview_source") == "provider_stock_sample"
            and previous.get("preview_source_url") == voice.get("preview_url")
            and previous.get("preview_sha256") == digest(preview.read_bytes())
        ):
            url = voice.get("preview_url", "")
            parsed = urlparse(url)
            public_storage = (
                parsed.hostname == "storage.googleapis.com"
                and parsed.path.startswith("/eleven-public-prod/")
            )
            provider_preview = (
                parsed.hostname
                in {"api.elevenlabs.io", "api.us.elevenlabs.io", "api.eu.elevenlabs.io"}
                and parsed.path == f"/v1/voices/{ident}/previews/audio"
            )
            if parsed.scheme != "https" or not (public_storage or provider_preview):
                raise ValueError("Unsupported stock preview URL")
            # Public catalog sample: no synthesis credits and no portrait upload.
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
            preview.parent.mkdir(parents=True, exist_ok=True)
            temporary = preview.with_suffix(".tmp")
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "sample.mp3"
                source.write_bytes(response.content)
                subprocess.run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(source),
                        "-ac",
                        "1",
                        "-ar",
                        "24000",
                        "-c:a",
                        "pcm_s16le",
                        "-f",
                        "wav",
                        str(temporary),
                    ],
                    check=True,
                    timeout=30,
                    capture_output=True,
                )
            if temporary.stat().st_size < 4800:
                raise ValueError("Provider stock preview was empty")
            temporary.replace(preview)
        presentation = {"male": "masculine", "female": "feminine"}.get(
            voice.get("labels", {}).get("gender"), "neutral"
        )
        save_json(
            profile_path,
            {
                "id": ident,
                "voice": ident,
                "provider": "elevenlabs",
                "name": voice["name"].split(" - ")[0],
                "instructions": DELIVERY,
                "preview_available": True,
                "preview_sha256": digest(preview.read_bytes()),
                "preview_source": "provider_stock_sample",
                "preview_source_url": voice["preview_url"],
                "preview_note": "ElevenLabs stock sample. Live delivery uses the call's selected speech model.",
                "speech_model": None,
                "review": {
                    "presentation": presentation,
                    "description": voice.get("description") or voice["name"],
                },
                "review_method": "provider_stock_catalog_metadata",
            },
        )
        print(f"Prepared stock voice: {voice['name']}", flush=True)

    portraits = []
    for entry in entries:
        target = settings.assets_dir / "images" / f"{entry['id']}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(manifest.parent / entry["image"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width * 3 != image.height * 4:
                raise ValueError(
                    "Supply a 4:3 portrait; the importer does not crop people automatically"
                )
            if image.size != (640, 480):
                image = image.resize((640, 480), Image.Resampling.LANCZOS)
            image.save(target)
        portraits.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "path": f"images/{entry['id']}.png",
                "source": "user_supplied",
                "synthetic": False,
                "real_person_reference_supplied": True,
                "image_sha256": digest(target.read_bytes()),
                "elevenlabs_voice_id": entry["voice_id"],
                "voice_note": entry["voice_note"],
                "note": entry.get("note", ""),
            }
        )
    save_json(
        library_path,
        {
            "schema_version": 1,
            "portraits": [p for p in existing if p["id"] not in ids] + portraits,
        },
    )
    print(f"Imported {len(portraits)} demo avatars; benchmark manifest unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    asyncio.run(prepare(args.manifest.resolve(), Settings()))
