import json
import wave

import numpy as np

from .config import ROOT, Settings


class Catalog:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.portraits = {
            p["id"]: p
            for p in json.loads(
                (ROOT / "configs/zero_shot_portraits.json").read_text()
            )["portraits"]
        }
        self.scenarios = {
            s["id"]: s
            for s in json.loads((ROOT / "configs/zero_shot_suite.json").read_text())[
                "scenarios"
            ]
        }

    def image(self, avatar_id: str):
        portrait = self.portraits[avatar_id]
        path = (self.settings.assets_dir / portrait["path"]).resolve()
        if not path.is_relative_to(self.settings.assets_dir.resolve()):
            raise ValueError("Portrait path outside assets directory")
        return path

    def audio(self, scenario_id: str):
        if scenario_id == "interrupt_marker":
            # Sustained signal distinguishes a successful interruption from a
            # short beep simply reaching its natural end.
            t = np.arange(48_000 * 4) / 48_000
            return (np.sin(t * 2 * np.pi * 880) * 6500).astype(np.int16)
        if scenario_id == "sync_marker":
            rate = 48_000
            samples = np.zeros(rate * 4, dtype=np.int16)
            # Align onset to a 25 FPS boundary; a half-frame onset would make
            # the RMS-driven visual pulse appear 20ms ahead of the audio.
            for start in (13 * 1920, rate * 2):
                t = np.arange(rate // 10) / rate
                samples[start : start + len(t)] = (
                    np.sin(t * 2 * np.pi * 880) * 6500
                ).astype(np.int16)
            return samples
        if scenario_id not in self.scenarios:
            raise KeyError(scenario_id)
        path = self.settings.assets_dir / "audio" / f"{scenario_id}.wav"
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels(),
                source.getframerate(),
                source.getsampwidth(),
            ) != (1, 48_000, 2):
                raise ValueError("Replay requires unchanged mono 48kHz PCM16 masters")
            return np.frombuffer(
                source.readframes(source.getnframes()), dtype="<i2"
            ).copy()

    def avatars(self):
        return [
            {
                "id": name,
                "name": f"Character {i + 1:02}",
                "image_url": f"/api/avatars/{name}/image",
                "image_available": self.image(name).is_file(),
                "synthetic": bool(p["synthetic"]),
                "casting_status": "not_configured"
                if not self.settings.openai_api_key.get_secret_value()
                else "not_cast",
            }
            for i, (name, p) in enumerate(self.portraits.items())
        ]

    def cases(self):
        return [
            {
                "id": name,
                "title": name.split("_", 1)[1].replace("_", " ").title(),
                "text": " ".join(
                    s["text"] for s in case["segments"] if s["kind"] == "speech"
                ),
                "available": (
                    self.settings.assets_dir / "audio" / f"{name}.wav"
                ).is_file(),
            }
            for name, case in self.scenarios.items()
        ] + [
            {
                "id": "sync_marker",
                "title": "Sync markers",
                "text": "Two tones with matching visual pulses.",
                "available": True,
            },
            {
                "id": "interrupt_marker",
                "title": "Interruption tone",
                "text": "Four-second tone for checking interruption.",
                "available": True,
            },
        ]
