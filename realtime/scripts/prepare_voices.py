"""Explicit paid API preparation; run as python -m realtime.scripts.prepare_voices."""

import argparse
import asyncio
from pathlib import Path

from realtime.catalog import Catalog
from realtime.config import settings
from realtime.voices import VOICES, VoiceService
from realtime.replay import freeze_pack


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voices",
        nargs="+",
        choices=sorted(VOICES),
        default=["cedar", "marin", "alloy"],
    )
    parser.add_argument("--rebuild-previews", action="store_true")
    parser.add_argument(
        "--cast",
        action="store_true",
        help="Match all three documented synthetic portraits",
    )
    parser.add_argument("--recast", action="store_true")
    parser.add_argument(
        "--freeze-pack",
        type=Path,
        help="Generate/resume a separate frozen v2 replay pack after casting",
    )
    args = parser.parse_args()
    service = VoiceService(settings, Catalog(settings))
    try:
        for voice in args.voices:
            profile = await service.prepare(voice, replace=args.rebuild_previews)
            print(f"Prepared {voice}: {profile['review']['description']}")
        if args.cast:
            for avatar_id in service.catalog.portraits:
                result = await service.cast(avatar_id, recast=args.recast)
                print(
                    f"{avatar_id}: {result['status']} / {result.get('voice_profile_id')}"
                )
        if args.freeze_pack:
            result = await freeze_pack(service, args.freeze_pack)
            print(
                f"Frozen pack: {args.freeze_pack} ({len(result['audio'])} distinct waveforms)"
            )
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
