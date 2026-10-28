"""Explicitly download pinned inference weights, retaining remote license files."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


COMPONENTS = {
    "avatarforever": [
        ("LetsThink/AvatarForever", "622ea82482c09e7fb2d8e29cb7ff1d009c820f05", "AvatarForever"),
        ("google/gemma-3-12b-it-qat-q4_0-unquantized", "68f7ee4fbd59087436ada77ed2d62f373fdd4482", "gemma-3-12b-it-qat-q4_0-unquantized"),
    ],
    "flashtalk": [
        (
            "Soul-AILab/SoulX-FlashTalk-14B",
            "f3fda6499a6022862cadd39fc41ae63b64706e1b",
            "SoulX-FlashTalk-14B",
        ),
        (
            "TencentGameMate/chinese-wav2vec2-base",
            "3991242c806928916fff4a8c0e4f76acf661b743",
            "chinese-wav2vec2-base",
        ),
    ],
    "liveavatar": [
        (
            "Wan-AI/Wan2.2-S2V-14B",
            "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",
            "Wan2.2-S2V-14B",
        ),
        (
            "Quark-Vision/Live-Avatar",
            "92cdccd12a91e8a63767a7c821b7c75e51d5a172",
            "Live-Avatar",
        ),
    ],
    "liveact": [
        ("Soul-AILab/LiveAct", "1bcdef1d8941edf2a9a6e1d21f5657f0a69fcd37", "LiveAct"),
        (
            "TencentGameMate/chinese-wav2vec2-base",
            "3991242c806928916fff4a8c0e4f76acf661b743",
            "chinese-wav2vec2-base",
        ),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=COMPONENTS)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for repo, revision, directory in COMPONENTS[args.model]:
        destination = args.root / directory
        print(f"Downloading {repo}@{revision} -> {destination}", flush=True)
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=destination,
            max_workers=4,
            ignore_patterns=["assets/*", "*-fairseq-ckpt.pt"],
        )
        records.append({"repo_id": repo, "revision": revision, "directory": directory})
    (args.root / f"{args.model}-revisions.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print("Pinned checkpoint download complete", flush=True)


if __name__ == "__main__":
    main()
