"""Reproduce framing previews from fetched originals. No HTML or training approval."""

import argparse
from pathlib import Path

from .core import Config, DEFAULT_ROOT, file_hash, read_json, within, write_json
from .media import probe, render_clip, require_tools, run


MANIFEST = Path(__file__).parent / "pilot_samples" / "manifest.json"


def reproduce(root, output):
    manifest = read_json(MANIFEST)
    originals = []
    for entry in manifest["sources"]:
        source_id = "pilot_" + Path(entry["file"]).stem
        source = read_json(root / "sources" / f"{source_id}.json")
        original = within(root, source["original"])
        if file_hash(original) != entry["original_sha256"]:
            raise ValueError(f"{source_id}: original differs from published pilot; left untouched")
        originals.append(original)
    # Never overwrite a gallery, reviewed dataset, or an earlier reproduction.
    output.mkdir(parents=True, exist_ok=False)
    videos = []
    for entry, original in zip(manifest["sources"], originals):
        paths = render_clip(original, output / Path(entry["file"]).stem,
                            entry["start_seconds"], entry["end_seconds"],
                            entry["crop_xywh"], Config(), probe(original))
        videos.append(paths["video"])
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for video in videos:
        command.extend(["-i", str(video)])
    filters = []
    for index in range(len(videos)):
        filters.extend([f"[{index}:v]fps=25,setsar=1,setpts=PTS-STARTPTS[v{index}]",
                        f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[a{index}]"])
    inputs = "".join(f"[v{i}][a{i}]" for i in range(len(videos)))
    filters.append(f"{inputs}concat=n={len(videos)}:v=1:a=1[v][a]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-c:a", "aac",
                    "-movflags", "+faststart", str(output / "compilation.mp4")])
    run(command)
    write_json(output / "manifest.json", manifest)
    print(f"Framing previews only (not training-approved): {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Dataset root containing originals fetched by the normal pipeline")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; must not already exist")
    args = parser.parse_args(argv)
    require_tools()
    reproduce(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
