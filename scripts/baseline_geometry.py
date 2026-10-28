"""Shared, fail-closed geometry contract for the FaceTime comparison."""

import json
from pathlib import Path


PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs/baseline_geometry.json"


def load_geometry():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    width, height = profile["width"], profile["height"]
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or width * 3 != height * 4
        or width % 16
        or height % 16
        or profile["post_generation_resize"] is not False
    ):
        raise ValueError("Expected direct 4:3 generation aligned to 16 pixels")
    return profile


def validate_geometry(stream, profile):
    expected = (profile["width"], profile["height"])
    actual = (stream.get("width"), stream.get("height"))
    if actual != expected:
        raise ValueError(f"Output geometry {actual} differs from required {expected}")
    if stream.get("sample_aspect_ratio") not in (None, "N/A", "1:1"):
        raise ValueError("Non-square output pixels would distort the 4:3 display")
    if int(stream.get("tags", {}).get("rotate", 0)) % 360:
        raise ValueError("Output rotation would change the required landscape framing")
    for data in stream.get("side_data_list", []):
        if int(data.get("rotation", 0)) % 360:
            raise ValueError(
                "Output rotation would change the required landscape framing"
            )


def configure_flashtalk(inference, profile):
    # Upstream get_base_data resolves target_size from its own module globals.
    # Updating only the YAML-derived dictionary would leave the old portrait crop.
    inference.infer_params.update(width=profile["width"], height=profile["height"])
    inference.target_size = (profile["height"], profile["width"])


def fixed_liveavatar_size(profile):
    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        if max_area != profile["width"] * profile["height"]:
            raise ValueError("Live Avatar area and geometry profile disagree")
        return profile["height"], profile["width"]

    return get_gen_size
