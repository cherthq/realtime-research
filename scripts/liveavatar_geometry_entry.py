"""Apply the explicit generation shape without editing the pinned vendor tree."""

from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor/liveavatar"))

from scripts.baseline_geometry import fixed_liveavatar_size, load_geometry  # noqa: E402
from scripts.liveavatar_cache import install_cache_guard  # noqa: E402


def main():
    profile = load_geometry()
    from liveavatar.models.wan.wan_2_2 import configs
    from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
    from liveavatar.models.wan.causal_model_s2v import CausalWanS2VSelfAttention

    install_cache_guard(CausalWanS2VSelfAttention)

    key = f"{profile['width']}*{profile['height']}"
    configs.SIZE_CONFIGS[key] = (profile["width"], profile["height"])
    configs.MAX_AREA_CONFIGS[key] = profile["width"] * profile["height"]
    configs.SUPPORTED_SIZES["s2v-14B"] = (*configs.SUPPORTED_SIZES["s2v-14B"], key)
    WanS2V.get_gen_size = fixed_liveavatar_size(profile)
    print(f"Direct generation geometry: {key}; no output resize/crop", flush=True)
    runpy.run_path(
        str(REPO / "vendor/liveavatar/minimal_inference/s2v_streaming_interact.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
