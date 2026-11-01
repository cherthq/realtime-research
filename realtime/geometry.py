"""Process-wide native geometry. Set AVATAR_RESOLUTION before starting Python.

The API, renderer, IPC and publisher share one shape for their entire lifetime.
A resolution change requires a restart; never change it during an active call.
Keep this module dependency-free so the isolated GPU environment can import it.
"""

import os

SUPPORTED_RESOLUTIONS = {"640x480": (640, 480), "512x384": (512, 384)}


def dimensions(value: str) -> tuple[int, int]:
    try:
        return SUPPORTED_RESOLUTIONS[value]
    except KeyError:
        raise ValueError("AVATAR_RESOLUTION must be 640x480 or 512x384") from None


WIDTH, HEIGHT = dimensions(os.environ.get("AVATAR_RESOLUTION", "640x480"))
