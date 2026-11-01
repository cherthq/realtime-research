"""Resolution negotiation is tested in fresh processes, just like deployment."""

import json
import os
import subprocess
import sys

import pytest

from realtime.geometry import dimensions


@pytest.mark.parametrize("value", ["640x480", "512x384"])
def test_native_contracts_and_real_ipc_agree(value):
    code = """
import asyncio, json, tempfile, sys
from pathlib import Path
import numpy as np
from realtime.contracts import WIDTH, HEIGHT, AudioChunk, FRAME_SAMPLES, MediaChunk
from realtime.streaming import GeneratedBatch, capabilities
from realtime.gpu_client import GPUProcess

async def run():
    with tempfile.TemporaryDirectory() as directory:
        image=Path(directory)/"portrait.bin"; image.write_bytes(b"fixture")
        gpu=GPUProcess([sys.executable,"-m","realtime.tests.fake_gpu"],Path(directory)/"gpu.log")
        try:
            await gpu.start()
            await gpu.open_session("resolution-test",image)
            a=AudioChunk("geometry",0,0,np.zeros(25*FRAME_SAMPLES,np.int16),25*FRAME_SAMPLES)
            batch=await gpu.generate(a)
            assert batch.frames.shape==(25,HEIGHT,WIDTH,3)
            MediaChunk(a,batch.frames[0])
            assert gpu.info["geometry"]=={"width":WIDTH,"height":HEIGHT,"fps":25}
            assert capabilities("avatarforever")["geometry"]=={"width":WIDTH,"height":HEIGHT,"fps":25,"native":True}
            wrong=np.zeros((25,384 if HEIGHT==480 else 480,512 if WIDTH==640 else 640,3),np.uint8)
            try: GeneratedBatch(a,wrong,.1)
            except ValueError: pass
            else: raise AssertionError("Wrong native shape accepted")
            print(json.dumps({"width":WIDTH,"height":HEIGHT,"fps_cap":capabilities("avatarforever")["video_max_fps"]}))
        finally: await gpu.close()
asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "AVATAR_RESOLUTION": value},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(result.stdout)
    assert (result["width"], result["height"]) == dimensions(value)
    assert result["fps_cap"] == 24


@pytest.mark.parametrize("value", ["576x432", "480x640", "0x0", "1920x1080", ""])
def test_unsupported_geometry_fails_before_allocation(value):
    with pytest.raises(ValueError, match="AVATAR_RESOLUTION"):
        dimensions(value)
