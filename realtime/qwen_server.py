"""Single-owner Qwen3-TTS GPU service, bound to localhost in a separate environment.

Load after AvatarForever's text encoder has been offloaded. Uses released Qwen
weights and a pinned MIT streaming runtime; no provider audio API is involved.
"""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import queue
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
SPEAKERS = {
    "aiden",
    "ryan",
    "serena",
    "vivian",
    "uncle_fu",
    "dylan",
    "eric",
    "ono_anna",
    "sohee",
}
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-gpu")
model = None
speech_stream = None
busy = False
last_timing = {}


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    speaker: str = "aiden"
    instruct: str = Field(
        default="Speak naturally and conversationally.", max_length=1000
    )


def load_model():
    global model, speech_stream
    if model is None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        priority = int(os.environ.get("QWEN_CUDA_STREAM_PRIORITY", "0"))
        if priority not in {-1, 0}:
            raise ValueError("QWEN_CUDA_STREAM_PRIORITY must be -1 or 0")
        path = Path(os.environ.get("QWEN_TTS_WEIGHTS", "checkpoints/qwen-tts"))
        candidate = FasterQwen3TTS.from_pretrained(
            str(path), device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        candidate.warmup(prefill_len=100)
        if priority:
            # CUDA priority is a scheduling hint, not preemption. Keep captured
            # graphs and weights alive on the single owning inference thread.
            # Wait for their initialization before replaying on another stream.
            speech_stream = torch.cuda.Stream(priority=priority)
            speech_stream.wait_stream(torch.cuda.current_stream())
            speech_stream.synchronize()
        model = candidate
    return {"loaded": True, "model": MODEL_ID, "revision": MODEL_REVISION}


@app.get("/health")
async def health():
    return {
        "loaded": model is not None,
        "busy": busy,
        "model": MODEL_ID,
        "cuda_stream_priority": speech_stream.priority if speech_stream else 0,
        "last_timing": last_timing,
    }


@app.post("/warm")
async def warm():
    return await asyncio.get_running_loop().run_in_executor(executor, load_model)


def speech_chunks(**kwargs):
    import torch

    # The context spans iteration, including generator.close() on cancellation;
    # constructing a lazy generator inside the context alone would do no work.
    with torch.cuda.stream(speech_stream) if speech_stream else nullcontext():
        yield from model.generate_custom_voice_streaming(**kwargs)


@app.post("/synthesize")
async def synthesize(request: SpeechRequest):
    global busy
    if request.speaker.lower() not in SPEAKERS:
        raise HTTPException(422, "Unknown Qwen speaker")
    if busy:
        raise HTTPException(409, "Qwen worker is busy")
    busy = True
    stopped = threading.Event()
    outgoing = queue.Queue(maxsize=8)

    def put(event):
        while not stopped.is_set():
            try:
                outgoing.put(event, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def generate():
        global busy, last_timing
        import numpy as np

        started, first, samples = time.monotonic(), None, 0
        stream = None
        try:
            load_model()
            stream = speech_chunks(
                text=request.text,
                language="English",
                speaker=request.speaker.lower(),
                instruct=request.instruct,
                # Cover 1.28 s of video plus 0.28 s of conditioning lookahead.
                # 21 codec steps also cover the 24->48 kHz resampler delay;
                # 20 steps would make the renderer wait for the next TTS batch.
                chunk_size=21,
            )
            for audio, rate, timing in stream:
                if stopped.is_set():
                    break
                if rate != 24000:
                    raise ValueError("Unsupported Qwen sample rate")
                audio = np.asarray(audio).reshape(-1)
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                if not pcm:
                    continue
                if first is None:
                    first = time.monotonic()
                samples += len(pcm) // 2
                if not put(
                    {
                        "type": "audio",
                        "sample_rate": rate,
                        "data": base64.b64encode(pcm).decode(),
                    }
                ):
                    break
            if not stopped.is_set():
                put({"type": "done", "samples": samples})
        except Exception:
            import traceback

            traceback.print_exc()
            put({"type": "error"})
        finally:
            if stream is not None:
                stream.close()
            elapsed = time.monotonic() - started
            last_timing = {
                "first_audio_ms": (first - started) * 1000 if first else None,
                "seconds": elapsed,
                "audio_seconds": samples / 24000,
                "rtf": elapsed / (samples / 24000) if samples else None,
                "cancelled": stopped.is_set(),
            }
            busy = False

    future = asyncio.get_running_loop().run_in_executor(executor, generate)

    async def events():
        try:
            while True:
                try:
                    event = outgoing.get_nowait()
                except queue.Empty:
                    if future.done():
                        await future
                        break
                    await asyncio.sleep(0.005)
                    continue
                yield json.dumps(event, separators=(",", ":")) + "\n"
                if event["type"] in {"done", "error"}:
                    break
        finally:
            stopped.set()

    return StreamingResponse(events(), media_type="application/x-ndjson")
