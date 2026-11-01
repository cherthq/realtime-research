"""Private localhost PCM streaming client; never silently falls back to another voice."""

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import time
from urllib.parse import urlsplit

import httpx


class QwenTTSClient:
    def __init__(self, settings):
        url = urlsplit(settings.qwen_tts_url)
        if url.scheme != "http" or url.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Qwen service must use private localhost HTTP")
        self.client = httpx.AsyncClient(
            base_url=settings.qwen_tts_url,
            timeout=httpx.Timeout(settings.qwen_tts_timeout_seconds, connect=5),
            trust_env=False,
        )
        self.busy_retries = 0
        self.busy_timeout = min(5.0, settings.qwen_tts_timeout_seconds)

    @asynccontextmanager
    async def _response(self, text, voice, instructions):
        # Cancellation closes HTTP before the GPU generator has necessarily
        # released its lease. Retry only a rejected request, never a partially
        # emitted stream: replaying that would duplicate spoken audio.
        deadline = time.monotonic() + self.busy_timeout
        while True:
            async with self.client.stream(
                "POST",
                "/synthesize",
                json={"text": text, "speaker": voice, "instruct": instructions},
            ) as response:
                if response.status_code != 409 or time.monotonic() >= deadline:
                    response.raise_for_status()
                    yield response
                    return
            self.busy_retries += 1
            await asyncio.sleep(0.05)

    async def warm(self):
        response = await self.client.post("/warm", timeout=300)
        response.raise_for_status()
        return response.json()

    async def stream(self, text, voice, instructions=""):
        completed, samples = False, 0
        async with self._response(text, voice, instructions) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "audio":
                    if event["sample_rate"] != 24000 or completed:
                        raise ValueError("Unexpected Qwen audio format")
                    pcm = base64.b64decode(event["data"], validate=True)
                    if len(pcm) % 2:
                        raise ValueError("Incomplete Qwen PCM sample")
                    samples += len(pcm) // 2
                    yield pcm
                elif event["type"] == "done":
                    if event["samples"] != samples or not samples:
                        raise ValueError("Incomplete Qwen speech")
                    completed = True
                else:
                    raise RuntimeError("Qwen speech generation failed")
        if not completed:
            raise RuntimeError("Qwen stream ended without completion")

    async def close(self):
        await self.client.aclose()
