import asyncio
import base64
import json

import httpx
import pytest

from realtime.config import Settings
from realtime.qwen_client import QwenTTSClient


@pytest.mark.parametrize(
    "events,valid",
    [
        (
            [
                {
                    "type": "audio",
                    "sample_rate": 24000,
                    "data": base64.b64encode(b"\x01\x00" * 8).decode(),
                },
                {"type": "done", "samples": 8},
            ],
            True,
        ),
        ([{"type": "done", "samples": 8}], False),
        ([{"type": "audio", "sample_rate": 24000, "data": "AAA="}], False),
        ([{"type": "audio", "sample_rate": 48000, "data": "AAA="}], False),
        ([{"type": "error"}], False),
    ],
)
def test_stream_requires_explicit_complete_pcm_coverage(events, valid):
    async def run():
        tts = QwenTTSClient(Settings(avatar_catalog="research", _env_file=None))
        await tts.client.aclose()
        tts.client = httpx.AsyncClient(
            base_url="http://127.0.0.1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content="\n".join(map(json.dumps, events))
                )
            ),
        )

        async def collect():
            return b"".join([part async for part in tts.stream("Test", "aiden")])

        try:
            if valid:
                assert len(await collect()) == 16
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    await collect()
        finally:
            await tts.close()

    asyncio.run(run())


def test_service_cannot_accidentally_send_text_to_an_external_endpoint():
    with pytest.raises(ValueError):
        QwenTTSClient(
            Settings(
                avatar_catalog="research",
                _env_file=None,
                qwen_tts_url="https://example.com",
            )
        )


@pytest.mark.parametrize("failure", [409, 500, "partial"])
def test_cancel_drain_is_retried_but_failed_or_partial_speech_is_not(failure):
    async def run():
        tts = QwenTTSClient(Settings(avatar_catalog="research", _env_file=None))
        await tts.client.aclose()
        calls = 0
        packet = {
            "type": "audio",
            "sample_rate": 24000,
            "data": base64.b64encode(b"\x01\x00" * 8).decode(),
        }

        def serve(request):
            nonlocal calls
            calls += 1
            if failure in (409, 500) and calls == 1:
                return httpx.Response(failure)
            events = [packet]
            if failure != "partial":
                events.append({"type": "done", "samples": 8})
            return httpx.Response(200, content="\n".join(map(json.dumps, events)))

        tts.client = httpx.AsyncClient(
            base_url="http://127.0.0.1", transport=httpx.MockTransport(serve)
        )
        try:
            if failure == 409:
                pcm = b"".join([part async for part in tts.stream("Test", "aiden")])
                assert len(pcm) == 16 and calls == 2 and tts.busy_retries == 1
            else:
                with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
                    _ = [part async for part in tts.stream("Test", "aiden")]
                assert calls == 1
        finally:
            await tts.close()

    asyncio.run(run())
