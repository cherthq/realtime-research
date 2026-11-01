import asyncio
import base64
import json

import pytest
from pydantic import SecretStr

from realtime.config import Settings
from realtime.elevenlabs_client import ElevenLabsTTSClient


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)

    async def send(self, text):
        body = json.loads(text)
        self.sent.append(body)
        if body.get("flush") and body.get("text"):
            self.incoming.put_nowait(
                {
                    "context_id": body["context_id"],
                    "audio": base64.b64encode(bytes(4800)).decode(),
                    "normalizedAlignment": {
                        "chars": ["Hi", " "],
                        "charStartTimesMs": [0, 70],
                        "charDurationsMs": [70, 20],
                    },
                }
            )
        if body.get("close_context"):
            self.incoming.put_nowait({"contextId": body["context_id"], "isFinal": True})

    async def close(self):
        self.closed = True


class DialogueSocket(Socket):
    def __init__(self, voice="fixtureVoice"):
        super().__init__()
        self.voice = voice
        self.registered = set()

    async def send(self, text):
        body = json.loads(text)
        self.sent.append(body)
        ident = body["context_id"]
        if "voices" in body:
            assert body["voices"] == [self.voice]
            self.registered.add(ident)
            return
        assert ident in self.registered, "Message sent to unregistered/closed context"
        assert "text" not in body, "TTS envelope sent to dialogue endpoint"
        if "inputs" in body:
            assert all(item["voice_id"] == self.voice for item in body["inputs"])
            assert body["flush"] is True
            assert all("new_turn" not in item for item in body["inputs"])
            self.incoming.put_nowait(
                {
                    "context_id": ident,
                    "audio": base64.b64encode(bytes(4800)).decode(),
                    "alignment": {
                        "chars": ["Hi", " "],
                        "char_start_times_ms": [0, 70],
                        "char_durations_ms": [70, 20],
                    },
                }
            )
        if body.get("close_context"):
            self.registered.remove(ident)
            self.incoming.put_nowait(
                {"context_id": ident, "is_final_audio_for_turn": True}
            )
            self.incoming.put_nowait({"context_id": ident, "is_final": True})


def settings(model="eleven_flash_v2_5"):
    return Settings(
        avatar_catalog="research",
        _env_file=None,
        tts_provider="elevenlabs",
        elevenlabs_api_key=SecretStr("fixture-private-key"),
        elevenlabs_voice_id="fixtureVoice",
        elevenlabs_model=model,
    )


@pytest.mark.parametrize("model", ["eleven_flash_v2_5", "eleven_v4_turbo"])
def test_portrait_specific_voice_is_isolated_per_client(model):
    async def run():
        config = settings(model)

        async def speak(voice):
            socket = DialogueSocket(voice) if "v4" in model else Socket()

            async def connector(url, **kwargs):
                if "v4" not in model:
                    assert f"/text-to-speech/{voice}/" in url
                return socket

            client = ElevenLabsTTSClient(config, connector=connector, voice_id=voice)

            async def phrases():
                yield "Hi"

            try:
                chunks = [chunk async for chunk in client.stream_response(phrases())]
                assert sum(len(chunk.pcm) for chunk in chunks) == 4800
            finally:
                await client.close()

        await asyncio.gather(speak("stockOne"), speak("stockTwo"))
        assert config.elevenlabs_voice_id == "fixtureVoice"

    asyncio.run(run())


@pytest.mark.parametrize("dialogue", [False, True])
def test_context_reuse_alignment_final_and_secret_header_only(dialogue):
    async def run():
        socket, connects = (DialogueSocket() if dialogue else Socket()), []

        async def connector(url, **kwargs):
            connects.append((url, kwargs))
            return socket

        model = "eleven_v4_turbo" if dialogue else "eleven_flash_v2_5"
        client = ElevenLabsTTSClient(settings(model), connector=connector)

        async def phrases():
            yield "Hi"

        for _ in range(2):
            chunks = [chunk async for chunk in client.stream_response(phrases())]
            assert len(b"".join(chunk.pcm for chunk in chunks)) == 4800
            assert chunks[0].cues == [(2160, "Hi")]
            assert chunks[-1].cues == [(2400, "Hi")]
        assert len(connects) == 1
        url, options = connects[0]
        assert "fixture-private-key" not in url
        assert options["additional_headers"]["xi-api-key"] == "fixture-private-key"
        assert "pcm_24000" in url and model in url
        if dialogue:
            assert "/text-to-dialogue/multi-stream-input?" in url
            assert "auto_mode" not in url and "inactivity_timeout" not in url
        contexts = {body["context_id"] for body in socket.sent}
        assert len(contexts) == (3 if dialogue else 2) and not client.contexts
        await client.close()
        assert socket.closed
        assert not client.heartbeat or client.heartbeat.done()

    asyncio.run(run())


@pytest.mark.parametrize("dialogue", [False, True])
def test_cancel_unregisters_before_provider_flush_and_next_reply_works(dialogue):
    async def run():
        socket = DialogueSocket() if dialogue else Socket()

        async def connector(*args, **kwargs):
            return socket

        model = "eleven_v4_turbo" if dialogue else "eleven_flash_v2_5"
        client = ElevenLabsTTSClient(settings(model), connector=connector)

        async def delayed():
            yield "Hi"
            await asyncio.sleep(60)

        stream = client.stream_response(delayed())
        first = await anext(stream)
        assert first.pcm
        old_id = next(iter(client.contexts))
        await stream.aclose()
        assert not client.contexts
        assert any(
            body.get("close_context") and body["context_id"] == old_id
            for body in socket.sent
        )
        socket.incoming.put_nowait(
            {"context_id": old_id, "audio": base64.b64encode(b"old").decode()}
        )

        async def next_reply():
            yield "Hi"

        chunks = [chunk async for chunk in client.stream_response(next_reply())]
        assert len(b"".join(chunk.pcm for chunk in chunks)) == 4800
        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("dialogue", [False, True])
def test_unexpected_disconnect_fails_without_replaying_or_leaking_text(dialogue):
    async def run():
        socket = DialogueSocket() if dialogue else Socket()

        async def connector(*args, **kwargs):
            return socket

        model = "eleven_v4_turbo" if dialogue else "eleven_flash_v2_5"
        client = ElevenLabsTTSClient(settings(model), connector=connector)

        async def delayed():
            yield "Hi"
            await asyncio.sleep(60)

        stream = client.stream_response(delayed())
        await anext(stream)
        socket.incoming.put_nowait(None)
        with pytest.raises(RuntimeError, match="connection ended"):
            await anext(stream)
        assert not client.contexts
        await client.close()

    asyncio.run(run())


def test_dialogue_offsets_include_audio_only_chunks_and_turn_marker_is_not_final():
    class ChunkedSocket(DialogueSocket):
        async def send(self, raw):
            body = json.loads(raw)
            if "inputs" not in body:
                return await super().send(raw)
            self.sent.append(body)
            ident = body["context_id"]
            self.incoming.put_nowait(
                {"context_id": ident, "audio": base64.b64encode(bytes(4800)).decode()}
            )
            for word in ["Hello", "world."]:
                self.incoming.put_nowait(
                    {
                        "context_id": ident,
                        "audio": base64.b64encode(bytes(4800)).decode(),
                        "alignment": {"chars": ["wrong"]},
                        "normalized_alignment": {
                            "chars": [word, " "],
                            "char_start_times_ms": [0, 70],
                            "char_durations_ms": [70, 20],
                        },
                    }
                )
                # A prosody boundary must not end the entire response stream.
                self.incoming.put_nowait(
                    {"context_id": ident, "is_final_audio_for_turn": True}
                )

    async def run():
        socket = ChunkedSocket()

        async def connector(*args, **kwargs):
            return socket

        client = ElevenLabsTTSClient(settings("eleven_v4_turbo"), connector=connector)

        async def phrases():
            yield "Hello world."

        chunks = [chunk async for chunk in client.stream_response(phrases())]
        cues = [cue for chunk in chunks for cue in chunk.cues]
        assert len(b"".join(chunk.pcm for chunk in chunks)) == 14400
        assert cues == [(4560, "Hello"), (6960, "Hello world."), (7200, "Hello world.")]
        await client.close()

    asyncio.run(run())


def test_dialogue_keeps_idle_connection_and_waiting_reply_alive_without_speech():
    async def run():
        socket = DialogueSocket()

        async def connector(*args, **kwargs):
            return socket

        client = ElevenLabsTTSClient(settings("eleven_v4_turbo"), connector=connector)
        client.KEEPALIVE_SECONDS = 0.005
        await client.warm()
        idle_id = socket.sent[0]["context_id"]
        await asyncio.sleep(0.02)
        assert any(
            b.get("keep_alive") and b["context_id"] == idle_id for b in socket.sent
        )
        assert not any("inputs" in b for b in socket.sent)

        async def phrases():
            yield "Hi"
            await asyncio.sleep(60)

        stream = client.stream_response(phrases())
        await anext(stream)
        reply_id = next(iter(client.contexts))
        await asyncio.sleep(0.02)
        assert any(
            b.get("keep_alive") and b["context_id"] == reply_id for b in socket.sent
        )
        await stream.aclose()
        count = sum(b["context_id"] == reply_id for b in socket.sent)
        await asyncio.sleep(0.02)
        assert sum(b["context_id"] == reply_id for b in socket.sent) == count
        await client.close()
        assert client.heartbeat.done()

    asyncio.run(run())


def test_empty_final_is_failure_not_a_completed_spoken_caption():
    from realtime.elevenlabs_client import ElevenLabsSpeechError

    class EmptySocket(DialogueSocket):
        async def send(self, raw):
            if "inputs" in json.loads(raw):
                return
            await super().send(raw)

    async def run():
        socket = EmptySocket()

        async def connector(*args, **kwargs):
            return socket

        client = ElevenLabsTTSClient(settings("eleven_v4_turbo"), connector=connector)

        async def phrases():
            yield "A reply that was never synthesized."

        try:
            with pytest.raises(ElevenLabsSpeechError, match="no speech audio") as error:
                async for chunk in client.stream_response(phrases()):
                    assert not chunk.cues
            assert error.value.code == "no_audio"
            assert not client.contexts
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "overage,expected_error", [(False, True), (True, False), (None, False)]
)
def test_credit_check_honors_overage_and_does_not_cache_zero_balance(
    overage, expected_error
):
    import httpx
    from realtime.elevenlabs_client import ensure_speech_credits, ElevenLabsSpeechError

    async def run():
        data = {
            "character_count": 10000,
            "character_limit": 10000,
            "can_extend_character_limit": overage,
        }

        def answer(request):
            assert request.headers["xi-api-key"] == "fixture-private-key"
            assert request.method == "GET"
            return httpx.Response(200, json=data)

        transport = httpx.MockTransport(answer)
        if expected_error:
            with pytest.raises(ElevenLabsSpeechError, match="no speech credits"):
                await ensure_speech_credits(settings(), transport=transport)
        else:
            await ensure_speech_credits(settings(), transport=transport)
        data["character_limit"] = 20000
        await ensure_speech_credits(settings(), transport=transport)

    asyncio.run(run())


def test_subscription_permission_error_does_not_deny_tts_permission():
    import httpx
    from realtime.elevenlabs_client import ensure_speech_credits

    asyncio.run(
        ensure_speech_credits(
            settings(),
            transport=httpx.MockTransport(lambda request: httpx.Response(403)),
        )
    )


def test_provider_quota_error_retains_safe_code_without_provider_text():
    from realtime.elevenlabs_client import ElevenLabsSpeechError

    class RejectedSocket(DialogueSocket):
        async def send(self, raw):
            body = json.loads(raw)
            if "inputs" in body:
                self.incoming.put_nowait(
                    {
                        "context_id": body["context_id"],
                        "type": "error",
                        "error_code": "quota_exceeded",
                        "message": "private-key private-input",
                    }
                )
                return
            await super().send(raw)

    async def run():
        socket = RejectedSocket()

        async def connector(*args, **kwargs):
            return socket

        client = ElevenLabsTTSClient(settings("eleven_v4_turbo"), connector=connector)

        async def phrases():
            yield "Private reply"

        try:
            with pytest.raises(ElevenLabsSpeechError) as error:
                async for _ in client.stream_response(phrases()):
                    pass
            assert error.value.code == "quota_exceeded"
            assert "private" not in str(error.value).lower()
        finally:
            await client.close()

    asyncio.run(run())
