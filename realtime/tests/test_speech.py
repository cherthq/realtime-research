import asyncio
import base64
from types import SimpleNamespace as Obj

import pytest

from realtime.speech import describe_voice, synthesize_script


class Connection:
    def __init__(self, *, voice="cedar", transcript="Hello there.", status="completed"):
        self.requests = []
        self.events = [
            Obj(
                type="session.updated",
                session=Obj(
                    model="resolved-test-model", audio=Obj(output=Obj(voice=voice))
                ),
            ),
            Obj(
                type="response.output_audio.delta",
                delta=base64.b64encode(bytes(4800)).decode(),
            ),
            Obj(type="response.output_audio_transcript.done", transcript=transcript),
            Obj(type="response.done", response=Obj(status=status)),
        ]

        async def record(**kwargs):
            self.requests.append(kwargs)

        self.session = Obj(update=record)
        self.conversation = Obj(item=Obj(create=record))
        self.response = Obj(create=record)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        async def events():
            for event in self.events:
                yield event

        return events()


@pytest.mark.parametrize(
    "change", [{"voice": "marin"}, {"transcript": "Hi!"}, {"status": "incomplete"}]
)
def test_preparation_rejects_voice_changes_paraphrases_and_partial_speech(change):
    connection = Connection(**change)
    client = Obj(realtime=Obj(connect=lambda **kwargs: connection))
    with pytest.raises(ValueError):
        asyncio.run(
            synthesize_script(client, "test", "cedar", "Hello there.", "Relaxed.")
        )


def test_preparation_uses_actual_effective_model_and_explicit_pcm():
    connection = Connection()
    client = Obj(realtime=Obj(connect=lambda **kwargs: connection))
    result = asyncio.run(
        synthesize_script(client, "test", "cedar", "Hello there.", "Relaxed.")
    )
    assert result["resolved_model"] == "resolved-test-model"
    assert result["voice"] == "cedar" and result["wav"][:4] == b"RIFF"
    assert connection.requests[0]["session"]["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert (
        "<script>Hello there.</script>"
        in connection.requests[1]["item"]["content"][0]["text"]
    )
    assert (
        connection.requests[2]["response"]["instructions"]
        == connection.requests[0]["session"]["instructions"]
    )


@pytest.mark.parametrize("result", ["valid", "text_only", "wrong_tool", "duplicate"])
def test_voice_review_accepts_json_text_or_one_named_tool_call(result):
    connection = Connection()

    async def record(**kwargs):
        connection.requests.append(kwargs)

    connection.input_audio_buffer = Obj(append=record, commit=record)
    content = (
        '{"presentation":"neutral","description":"Smooth, mid-range vocal delivery."}'
    )
    event = Obj(
        type="response.function_call_arguments.done",
        name="record_voice_review",
        arguments=content,
    )
    if result == "wrong_tool":
        event.name = "other"
    connection.events = [connection.events[0]]
    if result == "text_only":
        connection.events.append(Obj(type="response.output_text.done", text=content))
    else:
        connection.events.extend([event] * (2 if result == "duplicate" else 1))
    connection.events.append(
        Obj(type="response.done", response=Obj(status="completed"))
    )
    client = Obj(realtime=Obj(connect=lambda **kwargs: connection))
    if result not in {"valid", "text_only"}:
        with pytest.raises(ValueError):
            asyncio.run(describe_voice(client, "test", bytes(4800)))
        return
    review = asyncio.run(describe_voice(client, "test", bytes(4800)))
    assert review["content"] == content
    assert connection.requests[-1]["response"]["tool_choice"] == {
        "type": "function",
        "name": "record_voice_review",
    }
