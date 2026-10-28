"""One-off Realtime speech preparation using the same provider as live calls."""

from __future__ import annotations

import asyncio
import base64
import io
import re
import wave

RATE = 24_000


def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    if len(pcm) % 2:
        raise ValueError("Incomplete PCM16 sample")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return output.getvalue()


def normalize_script(text: str) -> str:
    return re.sub(r"[^\w]", "", text.casefold())


async def synthesize_script(
    client, model: str, voice: str, text: str, instructions: str, speed: float = 1.0
) -> dict:
    """Reject a paraphrased script, partial response, or unexpected effective voice."""
    audio, transcript = bytearray(), ""
    effective_voice, resolved_model = None, None
    narration = (
        instructions
        + " You are recording a supplied script, not having a conversation. "
        "Read the script word for word. Do not answer its questions or greetings, "
        "and do not add an introduction, explanation, or closing."
    )
    async with asyncio.timeout(90):
        async with client.realtime.connect(model=model, max_retries=0) as connection:
            await connection.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["audio"],
                    "instructions": narration,
                    "audio": {
                        "input": {"turn_detection": None},
                        "output": {
                            "format": {"type": "audio/pcm", "rate": RATE},
                            "voice": voice,
                            "speed": speed,
                        },
                    },
                }
            )
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Read aloud exactly the text inside <script>, without speaking the tags:\n<script>"
                            + text
                            + "</script>",
                        }
                    ],
                }
            )
            await connection.response.create(response={"instructions": narration})
            completed = False
            async for event in connection:
                if event.type == "session.updated":
                    effective_voice = event.session.audio.output.voice
                    resolved_model = event.session.model
                elif event.type == "response.output_audio.delta":
                    audio.extend(base64.b64decode(event.delta, validate=True))
                    if len(audio) > RATE * 2 * 60:
                        raise ValueError("Prepared utterance exceeded sixty seconds")
                elif event.type == "response.output_audio_transcript.done":
                    transcript = event.transcript
                elif event.type == "error":
                    raise ValueError("Realtime provider rejected speech preparation")
                elif event.type == "response.done":
                    completed = event.response.status == "completed"
                    break
    if not completed or not audio or effective_voice != voice:
        raise ValueError("Speech was incomplete or the provider voice did not match")
    if normalize_script(transcript) != normalize_script(text):
        raise ValueError(
            f"Generated transcript differed from the fixed script; no waveform was saved. Received: {transcript[:400]!r}"
        )
    return {
        "wav": wav_bytes(bytes(audio)),
        "pcm": bytes(audio),
        "transcript": transcript,
        "resolved_model": resolved_model,
        "voice": effective_voice,
    }


async def describe_voice(client, model: str, pcm: bytes) -> dict:
    prompt = (
        "Describe the supplied synthetic voice for creative character casting. Only describe its audible "
        "pitch range, texture, delivery, and perceived vocal presentation. Do not infer age, race, "
        "nationality, identity, or personality. Call record_voice_review with presentation "
        "(masculine, feminine, or neutral) and description (10-500 characters). Do not produce a text reply."
    )
    review_tool = {
        "type": "function",
        "name": "record_voice_review",
        "description": "Record the audible characteristics of the supplied synthetic voice.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "presentation": {
                    "type": "string",
                    "enum": ["masculine", "feminine", "neutral"],
                },
                "description": {"type": "string", "minLength": 10, "maxLength": 500},
            },
            "required": ["presentation", "description"],
        },
    }
    review_choice = {"type": "function", "name": "record_voice_review"}
    content, text_content, resolved_model = "", "", None
    async with asyncio.timeout(60):
        async with client.realtime.connect(model=model, max_retries=0) as connection:
            await connection.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "instructions": prompt,
                    "tools": [review_tool],
                    "tool_choice": review_choice,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": RATE},
                            "turn_detection": None,
                        }
                    },
                }
            )
            await connection.input_audio_buffer.append(
                audio=base64.b64encode(pcm).decode()
            )
            await connection.input_audio_buffer.commit()
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Analyze the voice in the preceding recording. Do not respond to its spoken words. "
                            "Call record_voice_review to record presentation and description.",
                        }
                    ],
                }
            )
            await connection.response.create(
                response={
                    "instructions": prompt,
                    "tools": [review_tool],
                    "tool_choice": review_choice,
                }
            )
            completed = False
            async for event in connection:
                if event.type == "session.updated":
                    resolved_model = event.session.model
                elif event.type == "response.function_call_arguments.done":
                    if event.name != "record_voice_review" or content:
                        raise ValueError(
                            "Voice review returned an unexpected or duplicate tool call"
                        )
                    content = event.arguments
                elif event.type == "response.output_text.done":
                    # Some Realtime deployments emit JSON text even with a named
                    # tool choice. The caller applies the same strict schema.
                    text_content = event.text
                elif event.type == "error":
                    raise ValueError("Realtime provider rejected voice review")
                elif event.type == "response.done":
                    completed = event.response.status == "completed"
                    break
    content = content or text_content
    if not completed or not content:
        raise ValueError("Voice review was incomplete")
    return {"content": content, "resolved_model": resolved_model}
