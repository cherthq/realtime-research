# Conversation and voice tuning

This tuning keeps the configured conversation model, ElevenLabs model and avatar
voices. It changes how replies are written and how text reaches the speech
provider. It does not change AvatarForever, video timing or interruption handling.

## Controls

The conversation instructions in `realtime/voice_agent.py` favor a direct,
specific response in one or two short sentences, with contractions and ordinary
spoken language. Relevant follow-up questions are allowed, but generic praise,
service-style closers and unrequested advice are discouraged. Requested detailed
answers remain available. English is still the default; there is no application
output-token ceiling. The agent must remain honest about being AI and what it
can actually see or remember.

Two optional backend settings control speech delivery:

```dotenv
# Omit this line to keep the provider's saved voice default.
ELEVENLABS_STABILITY=0.4
# Maximum wait for more text after an unfinished clause. Default: 120 ms.
ELEVENLABS_PHRASE_WAIT_MS=120
```

Stability is sent once, when each speaking context is registered. A value of
`0.4` is an experimental starting point, not a universally better voice setting.
Evaluate each voice separately. Other REST or playground controls are not
automatically assumed to work with the dialogue WebSocket.

The phrase buffer joins nearby clauses before forcing speech generation. It
keeps the existing 180-character aggregation bound, text order and finish
sentinel. Complete sentences and completed replies do not wait. An unfinished
clause waits at most the configured deadline, which does not restart as more
fragments arrive. Cancellation stops that wait. The Qwen path keeps its existing
zero-wait behavior.

To undo the speech tuning, unset `ELEVENLABS_STABILITY` and set
`ELEVENLABS_PHRASE_WAIT_MS=0`, then restart the backend between calls. Changing
these settings does not change the saved ElevenLabs voices.

## Experiments on 3 November 2026

The local probes used `gpt-realtime` and `eleven_v4_turbo`; models and voice
assignments were held fixed.

### Reply writing

Two exploratory passes each compared 12 caller inputs with the old and candidate
instructions. Cases covered greetings, personal updates, frustration, boredom,
an explanation, a correction, AI identity, unavailable camera input, requested
detail, a goodbye, a contextual follow-up and a disagreement.

The second pass included the application's explicit per-turn instruction that
no fresh camera image was available. The first pass omitted it and is not a
valid test of the deployed camera behavior. All second-pass generations
completed. These are small, unblinded examples, not a naturalness score or proof
of consistent improvement.

For “No, I meant tomorrow, not today,” the old prompt produced:

> Got it, you’re talking about tomorrow. Thanks for clarifying! Anything else you want to check for tomorrow?

The candidate produced:

> Oh, tomorrow. Got it. Thanks for clearing that up.

The candidate also asked a more specific troubleshooting question for a crashing
app. However, generic closers and unsolicited advice still appeared in some
cases, and one contextual reply assumed preparation that the caller had not
mentioned. Prompting reduces these tendencies; it does not guarantee natural
conversation or eliminate unsupported assumptions. Longer conversational trials
and listener feedback remain necessary.

### Speech delivery

Three stock voices (Brick room, Terrace and Desk call) read the same two-sentence
reply with a fixed replay of text arrival times. Four settings were compared:
saved defaults, stability `0.4` alone, the 120 ms phrase buffer alone, and both
changes. The comparison was repeated in reverse order: 24 samples total. A
separate audio-tag probe added one more sample.

| Setting | Median time to first audio | Forced phrase flushes |
| --- | ---: | ---: |
| Saved defaults, no wait | 158 ms | 3 |
| Stability 0.4, no wait | 132 ms | 3 |
| Saved defaults, 120 ms clause wait | 198 ms | 2 |
| Stability 0.4, 120 ms clause wait | 173 ms | 2 |

These timings start at the synthetic text producer on an already connected
socket. They exclude transcription, conversation generation, avatar rendering,
network playback and device output. Six samples per setting, network variation
and a cold outlier make them unsuitable for claiming a speed advantage from
stability. The buffer reduced forced flushes from three to two in this replay;
its real tradeoff is a bounded wait for more linguistic context.

All 25 samples returned audio and complete expected provider alignment text,
with no full-scale PCM clipping. Alignment is provider metadata, not independent
speech recognition or a listening assessment. These checks do not establish
which sample sounds most natural. A local A/B page retains the actual recordings
for listening; generated recordings and raw experiment output stay Git-ignored.

The audio-tag probe returned the tag in alignment text. Automatic tags, fake
laughs and filler insertion were therefore not enabled. They need separate
caption handling and perceptual evaluation before use.

## Why these changes

- The [Realtime prompting guide](https://developers.openai.com/cookbook/examples/realtime_prompting_guide)
  recommends explicit tone, length and example guidance. We apply this to the
  existing model rather than adding another response-rewriting model and another
  inference step.
- The [ElevenLabs multi-context protocol](https://elevenlabs.io/docs/api-reference/text-to-dialogue/ttd-multi-websocket)
  registers voice settings on the first context message. It buffers text and
  generates on flush; a new turn creates a prosody boundary. We retain one
  context per reply and reduce premature clause flushes.
- [Pipecat's dialogue integration](https://reference-server.pipecat.ai/en/latest/_modules/pipecat/services/elevenlabs/dialogue/tts.html)
  provides a concrete reference for stability on the dialogue WebSocket. General
  TTS sliders should not be copied blindly between API endpoints.
- [ElevenLabs' speech guidance](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)
  covers punctuation, delivery cues and voice-dependent results. We keep ordinary
  punctuation and test voices individually instead of injecting random emotion.

## Verification

Regression coverage checks context registration, omitted/default stability,
phrase ordering, final text, bounded waiting and cancellation. Existing tests
cover interruption, transcript timing and search turns. Provider smoke tests
exercise actual synthesis, but human listening remains the acceptance check for
naturalness. Do not treat a successful request or lower latency as a quality
score.

The backend suite passed **247 tests**, with six optional tests skipped. After
deployment, a synthetic-microphone call through the public HTTPS app completed
both user and assistant captions, received audio packets and displayed 288 video
frames at 512×384, with no browser errors. This confirms the call path works; it
does not verify perceived naturalness, physical speaker output or perfect lip-sync.
