# Interactive avatar demo

Microphone → OpenAI conversation/vision → ElevenLabs speech → persistent
AvatarForever worker → synchronized audio/video over LiveKit Cloud → browser.

**[Setup instructions](../docs/recreate-demo.md)** ·
**[Public HTTPS deployment](../docs/public-demo.md)** ·
**[Conversation and voice tuning](../docs/natural-conversation.md)** ·
**[Known limitations](QUALIFICATION.md)**

The client includes picture-in-picture camera, swappable views, device selection,
fading live captions, conversation history, interruption, and optional web lookup
with a spoken acknowledgement and source links. The ten bundled avatars use
stock voice assignments. Provider keys and custom voices remain server-side.

## Source layout

| Location | Responsibility |
| --- | --- |
| `app.py`, `catalog.py`, `voices.py` | Call lifecycle and public avatar/voice catalog |
| `text_voice_agent.py`, `elevenlabs_client.py` | Streamed dialogue and speech |
| `camera.py`, `transcripts.py`, `web_search.py`, `search_turn.py` | Vision, captions and lookup |
| `worker.py`, `continuous.py`, `playback.py` | Continuous generation, timing and cancellation |
| `gpu/`, `gpu_pool.py` | Separate persistent CUDA worker |
| `public_gateway.py` | Restricted HTTPS-facing interface to the private app |
| `web/` | React call interface |
| `scripts/`, `tests/` | Launchers, optional evaluation tools and regressions |

`AVATAR_CATALOG=demo` is the default and exposes only the ten bundled images
and ElevenLabs stock voices. `AVATAR_CATALOG=research` explicitly enables the
old frozen benchmark portraits and private local library; never use it for the
public gateway. Baseline evaluation and optional Qwen/OpenAI speech modules
remain available separately.

Build with `npm ci && npm run build` in `web/`. From the repository root,
run `.venv-realtime/bin/python -m pytest realtime/tests -q` for backend checks.
Browser fixtures live in `web/tests`; real GPU/provider checks are opt-in.

Generated files belong in ignored `data/`, `runs/`, `.cache/` and `web/dist/`.
