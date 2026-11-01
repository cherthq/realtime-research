# Recreate the demo

Run the backend on Ubuntu x86_64 with an H100 80 GB. Install Git, Python 3.11
with venv, Node.js 22.12+ and FFmpeg first. Allow at least 150 GB of free storage.
Keep the application and GPU model in separate Python environments.

## 1. Install and build

```bash
git clone https://github.com/cherthq/realtime-research.git
cd realtime-research
python3.11 -m venv .venv-realtime
.venv-realtime/bin/python -m pip install -r realtime/requirements.txt -c realtime/constraints-linux-py311.txt
cd realtime/web
npm ci
npm run build
cd ../..
```

The ten portraits are already in the repository. No private asset ZIP or voice
clone is needed for a normal call. Generated media and private imports stay ignored.

## 2. Install AvatarForever and download weights

```bash
bash infra/setup-avatarforever.sh
.venv-avatarforever/bin/hf auth login
.venv-avatarforever/bin/python scripts/download_baseline_weights.py avatarforever --root checkpoints/avatarforever
```

Before downloading, accept access to
[the Gemma encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
with the account belonging to your Hugging Face token. Enter the token privately
at the login prompt. The downloader uses pinned revisions and resumes existing files.

The configured weight root contains `AvatarForever/`,
`gemma-3-12b-it-qat-q4_0-unquantized/` and `avatarforever-revisions.json`.
Use a persistent filesystem if these files must survive instance termination.

## 3. Configure privately

```bash
cp -n realtime/.env.example realtime/.env
chmod 600 realtime/.env
nano realtime/.env
```

Fill `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`
and `LIVEKIT_API_SECRET`. LiveKit Cloud URLs start with `wss://`.
Use `TTS_PROVIDER=elevenlabs` and an ElevenLabs model available to your account;
the tested deployment uses `eleven_v4_turbo`. `ELEVENLABS_VOICE_ID` is optional
for the bundled demo: each portrait already has a stock voice.

Enable the prepared GPU:

```dotenv
AVATAR_CATALOG=demo
AVATARFOREVER_ENABLED=true
AVATARFOREVER_COMPILE_BLOCKS=true
AVATARFOREVER_CACHE_PROMPT=true
AVATARFOREVER_LISTENING_PROFILE=onset_boost
AVATAR_CONTINUOUS_MOTION=true
GPU_LOAD_TIMEOUT_SECONDS=900
```

The default model paths point at `.venv-avatarforever/bin/python` and
`checkpoints/avatarforever`. Override `AVATARFOREVER_PYTHON` and
`AVATARFOREVER_WEIGHTS` only when using another location.
Optional current-information lookup: `WEB_SEARCH_ENABLED=true`.

Never put API credentials in frontend variables, HTML, GitHub Pages or Git.

## 4. Start and prewarm

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_host prepare
AVATAR_RESOLUTION=512x384 .venv-realtime/bin/python -m realtime.scripts.demo_host serve --livekit-cloud
```

`prepare` runs once and creates a private server access profile; reuse it on
subsequent starts. The server stays in the foreground. In a second SSH terminal:

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_host warm --portrait call01 --timeout 900
```

Initial loading/compilation takes several minutes. Warm-up refuses to interrupt
an active call and releases its temporary session when ready. Native 640×480
is also supported: change `AVATAR_RESOLUTION` before starting the process.
Publication stays capped at 24 FPS.

## 5. Open from your laptop

```bash
ssh -N -L 8080:127.0.0.1:8080 ubuntu@YOUR_GPU_HOST
```

Open `http://127.0.0.1:8080`, using the private `access_code` from
`.cache/realtime-host/profile.json` when prompted. Allow microphone access,
choose an avatar in Settings, and start the call. The optional camera is shared
with the AI only during a call; turn it off to stop sharing.

For a public website without an access-code page, use the
[restricted HTTPS gateway](public-demo.md). Do not expose the private app port.

## Verify and stop

```bash
.venv-realtime/bin/python -m pytest realtime/tests -q
```

Try a short reply, interrupt it, switch the camera off/on and ask a current-info
question if web search is enabled. A passing unit test does not qualify model
quality; inspect the actual stream and [known limitations](../realtime/QUALIFICATION.md).

End the call before stopping/restarting the server. Ctrl+C stops the foreground
host. Stopping this script does not terminate or stop billing for a cloud GPU.

### Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing model / gated download | Hugging Face account access and configured weight root |
| Speech unavailable | Provider key, model access and credits; no silent provider fallback |
| Microphone unavailable | Browser permission and HTTPS or localhost |
| Busy message | One GPU permits one call; wait for the current call to end |
| GPU startup timeout | Prewarm first; inspect ignored `runs/realtime/gpu/` logs |
| Local Windows page stalls | Use the loopback helper in `realtime/scripts/windows_demo.py` |

Optional open-weight speech uses `infra/setup-qwen-tts.sh` and
`realtime.qwen_server`; sharing a GPU with avatar generation can increase delay.
For the simplest reproduction, use the documented ElevenLabs path first.
