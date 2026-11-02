<p align="center">
  <a href="https://www.trychert.com/"><img src="docs/assets/chert-logo.png" alt="Chert" width="88" height="88"></a>
</p>

<h1 align="center">Chert Realtime Avatars</h1>

<p align="center"><strong>A self-hosted platform for real-time conversations with AI avatars.</strong></p>

<p align="center">
  <a href="https://dunlichan.com/projects/chert/"><img src="https://img.shields.io/badge/Demo-Try%20it-28724f" alt="Try the live demo"></a>
  <a href="https://www.trychert.com/"><img src="https://img.shields.io/badge/Chert-Website-222222" alt="Chert website"></a>
  <a href="realtime/QUALIFICATION.md"><img src="https://img.shields.io/badge/status-experimental-e69838" alt="Experimental"></a>
</p>

A browser video call with a generated avatar: speak, interrupt, share your camera,
and read live captions. AvatarForever runs on your GPU; OpenAI handles conversation
and vision, ElevenLabs generates speech, and LiveKit Cloud carries audio/video.

Ten reference portraits and stock-voice assignments are included. Model weights,
private character assets and API keys stay on your server.

## Demo

Watch a conversation with live captions, interruption and an animated avatar.

https://github.com/user-attachments/assets/953b779c-8f33-426b-8076-3ae8d0138d2b

**[Try the live demo →](https://dunlichan.com/projects/chert/)**

The public demo supports **one caller at a time**. If it is busy, join the queue;
your position updates automatically. You have 30 seconds to start when your turn arrives.

## Quickstart

### Requirements

| Component | Required |
| --- | --- |
| GPU host | Ubuntu x86_64, H100 80 GB, at least 150 GB free storage |
| Software | Git, Python 3.11 with venv, Node.js 22.12+, FFmpeg |
| Services | LiveKit Cloud, OpenAI and ElevenLabs credentials |
| Model access | Hugging Face access to the [Gemma text encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

### Install

Run on your GPU host:

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

### Configure and run

1. **Prepare the model.** Run the [GPU setup script](infra/setup-avatarforever.sh)
   and download the pinned AvatarForever and Gemma weights.
2. **Add credentials.** Copy [realtime/.env.example](realtime/.env.example) to
   `realtime/.env` and configure the providers and GPU worker. Keep this file private.
3. **Start the application.** Prepare the host, start the server and prewarm the
   model, then open the web app through an SSH tunnel or HTTPS.

The setup guide includes the commands and configuration for each step:

**[Complete setup →](docs/recreate-demo.md)** · **[Deploy on a public domain →](docs/public-demo.md)**

Provider keys stay on the server. The browser receives a short-lived room token
and a key scoped to its own call.

## Repository guide

| Directory | Contents |
| --- | --- |
| [realtime/](realtime/README.md) | Web app, voice agent, streaming worker and tests |
| [realtime/demo-avatars/](realtime/demo-avatars/README.md) | Ten reference portraits and stock voice assignments |
| [data_fetching/](data_fetching/README.md) | 4:3 talking-video preparation and review pipeline |
| [configs/](configs/models.json) · [scripts/](scripts/workspace.py) | Pinned model sources and baseline tools |
| [infra/](infra/lambda.md) | GPU setup and deployment helpers |

For model comparisons, follow the [baseline workflow](docs/running-baselines.md)
and [zero-shot evaluation protocol](docs/zero-shot-baseline.md). Dataset preparation,
evaluation and training are separate from running a call.

## Known limitations

- **Capacity:** the current deployment serves one call per GPU, with a ten-minute call limit.
- **Visual quality:** long calls can drift; idle mouth motion and lip-sync remain imperfect.
- **Playback:** output uses 4:3 framing with a 24 FPS publication cap. Actual performance
  depends on the GPU and configuration; lower resolution does not resolve visual drift.

See [validation results and tested scope](realtime/QUALIFICATION.md) before deploying.

Upstream models and reference images retain their own terms. Bundled voices are
stock voice selections, not clones of the people in the portraits.
