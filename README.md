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

> [!NOTE]
> The browser receives only session-scoped connection details. Keep provider keys
> in the backend’s private environment file.

## Demo

Watch a conversation with live captions, interruption and an animated avatar.

https://github.com/user-attachments/assets/953b779c-8f33-426b-8076-3ae8d0138d2b

**[Try the live demo →](https://dunlichan.com/projects/chert/)**

One person can call at a time. Join the waiting queue when the demo is busy;
your position updates automatically, with 30 seconds to start when it is your turn.

## Set up

Use an Ubuntu GPU host with an H100 80 GB, Python 3.11, Node.js 22.12+, FFmpeg and
at least 150 GB free disk space. You need LiveKit Cloud, OpenAI and ElevenLabs
credentials, plus Hugging Face access to the required Gemma text encoder.

1. Clone this repository and install the app dependencies.
2. Build the web client with `npm ci && npm run build` in `realtime/web`.
3. Run the GPU setup script and download the pinned AvatarForever/Gemma weights.
4. Copy `realtime/.env.example` to `realtime/.env` and fill the private credentials.
5. Start the demo host, prewarm the model, and connect through SSH or HTTPS.

**[Copy-and-paste setup instructions →](docs/recreate-demo.md)**

For a public domain, use the [HTTPS deployment guide](docs/public-demo.md).
Secrets stay on the backend; the browser receives only a short-lived room token
and a key scoped to its own call.

## What is included

| Directory | Purpose |
| --- | --- |
| [realtime/](realtime/README.md) | Web app, voice agent, streaming worker and tests |
| [realtime/demo-avatars/](realtime/demo-avatars/README.md) | Ten demo reference images and stock voices |
| [data_fetching/](data_fetching/README.md) | Review-first 4:3 talking-video preparation pipeline |
| [configs/](configs/models.json) and [scripts/](scripts/workspace.py) | Pinned model sources and baseline tooling |
| [infra/](infra/lambda.md) | GPU setup and deployment helpers |

The existing [model baseline workflow](docs/running-baselines.md) and
[zero-shot protocol](docs/zero-shot-baseline.md) remain separate from the live demo.
No dataset collection, download or training starts automatically.

## Current limits

This is an experimental **one-call-per-GPU** demo with a ten-minute call limit,
not a production capacity claim. Long calls can drift visually; idle mouth
motion and lip-sync are not perfect. See [tested scope and limitations](realtime/QUALIFICATION.md).
The stream uses 4:3 geometry and a 24 FPS publication cap. Lower resolution can
reduce compute but is not a cure for drift.

Upstream models and reference images retain their own terms. The bundled stock
voices are creative casting choices, not clones of the pictured people.
