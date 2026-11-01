# Chert realtime avatar demo

A browser video call with a generated avatar: speak, interrupt, share your camera,
and read live captions. AvatarForever runs on your GPU; OpenAI handles conversation
and vision, ElevenLabs generates speech, and LiveKit Cloud carries audio/video.

Ten reference portraits and stock-voice assignments are included. API keys, model
weights, recordings and private character assets are not included.

## Demo

The interface has a large avatar, an inset self-camera, fading captions and call
controls underneath. Swap the camera views or choose another avatar in Settings.
A recorded walkthrough will be added when it is ready.

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
