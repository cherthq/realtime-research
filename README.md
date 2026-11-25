<p align="center">
  <a href="https://www.trychert.com/"><img src="docs/assets/chert-logo.png" alt="Chert" width="80" height="80"></a>
</p>

<h1 align="center">Chert Realtime Avatars</h1>

<p align="center"><strong>Realtime video diffusion avatar model infra demo</strong></p>

<p align="center">
  <a href="https://dunlichan.com/projects/chert/"><strong>Try the browser demo ↗</strong></a> ·
  <a href="facetime/README.md">Connect FaceTime</a> ·
  <a href="#quickstart">Self-host</a> ·
  <a href="docs/avatar-experiment-log.md">Experiment log</a>
</p>

## Demo

https://github.com/user-attachments/assets/953b779c-8f33-426b-8076-3ae8d0138d2b

Choose one of ten portraits, turn on your microphone and start talking. Try interrupting mid-answer, turning on live captions, or showing the avatar something through your webcam.

The public browser demo serves **one caller at a time**. If someone is on a call, join the waiting queue. Your position updates automatically, and you have **30 seconds** to start when your turn arrives.

## How a call works

The voice backend produces the reply audio. AvatarForever uses that same audio to generate the avatar’s mouth and facial motion. LiveKit carries the caller’s microphone in and the synchronized voice and video back out.

| Call path | What runs | Setup |
| --- | --- | --- |
| **Browser** | The documented setup uses OpenAI for conversation and camera questions, and ElevenLabs for the portrait’s stock voice. | [Browser setup](docs/recreate-demo.md) |
| **GPT-Live browser preview** | GPT-Live 1.1 listens and speaks directly in the Marin voice. Chert can retrieve project docs, search the web, or inspect a shared camera frame when enabled. | [Voice preview](gpt_live_trial/README.md) |
| **FaceTime** | An inbound call to your Chert line connects to the GPT-Live preview with the **Brick Room** avatar. The connector reserves the GPU when the caller joins. Caller video depends on the line’s camera capability. | [FaceTime connector](facetime/README.md) |

## Quickstart

The tested host is an **Ubuntu x86_64 machine with one H100 80 GB** and at least **150 GB of free disk space**. Install Git, Python 3.11 with venv, FFmpeg, and Node.js 22.12+ for the web client. Use Node.js 24 for the FaceTime connector.

1. **Install the app and build the web client.** On the GPU host, run the following commands.

   ```bash
   git clone https://github.com/cherthq/realtime-research.git
   cd realtime-research
   python3.11 -m venv .venv-realtime
   .venv-realtime/bin/python -m pip install -r realtime/requirements.txt -c realtime/constraints-linux-py311.txt
   (cd realtime/web && npm ci && npm run build)
   ```

2. **Prepare the renderer.** Install AvatarForever in its own environment and download the pinned weights. You’ll need Hugging Face access to the Gemma text encoder. [GPU setup and weight download](docs/recreate-demo.md#2-install-avatarforever-and-download-weights)
3. **Configure the backend.** Copy `realtime/.env.example` to `realtime/.env`. Set the LiveKit Cloud and OpenAI credentials there. The stock-voice browser setup also needs an ElevenLabs key. The GPT-Live path needs OpenAI project access to `gpt-live-1.1`. [Private configuration](docs/recreate-demo.md#3-configure-privately)
4. **Start the host and prewarm the portrait.** Open the browser app through the SSH tunnel. For an iPhone call, start the GPT-Live preview and connect the named agent to a FaceTime line in your Chert project. [Host and warm-up commands](docs/recreate-demo.md#4-start-and-prewarm) · [FaceTime setup](facetime/README.md)

For public hosting, the HTTPS gateway keeps provider credentials on the server and gives each browser a short-lived room token and a key scoped to its call. [HTTPS deployment guide](docs/public-demo.md)

## Known limitations

The current deployment supports **one active call per GPU**. The public browser demo has a **ten-minute call limit**. A FaceTime call and a browser call on the same renderer share that slot.

Video uses 4:3 framing with a **24 FPS publication cap**. Frames can repeat under load. On longer calls, color or identity can drift, and the mouth can move during silence or miss the first sounds of a reply.

Two inbound FaceTime calls reached the agent and Chert reported outbound audio and video on both. Handset latency, perceived lip-sync, and incoming camera quality still need a phone-side check. [Tested scope](realtime/QUALIFICATION.md) · [Experiment results](docs/avatar-experiment-log.md)

## Explore the repository

| For | Start here |
| --- | --- |
| Call handling, web client and GPU worker | [Realtime app](realtime/README.md) |
| Portraits and stock voice assignments | [Avatar gallery](realtime/demo-avatars/README.md) |
| Preparing and reviewing 4:3 talking videos | [Data pipeline](data_fetching/README.md) |
| Reproducing model comparisons | [Pinned model sources](configs/models.json) · [Baseline workflow](docs/running-baselines.md) · [Zero-shot protocol](docs/zero-shot-baseline.md) |
| Provisioning the GPU host | [Infrastructure guide](infra/lambda.md) |

Upstream models and reference portraits retain their own terms. The bundled gallery pairs each portrait with a stock voice selected for the demo.
