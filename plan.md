# Realtime demo roadmap

## Available

- AvatarForever streaming on a persistent H100 worker, BF16 and four denoising steps.
- 4:3 video, optional 512×384 or 640×480 generation, 24 FPS publication cap.
- LiveKit Cloud audio/video, ElevenLabs speech and OpenAI dialogue/vision.
- Continuous listening motion, interruption, live captions and web-search acknowledgement.
- Ten bundled public reference portraits with stock voices.
- Server-side API keys, restricted public gateway and one active GPU call.

## Deployment checks

- Build and test from a clean checkout using the public setup guide.
- Verify only ten portraits and their stock voices are publicly reachable.
- Check the HTTPS domain, camera permissions, a complete call and interruption.
- Keep generated media, credentials, private voices and research outputs ignored.

## Research still needed

- Long-call identity, clothing, background and color drift remains unresolved.
- Qualify listening-mouth control without suppressing the beginning of speech.
- Measure first-response latency and playback stability across all portraits.
- Compare teachers and distillation only against fixed quality/latency baselines.

Keep the ten-minute call limit. Do not hide drift with periodic resets or still
images, change precision without qualification, or describe throughput as proof
of stable visual quality. Experimental drift investigations are preserved locally
and are not part of the deployed runtime.
