# GPT-Live 1.1 voice comparison

The control and trial are two private websites backed by one avatar renderer. Open the control on port 18821 or the GPT-Live trial on port 18837. End the current call before switching. Both sites share the same admission lock, portrait catalog, session manager and GPU pool.

The trial streams microphone audio to GPT-Live over a server-side WebSocket. The returned 24 kHz audio drives both AvatarForever and the caller's audio track. The browser hears the result through LiveKit. A response uses the Marin voice and can hear the caller during playback. The trial exercises conversation and in-call recall. Camera analysis, web lookup and durable memory are outside this comparison.

## Run on a prepared Chert host

Follow the [demo setup](../docs/recreate-demo.md) to install the dependencies, build the client and configure the renderer and LiveKit. Supply an OpenAI project key with access to `gpt-live-1.1` in the backend environment. The existing baseline uses its configured speech credentials.

```sh
.venv-realtime/bin/python -m gpt_live_trial.preview
```

The default factory is `realtime.app:create_app`. Set `--base-factory package.module:factory` to use a prepared private preview with its own avatar checkpoint and defaults. The process opens two loopback listeners. Start it on a free GPU or through the host's existing worker handoff so only one process owns the renderer.

To access the two websites from a laptop, forward both loopback ports through the authorized SSH connection. Choose free local ports if either is already in use.

```sh
ssh -N -L 18821:127.0.0.1:18821 -L 18837:127.0.0.1:18837 ubuntu@GPU_HOST
```

The trial exposes its identity at `/api/preview/provider`. A live call reports `gpt-live-1.1` in the session's `voice_model`. Each started trial uses billable OpenAI Live session time. Ending the call closes the upstream session.

## What to compare

Use the same portrait, microphone and prompts on both sites. Try an ordinary question, a factual correction while the avatar is still speaking, a brief “mm-hm,” and a question about something you mentioned earlier. Time the first audible answer at the browser and the end of audible speech after a correction. Capture what the caption says and which words actually play. The GPU can buffer synchronized audio and video after the voice model has already reacted.

Run the local regression checks with `python -m pytest -q gpt_live_trial`. They cover microphone transcript grouping, silence at the speech boundary, brief pauses, explicit interruptions, and single-call admission across both sites.

OpenAI documents the session and continuous audio event format in its [GPT-Live guide](https://developers.openai.com/api/docs/guides/live) and [WebSocket audio guide](https://developers.openai.com/api/docs/guides/voice-websockets).
