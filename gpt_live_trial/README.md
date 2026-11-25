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

### Private browser comparison, November 25, 2026

One Chromium call per voice stack used the same synthetic microphone recording, warm H100, Brick Room portrait, two-step renderer, 512×384 frames and 24 FPS cap. Times run from the end of caller speech to the first sustained audio detected in the browser's playback path.

| Caller turn | GPT-Live 1.1 + Marin | Original Sama + ElevenLabs |
| --- | ---: | ---: |
| Mix blue and yellow paint | 4.46 s | 3.92 s |
| Pepper's collar is teal | 5.94 s | 4.01 s |
| Recall the collar color | 4.61 s | 3.99 s |
| Median, three turns | **4.61 s** | **3.99 s** |

Both answered the paint question correctly. GPT-Live recalled “Teal.” Sama asked the caller to enable the webcam. The Sama recall time therefore measures the start of an incorrect answer. The browser received approximately 23.7 and 23.8 decoded frames per second respectively. Neither call had a decoded-frame gap above 250 ms. An earlier run of the same prompts measured 4.95 s and 4.40 s and produced the same recall outcomes.

In a separate overlap test, the caller interrupted an explanation of rain to ask about fog and later said “mm-hm.” GPT-Live switched to fog and continued through the acknowledgment. Sama also switched to fog, but treated “mm-hm” as a second interruption. From the start of the correction to the browser muting the previous answer, GPT-Live took 1.70 s and Sama took 0.66 s. Both missed the project’s 350 ms target. The early trial build failed when a pause ended between video frames. The corrected build kept publishing through the end of the replay. These short synthetic calls do not establish a general quality ranking.

OpenAI documents the session and continuous audio event format in its [GPT-Live guide](https://developers.openai.com/api/docs/guides/live) and [WebSocket audio guide](https://developers.openai.com/api/docs/guides/voice-websockets).
