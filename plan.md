# Realtime avatar demo platform plan

Updated: 2026-10-28.

**Active direction: test AvatarForever first on Lambda, then expose a persistent
avatar worker to a browser application over WebRTC.** The model adapter remains
replaceable so other checkpoints and architectures can be evaluated later.

This document describes the target platform and the implementation status below.
The local diagnostic application is implemented. Native AvatarForever chunks now
run on a Lambda H100 80GB, and real voice/avatar browser calls pass for all three
characters. The 18-output pilot and 30 warm decoded-chunk turns completed. An
optional compiled-block path now passes the ten-minute paced worker replay at
38.62 generated FPS with a 1.28s peak input queue, using unchanged weights, BF16
precision and 640x480 output. The updated browser path completed 31 spoken turns;
30 warm turns measured 2.653s median and 3.256s p95, missing the 3s p95 target.
Physical-device qualification remains open;
see [the measured qualification record](realtime/QUALIFICATION.md).
Transient browser control loss now pauses playback and retries the same session
for up to 30 seconds. Seven recovery/startup checks pass on the remote SSH route,
including expired access and fallback from a failed event stream to polling.
Idle publication now keeps both audio and video tracks continuous using silence
and held images; these are excluded from generated-frame and response coverage.
The first compiled live latency run failed after 17 turns; its delivery failure
is retained alongside the completed retest and successful ten-minute GPU replay.
Final compiled calls passed for all three characters, including a temporary
control outage recovering the same call and voice, interruption and cleanup.
Updating this document does not launch inference, download checkpoints, or
restart the stopped 108-output baseline batch.

## 1. Goal, current state, and defaults

Build an interruptible, voice-driven, FaceTime-style browser demo:

**Browser microphone -> LiveKit/WebRTC -> voice agent -> response PCM audio ->
AvatarForever on Lambda GPU -> synchronized 640x480 audio/video ->
LiveKit/WebRTC -> browser.**

The browser handles microphone capture, controls, and playback. Lambda handles
the application backend, conversation orchestration, avatar inference, decoding,
and media publication. Initially the voice agent calls OpenAI Realtime, so speech
and conversation inference run through that provider; they are not self-hosted
on Lambda. Self-hosted LiveKit transports the media; no LiveKit Cloud account is
required. Run it on localhost during development and on our infrastructure for
the remote demo. Self-hosting voice models is a
separate later decision.

### Call interface (updated 2026-10-28)

The primary view is two simple 4:3 video tiles: the user's camera on the left and
the avatar on the right. Place camera, microphone, start/end, and interrupt
controls underneath. Use a plain neutral background; no dashboard, promotional
headline, permanent sidebars, or visible metric cards. On phones, stack the tiles.

Keep avatar/model/voice selection, scripted replies, and connection details in
one Settings dialog. Camera preview is opt-in and stays in the browser: it is
not uploaded, published to LiveKit, recorded, or sent to the voice agent. Release
the camera on camera-off, call end, and page exit, including late permission
responses. Explicitly label the diagnostic renderer as a preview.

### Existing foundation

- Three synthetic 4:3 upper-body portraits and six fixed Kokoro `af_heart` WAVs
  are prepared under ignored `data/eval/facetime-zero-shot-v1/`.
- Scenario definitions, silence ranges, provenance, and hashes are in the
  [scenario pack](configs/zero_shot_suite.json) and
  [baseline protocol](docs/zero-shot-baseline.md).
- Native file-input baseline runners exist for FlashTalk, LiveAvatar, and
  LiveAct. FlashTalk can reuse a loaded pipeline; these wrappers do not establish
  a live audio-input or browser-streaming interface.
- A React diagnostic web application, FastAPI session API, bounded PCM buffer,
  replaceable diagnostic renderer, and LiveKit media publisher are implemented
  under `realtime/`. The frontend production build and 75 backend tests pass,
  including a real LiveKit frame/audio/cancellation test in Ubuntu through WSL.
- A checksum-verified self-hosted LiveKit installer is included. No LiveKit
  Cloud account is needed. A Windows localhost bridge now works around this
  laptop's stalled loopback HTTP responses; services run in WSL and WebRTC uses
  its private IP. `realtime/start-demo.cmd` serves the app at port 18800 without
  WSLg windows or browser flags. Five Windows Chrome call/UI checks and one
  audio-graph timing check pass. Seven Linux browser checks have passed against the private TCP profile,
  including the opt-in audio-graph timing test: access-code login/logout, two-panel 4:3 playback, local camera cleanup,
  denied/late camera permission, microphone publication, interruption, a second
  isolated session, and mobile layout. Desktop/mobile
  screenshots were visually reviewed. The Windows route uses bounded HTTP
  polling for control; normal hosts retain SSE. This is a local development
  workaround, not a change to the Lambda deployment architecture.
- The diagnostic renderer is explicitly labeled and does not generate avatar
  motion. The OpenAI Realtime bridge, voice-preview review, image-based casting,
  cache/override validation, and frozen v2 audio preparation are implemented and
  tested with injected provider responses. Real OpenAI preparation now succeeded:
  three voice previews, image-based assignments for all three fictional portraits,
  and a frozen v2 pack with 12 distinct WAVs covering 18 image/audio cases.
  Two portraits share a selected voice; v1 inputs remain unchanged.
- AvatarForever is registered in `configs/models.json`; pinned source is staged.
  Dedicated setup, weight-download support, and a native direct-4:3 file smoke
  runner are implemented. An experimental bounded streaming adapter, resident subprocess, cancellation,
  and media publisher are implemented. A native H100 smoke generated 89 frames
  at direct 640x480. Three chunks took 0.986/1.108/1.299 seconds for
  1.00/1.28/1.28 seconds of media; cold loading took 157 seconds. This short sample
  does not establish sustained realtime performance or lip-sync quality. The UI keeps
  AvatarForever disabled unless explicitly enabled on a prepared GPU host.
- A private-host supervisor and three-port SSH tunnel command are implemented.
  The profile generates private LiveKit/access credentials; browser login uses
  revocable HttpOnly cookies. Local browser stats confirm actual ICE/TCP media.
  The actual Windows-to-Lambda route passed the two-panel diagnostic call test
  over SSH-forwarded TCP, including audio, interruption and cleanup. Cloud
  firewall configuration has not been independently audited.
- The model pool retains weights across calls and clears character state; two-call
  isolation and cancellation pass with a synthetic GPU subprocess over real
  LiveKit. Paced/throughput replay, 18-case selection, warm-turn repetition and
  continuous replay commands are implemented. Native review-video frame coverage
  is tested with FFmpeg, including a partial final audio chunk. GPU allocation and
  host-memory telemetry are wired. Native smoke loading peaked at 73.6 GB of
  PyTorch GPU allocations and retained about 39.6 GB after text encoder offload.
  Driver usage and process peaks are separate measurements.
- Frame IDs now accompany real video frames. Browser render callbacks send
  session-scoped playback receipts; stale/unknown frames cannot advance the
  history offset. Local clicks and event-driven server cancellation mute output,
  and only a newer response's displayed frame can reopen it. Repeated-response
  and delayed-HTTP cancellation tests pass. This confirms browser presentation
  and mute state, not physical audio output. Missing feedback retains zero audio
  in truncated history. The previous Lambda SSH address timed out on 2026-10-28;
  that is not proof the instance has stopped or is no longer billable.
- Playback uses a Web Audio gain gate; autoplay recovery cannot bypass an
  interruption. A pinned-SDK compatibility path connects Chromium audio receivers
  when video metadata enables legacy encoded streams. Real audio decoding is
  now verified by signal samples, not merely by a playing element. Both tracks
  share a stream name; an 80ms video jitter-buffer hint reduces video leading
  audio, with remote/device qualification still required.
- Frame-aligned flash/beep and sustained-tone interruption tests are implemented.
  The latest local private-TCP sample had ten markers with 34.8ms maximum/p95
  absolute graph/video offset; local/server audio-graph stops were 32ms/18.2ms.
  These are small-sample foreground browser measurements, not physical-speaker
  or model lip-sync results. Reports retain the measurement scope and environment.
- The first live call now performs a missing synthetic-character voice match
  before reserving the worker, records preparation time separately, and reuses
  the result. Overrides skip casting; changed inputs require explicit recasting.
  Injected-provider tests verify the route and cache behavior without real API calls.
- Session and worker teardown each use a shared shielded task. A cancelled HTTP
  request cannot abandon cleanup or release the single-session slot early.
  Cancelled-request and media-queue-failure tests verify all release operations;
  repeated end requests wait for the same teardown. The six UI lifecycle tests
  were rerun against the private TCP host after these changes.

### Remaining completion gates

The private interactive demo works. Broader qualification remains incomplete:

- Physical-device microphone/speaker verification is deferred at the user's
  request. Acoustic stop/flash-beep measurements and background-tab/mobile
  behavior remain unverified. Browser audio-graph measurements do not satisfy
  physical playback timing gates.
- The remote synthetic transport test measured 49.9ms p95 graph/video offset,
  but its 489.3ms HTTP-request-to-stop observation exceeded the 300ms target.
  It includes network transit and does not measure speech detection or speakers.
- The 18 model outputs and 30 warm decoded-chunk turns are complete. The original
  eager ten-minute test failed the bounded input queue; the compiled worker later
  completed all 15,000 frames with a 1.28s peak pending queue and bounded memory.
  This qualifies decoded-chunk throughput, not a ten-minute browser conversation.
- The first compiled live latency attempt failed after 17 successful turns.
  After the idle continuity change, all 31 turns completed with no reported
  renderer stalls or missing audio/video onset. Warm p95 was 3.256s, exceeding
  the 3s target. Preserve both reports; this sample is not a reliability SLA.
- Remote recovery/startup tests pass. Public HTTPS/WSS/TURN remains undeployed
  and is needed only for a later publicly shared service. The current demo uses
  private Windows-to-Lambda SSH/TCP transport.
- Perceptual lip-sync and equivalence of incremental audio/VAE boundaries to the
  upstream full-file renderer remain unscored.

### Demo defaults

| Setting | Decision |
| --- | --- |
| First real model | AvatarForever; FlashTalk is an optional comparison |
| Capacity | One active session and one loaded avatar model |
| Compute | Existing Lambda H100-class host, after checking its current hardware and availability |
| Generated geometry | Direct 640x480, landscape 4:3, preserving head, shoulders, and torso |
| Frame rate | Retain native timing; target approximately 25 unique generated FPS |
| Inputs | Existing three portraits; fixed audio for replay; microphone for live mode |
| Voice | Image-based casting for synthetic characters; save the selected voice and use it with `gpt-realtime` |
| Transport | Self-hosted LiveKit; localhost first, Lambda deployment later; configurable endpoint |
| Controls | Select portrait/model, preview or override its voice before calling; start, mute, interrupt by speaking, end |
| Duration | Ten-minute call limit; thirty-second reconnect grace period |
| Recording | Off by default; explicit debug recording only |

No stretching, letterboxing, post-generation resize, or interpolation qualifies
as direct 4:3 generation. Enforce the existing
[geometry configuration](configs/baseline_geometry.json). If AvatarForever
cannot generate this shape, record the limitation rather than silently changing
the requirement.

The first demo does not include Apple FaceTime integration, user-camera
understanding, multi-user GPU scheduling, training, distillation, or generated
listening reactions. A listening state can hold the portrait or last valid
frame; this must not be counted as newly generated motion.

### Automatic voice casting from the character image

**Reference image -> vision-based character casting -> matching available voice
-> optional preview -> saved voice profile -> voice agent -> AvatarForever.**

The default for fictional/synthetic avatars is automatic image-based casting,
not a fixed mapping from portrait ID or a manual male/female selector. Choose a
voice that fits the character's visual design and any authored character
description; treat this as a creative casting choice rather than discovery of
the person's actual voice.

Our three existing references are documented fictional characters in
[portrait provenance](configs/zero_shot_portraits.json). Pass their actual images
to the casting step. Their authored descriptions can constrain the casting, but
do not preassign `ref01`/`ref02`/`ref03` to `cedar` or `marin`.

Implementation:

1. Build a versioned catalog of available provider voices with cached audio
   previews and reviewed descriptions of vocal presentation, pitch range,
   texture, and delivery. Include masculine, feminine, and neutral options.
   Do not guess those descriptions from the provider's voice names.
2. Run a configurable image-capable casting model once during avatar preparation.
   Start with `gpt-5.6-terra` through the Responses API, and record its resolved
   model revision. Send the reference image, verified fictional-character
   context, optional authored constraints, and the available voice descriptions.
   This is an external API step orchestrated by Lambda, separate from GPU avatar
   inference and live response generation.
3. Ask for a creative casting recommendation from that catalog. Return structured
   output with `status`, `voice_profile_id`, and a short casting description.
   Restrict the profile ID to catalog entries; do not return a gender classifier,
   invented provider voice, or claims about a real person's actual voice.
4. Validate the response and persist the selected profile. Cache by image hash,
   character context/preferences, casting model/prompt revision, and catalog
   revision. Repeated calls reuse the stored result instead of rerunning casting
   or randomly changing the voice.
5. Show **Auto-match voice** as the default. Matching completes before the call;
   preview and manual override remain optional. A user can request recasting or
   select a specific voice, and an explicit override takes precedence.
6. Apply the resolved provider voice before the first audio response and keep it
   fixed across turns, interruption, and reconnect. A voice change after speaking
   requires restarting the voice session in v1.

Use the image itself as input, not just its filename or original generation
prompt. Any example such as a gentle or assertive delivery is a proposed creative
performance, not a claim that facial appearance determines personality.

For real-person photos or unknown provenance, do not infer gender or sensitive
traits from appearance. Use a supplied voice preference or character brief and
save the choice; return `needs_preference` when it is missing. Do not decide that
an unknown photo is fictional by looking at it. Accent and language come from
explicit preferences, not ethnicity inferred from an image.

Handle refusals, timeouts, invalid catalog IDs, and unavailable voices as visible
preparation errors with retry/manual selection. Do not silently replace a failed
match with a random global voice. Version changes require an explicit recast
rather than silently changing existing avatars.

The first demo selects the best fit from an existing voice library. It does not
synthesize a unique new voice identity or clone a real person's voice from a
photo. AvatarForever animates the selected audio; it does not choose the voice.

OpenAI supports image inputs and structured outputs for this casting step:
[image input guide](https://developers.openai.com/api/docs/guides/images-vision),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[initial casting model](https://developers.openai.com/api/docs/models/gpt-5.6-terra).
Realtime voice selection must occur before audio output starts; see the
[voice options](https://developers.openai.com/api/docs/guides/realtime-conversations#voice-options).

### Fixed benchmark audio versus avatar-matched demo audio

The existing v1 assets deliberately use the same Kokoro `af_heart` voice across
all portraits for controlled comparison. That explains the mismatched voice in
earlier clips; it was not random per-avatar voice selection.

Preserve those masters and their hashes as the fixed-audio reference track. For
the first visual demo and AvatarForever testing, prepare a separate frozen
`facetime-avatar-voices-v2` pack after casting each of the three portraits: the
same six scripts, one saved waveform per distinct resolved profile/scenario, and
explicit inserted PCM silence. Generate each waveform once, then reuse it for
every model. If two avatars resolve to the same complete voice profile, share
the saved scenario audio. Freeze the casting results before generating the pack.

Record image hash, casting decision/model/prompt/catalog revisions, resolved
voice/provider/model/profile revision, transcript, duration, silence sample
ranges, and WAV hashes. Store the new pack under ignored `data/eval/`; do not
overwrite the v1 manifest or reuse its run IDs. Label v1 and v2 results separately:
different voices/waveforms are not directly identical benchmark inputs. Recompute
sample ranges and final-chunk handling from the new audio durations.

## 2. AvatarForever test choice and reproducibility

AvatarForever is the first model to install, smoke-test, and adapt. Do not require
a FlashTalk integration or the old three-model matrix to finish first.

The authors report 27.2 FPS at 768x512 on one H100 for a 22B backbone. That result
is not a measurement of our GPU, 640x480 geometry, incremental audio input, or
complete networked call. Released weights alone do not establish those properties.
Sources: [project](https://github.com/leeruibin/avatarforever),
[paper](https://arxiv.org/abs/2608.12107).

Verified release references on 2026-10-28:

| Asset | Reference |
| --- | --- |
| Inference repository | [leeruibin/avatarforever](https://github.com/leeruibin/avatarforever) |
| Code revision | `de4deb6a55c9567a17dad58d67390fe089993630` |
| Checkpoint repository | [LetsThink/AvatarForever](https://huggingface.co/LetsThink/AvatarForever) |
| Weight revision | `622ea82482c09e7fb2d8e29cb7ff1d009c820f05` |
| Weight file | `avatarforever-ltx-2.3-22b.safetensors` |
| Auxiliary text encoder | `google/gemma-3-12b-it-qat-q4_0-unquantized` at `68f7ee4fbd59087436ada77ed2d62f373fdd4482`; gated access |
| Environment | Dedicated Python 3.11 environment following the pinned upstream workspace |

The inspected CLI supports a reference image, explicit width/height, audio-file
input, and frame rate. It is not itself an incremental-audio service. The adapter
must establish how to feed arrived audio and obtain decoded chunks without
waiting for a complete MP4.
See [pinned inference entrypoint](https://github.com/leeruibin/avatarforever/blob/de4deb6a55c9567a17dad58d67390fe089993630/inference.py).

Before inference:

- Add an `avatarforever` registry entry and dedicated setup/download support,
  preserving existing model pins and environments.
- Record separate code, checkpoint, backbone, and auxiliary-model license terms.
  Use the official gated text-encoder access process where required.
- Inspect actual GPU type, VRAM, host RAM, available disk, CUDA, PyTorch, and
  attention backend. An 80GB GPU is a test target, not a proven fit.
- Record preprocessing, precision, attention implementation, history/cache
  settings, image conditioning, prompt, seed, and geometry.
- Keep vendor sources pinned. Store any compatibility change as a reviewable
  adapter or patch with its hash, rather than silently editing dependencies.
- Cache invariant image/text conditioning where supported. If text-encoder
  offloading is needed, label and measure it. Quantization or other performance
  changes get separate named profiles.

Model roles after the first integration:

| Model | Role |
| --- | --- |
| AvatarForever | First GPU test and first real streaming adapter |
| Deterministic mock | CPU-only transport, timing, cancellation, and UI tests |
| FlashTalk | Existing quality reference; optional second adapter |
| LiveAvatar / LiveAct | Later interchangeable adapters and retained offline baselines |
| HunyuanVideo-Avatar | Optional offline reference under its applicable license |

A different checkpoint of the same supported architecture should be configurable.
A new architecture may require a new adapter. Model switching happens between
sessions; it is not an instantaneous in-call swap.

## 3. Lambda and web application architecture

| Component | Technology and location | Responsibility |
| --- | --- | --- |
| Browser UI | React, TypeScript, Vite, LiveKit client SDK | Microphone, avatar playback, call state, model selector, diagnostics |
| Control service | FastAPI on Lambda CPU | Voice-casting orchestration, exclusive GPU lease, preparation, room credentials, lifecycle, health |
| Voice agent | OpenAI Realtime SDK bridge on Lambda CPU | Voice-provider session, turn detection, response audio, interruption |
| Avatar worker | Persistent process in the selected model environment on Lambda | Conditioning, GPU inference, decoded frames, synchronized publication |
| Media service | Self-hosted LiveKit, locally or on Lambda CPU | WebRTC rooms, connectivity, audio/video delivery |
| Storage | Lambda persistent filesystem | Pinned assets, checkpoints, inputs, run metadata, optional recordings |

Use the custom avatar-worker media pattern: the voice agent forwards response
audio to the worker, and the worker publishes matching audio and video. The
implementation uses the OpenAI Realtime SDK directly so queue invalidation and
history truncation share the avatar publisher's playback position. This replaces
the planned LiveKit Agents wrapper without changing the media route. Current
truncation uses displayed-frame receipts when audio is enabled, bounded by server
publication progress, and zero when confirmation is missing. This is a conservative
audio proxy; audible-stop qualification remains required. Pin and test SDK versions and feedback.
[Custom avatar documentation](https://docs.livekit.io/agents/models/avatar/).

Keep model execution separate from control/network handling so a long GPU
operation does not block cancellation. Do not transfer raw frames through the
FastAPI server or create a subprocess for each spoken sentence.

For initial team use, run `python -m realtime.scripts.demo_host prepare`, then
`serve`, and connect with the helper's `tunnel` command. Forward all three ports:
8080 UI/control, 7880 signaling, and 7881 ICE/TCP media. The private profile binds
HTTP to loopback, advertises loopback candidates and forces TCP. LiveKit's media
listener still binds all interfaces: keep service ports blocked to public ingress
and allow SSH only. See the [deployment commands](realtime/README.md#private-lambda-demo-over-ssh).
Use `python realtime/scripts/livekit_local.py run` only for local development.
Its pinned binary is checksum-verified and stored in ignored
`.cache/livekit/`; local development credentials are not production credentials.
Public sharing later requires HTTPS/WSS, private signing credentials, team access
control, firewall configuration, and a verified TURN fallback. Configure the
server's advertised address for the remote host; do not deploy the localhost
development command to a public interface. LiveKit Cloud remains optional.
[Local LiveKit guide](https://docs.livekit.io/transport/self-hosting/local/).
[Lambda connection and tunneling guide](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/).

### Session lifecycle and API

1. List models, their capabilities, and qualification state.
2. Resolve an explicit voice override or cached image-based casting result. If
   missing, complete eligible character casting before reserving the GPU slot;
   report casting time separately from live response latency.
3. Load weights if necessary, warm up, and verify readiness.
4. Create the room, dispatch the voice agent, and join the avatar worker.
5. Issue short-lived browser credentials scoped to that room.
6. Accept microphone audio and stream synchronized responses.
7. On end, error, or expired reconnect grace, cancel generation and clear all
   conversation/identity state; release the session slot.
8. Retain model weights for the next call, or unload before changing models.

| Interface | Behavior |
| --- | --- |
| `GET /api/models` | Capabilities, availability, loading state, qualification status |
| `GET /api/avatars` | Available portraits, casting status, and saved voice profile IDs |
| `POST /api/avatars/{id}/cast-voice` | Cast an eligible character image, reuse its valid cached result, or explicitly recast |
| `GET /api/voices` | Configured voice profiles, presentation labels, and cached preview references |
| `POST /api/sessions` | Model/portrait selection and optional voice-profile override; resolve voice, reserve GPU, prepare |
| `GET /api/sessions/{id}` | Preparation state and room connection details when ready |
| `DELETE /api/sessions/{id}` | Idempotent cancellation and cleanup |
| Session event stream | State changes, errors, and diagnostic measurements |

Return busy for a second caller; no waiting queue in v1. Authenticate control
requests, bind sessions to their callers, and keep worker control private.
OpenAI, Hugging Face, and LiveKit secrets stay on the server. Browser tokens permit
only the required room subscription and microphone publication.

No database, Kubernetes, or autoscaling is needed for the single-host demo.
Stopping a call releases its session; it does not terminate the Lambda instance.

## 4. Streaming adapter and playback behavior

Implement one shared lifecycle:

`describe -> load -> warmup -> open_session -> push_audio / end_segment ->
emit_media -> cancel -> close_session -> unload`.

Capabilities describe supported geometry/FPS, input sample rate, minimum chunk,
lookahead, incremental-input support, image/prompt conditioning, and reset behavior.
Expose qualification as `offline_only`, `streaming_experimental`, or
`realtime_qualified`; a file-input smoke test cannot set the last two states.

Each audio/media event carries session ID, response ID, cancellation generation,
sequence number, and sample/frame position. Use monotonic host timestamps for
measurements and media positions for audio/video alignment.

### Audio and model execution

- Standardize transport on mono 48kHz PCM in 20ms packets. Use stateful resampling
  at provider/model boundaries and preserve the mapping to the original timeline.
- Accumulate only arrived samples into the model's native chunks. Include
  buffering, lookahead, audio features, generation, and decoding in latency.
- Inspect normalization and feature extraction for hidden whole-file or
  future-audio dependence. A streaming adapter must use a verified bounded
  context or declare incremental input unsupported.
- Keep temporal caches within a session and bounded over long calls. Publish only
  new frames, removing context overlap and final padding.
- Preserve real silence and complete final partial chunks. Padding for model
  conditioning is not additional audible speech or valid output duration.
- Keep inference on a dedicated execution path with bounded queues. Repeated
  independent short-clip generation is not a substitute for session continuity.
- Use the same response audio timeline for conditioning and browser playback;
  do not independently time-stretch the soundtrack.

### Synchronization and backpressure

The voice agent must not play its response directly to the browser ahead of the
avatar. Release audio only when the matching video is ready.

Use LiveKit's avatar integration and AV synchronizer with explicit sample/frame
accounting. The synchronizer helps queue and pace media; its timestamp argument
does not substitute for our media clock, and it does not prove model lip-sync.
[AVSynchronizer reference](https://docs.livekit.io/reference/python/livekit/rtc/synchronizer.html).

Initial bounds:

- Up to twenty seconds of pending response audio, with short replies by default.
- At most two decoded native chunks waiting for publication.
- Approximately 200ms queued at the publishing boundary.
- Pause matching audio/video together on a rendering stall; cancel the response
  and expose the error if a stall exceeds two seconds after playback starts.
- Cancel a response that exceeds the pending-audio bound instead of allowing
  memory or conversational delay to grow indefinitely.

Record actual queue sizes and distinguish model buffering from network jitter.
Do not repeat frames or slow the audio to hide insufficient generation speed.

### Interruption and recovery

Use provider speech-start detection as the authoritative interruption event.

1. Cancel voice generation and invalidate the old response generation.
2. Clear queued audio/video and signal browser playback cancellation.
3. Discard late old-response results, even if a CUDA operation completes later.
4. Restore an adapter-supported state anchored to played media; otherwise reset
   conditioning to the original reference before producing the next response.
5. Truncate voice conversation history at the audio position played through the
   avatar path, not the amount already generated by the provider.

Playback cancellation must work while GPU inference is still occupied. Never
reuse interrupted, unheard future context as if the user saw or heard it.

Use playback feedback from the avatar integration and record whether the offset
is a server estimate or browser-confirmed. Verify interruption handling with the
chosen voice SDK rather than assuming its default audio sink sees avatar delay.
[Realtime conversation handling](https://developers.openai.com/api/docs/guides/realtime-conversations).

## 5. Implementation and test sequence

### Phase A: contracts and local diagnostic mode

Create one `realtime/` area for the browser, control/voice services, shared
contracts, and workers. Keep model-specific imports out of the CPU services.
Reuse existing configs and assets rather than building a second benchmark stack.

Implement a deterministic mock that produces identifiable moving frames and
flash/beep markers through the same media path. Label it clearly as diagnostic
output. Add start/end, busy handling, cancellation, reconnect, and cleanup.

Acceptance: the browser can connect, play aligned test media, interrupt it, and
start a new isolated session without downloading model weights.

### Phase B: AvatarForever installation and 4:3 smoke test on Lambda

This is the first real-model test; do not run FlashTalk first by default.

- Cast the three synthetic reference images and freeze their voice profiles;
  prepare the v2 audio without requiring a live microphone conversation.
- Verify the host and stage the pinned weights, dependencies, portraits, and
  frozen avatar-matched v2 audio. Keep the fixed-voice v1 pack available as a
  separately labeled diagnostic track; do not silently substitute it for v2.
- Run one portrait with `s01_greeting`, seed 0, at direct 640x480 and native 25 FPS.
- Run `s05_pause_resume` to expose silence handling and boundary behavior.
- Respect native frame-count constraints; retain complete audio and record/trim
  conditioning-only final padding in the adapter.
- Validate output dimensions, square pixels, rotation, decodability, duration,
  reference conditioning, and complete audio coverage.
- Record cold load/compilation separately from warm generation, VRAM, and failures.

An upstream-default-resolution run may diagnose a problem but does not count as
the requested 4:3 result. OOM, missing gated dependencies, and geometry errors
are recorded failures; no silent resolution or model change.

Acceptance: reproducible 4:3 file-input results and a measured hardware profile.
This phase does not establish realtime behavior.

### Phase C: persistent AvatarForever streaming worker

Refactor the pinned inference path behind the adapter to consume incremental audio
and emit decoded frames before file completion. Preserve native cache semantics.
Use the mock's same publication path and cancellation contract.

Run both:

- **Paced replay:** release the frozen audio in 20ms packets at wall-clock pace.
- **Throughput replay:** provide audio immediately to measure compute capacity.

Report these modes separately. First run greeting/pause cases, then 60-second
sequences, then ten-minute sequences if shorter tests pass. Verify future-audio
isolation before claiming live input support.

Acceptance: incremental media, bounded queues/cache, correct duration, and
recoverable cancellation. If this cannot be established, mark the adapter
offline-only and record the blocker.

### Phase D: live voice and browser call

Connect the configured OpenAI Realtime voice provider through the voice bridge.
Run or reuse the avatar's image-based voice casting before creating the provider
session, expose optional preview/override controls, and verify the effective voice
in session configuration.
Route all generated audio through the AvatarForever worker. Implement automatic
turn-taking, playback-aware history truncation, and microphone mute/end controls.

Show loading, listening, responding, buffering, and error states. Add a developer
panel with first-response latency, unique FPS, queue depth, interruption timing,
and memory. Distinguish a connected call from a realtime-qualified model.

Acceptance: a repeatable microphone conversation through the same media path used
by replay, including interruption and a second clean session.

### Phase E: evaluation and handoff

Use all three existing portraits and their six frozen avatar-matched v2 audio
scenarios with seed 0 for an 18-output AvatarForever pilot. Record the resolved
profile and actual audio hash for every cell. A second seed is an explicit
extension; the old
108-output matrix is not a prerequisite or an automatic follow-up.

Collect at least thirty warm short turns for latency statistics and a ten-minute
scripted session for continuity. Keep the evaluation inputs out of later training;
upstream pretraining overlap remains unknown.

Provide setup, start, stop, model-selection, mock-test, replay-test, and
troubleshooting instructions. Record dependency locks and environment-variable
templates. Keep checkpoints, secrets, recordings, generated media, logs, and
vendor downloads ignored by Git.

### Measurements and acceptance targets

These are provisional demo targets, not achieved results.

| Measurement | Target or reporting rule |
| --- | --- |
| Geometry | Every generated frame directly 640x480 |
| Warm generation | Sustain native playback rate; about 25 unique FPS at 25 FPS output |
| Realtime factor | <= 1.0; preferably <= 0.8 for compute headroom |
| End-to-end response | Warm p95 <= 3s from user speech end to first synchronized audible/visible response |
| Interruption | p95 <= 300ms from confirmed interrupt to audible stop; report speech-detection delay separately |
| Transport A/V offset | p95 absolute offset <= 80ms on synthetic flash/beep tests |
| Stability | Ten minutes without crashes, unbounded memory/queues, or progressive delay |
| Failure reporting | OOM, timeouts, stalls, unavailable metrics, and unsupported features are explicit |

Record cold loading, compilation, first packet to first decoded chunk, first
browser playback, audio-feature/DiT/VAE time where measurable, useful generated
frames, queue depth, deadline misses, dropped frames, peak VRAM, host RAM, and
GPU-seconds per output second. Detailed profiling runs separately when it changes
normal scheduling.

Store under ignored `runs/<model>/<run-id>/`: metadata, input hashes, source and
weight revisions, resolved environment/configuration, timestamped events, metrics,
and explicitly enabled media. Resume only when inputs and settings match.
Default telemetry excludes microphone audio and transcripts.

Measure transport synchronization separately from model lip-sync. An aligned
file or flash/beep test cannot prove mouth motion matches speech. Keep the initial
automated suite objective-only; identify missing quality collectors explicitly.
Preview the 18 outputs for identity, mouth, torso, and chunk-boundary failures
without inventing a single aggregate quality score or declaring naturalness proven.

### Required verification

- Stateful resampling, pauses, overlap removal, and final partial chunks.
- Future-audio isolation and valid monotonically advancing media positions.
- Delayed packets, bounded queues, stalls, and overload behavior.
- Cancellation races and rejection of late old-response frames.
- Reference/conversation isolation across sequential sessions.
- Actual image delivery to the casting step, fictional-character provenance,
  allowed-catalog output validation, and no hardcoded portrait-to-voice mapping.
- Casting-cache reuse across calls, invalidation after an explicit recast, override
  precedence, timeout/refusal handling, and rejection of unavailable voice IDs.
- Real/unknown-person photos use supplied voice preferences rather than image-based
  gender inference; missing preferences produce a clear preparation state.
- Voice retention across turns, interruption, and reconnect; changing a voice
  starts a fresh provider session rather than mutating an active speaking session.
- Matching selected voice in provider configuration, cached preview metadata,
  replay audio, and run records; unchanged v1 hashes and distinct v2 manifests.
- Worker crash, OOM, failed warm-up, busy requests, and idempotent cleanup.
- Microphone permission denial, autoplay restrictions, reconnect, and TURN routing.
- A flash/beep transport check plus separately reported model lip-sync assessment.
- Exact 4:3 geometry and honest unique-frame counts throughout long sessions.

Platform completion and model qualification are separate outcomes. If AvatarForever
does not meet the speed or quality requirement, retain its measurements and test
another adapter/configuration explicitly. Do not quietly substitute FlashHead Lite
or describe buffered, repeated-frame playback as a successful realtime demo.

## 6. Retained baseline work and later training

The earlier LiveAct / LiveAvatar / FlashTalk comparison remains useful as an
offline quality reference. Preserve its configs, scripts, assets, partial outputs,
and [running guide](docs/running-baselines.md). The previous 108-output batch was
stopped; this plan does not resume it. Prior geometry variants must remain labeled
and cannot count as direct-4:3 results.

Retain known compatibility findings when those models are revisited:

- The tested LiveAvatar BF16 path on an 80GB H100 needed CPU KV-cache offloading at
  640x480 and conditioning-cache growth beyond upstream's fixed capacity. Preserve
  the adapter/hash and measure its speed cost.
- The pinned FlashTalk config emits 28 new frames at 25 FPS: 1.12 seconds of media
  per chunk. Include audio accumulation in live latency. Existing cold-inclusive
  file timings are not controlled warm streaming results.
- Released inference weights do not establish a supported fine-tuning recipe.

After the demo, select a teacher using quality, feasible training access, license
terms, and measured hardware cost. Verify a small adaptation and checkpoint
save/reload before using a large dataset. Distillation follows that decision and
must preserve image/audio conditioning and long-session stability.

Keep [TalkingMachines](https://arxiv.org/html/2506.03099),
[Self Forcing](https://arxiv.org/abs/2506.08009),
[LongLive](https://arxiv.org/abs/2509.22622),
[StreamAvatar](https://arxiv.org/html/2512.22065),
[LongLive-2.0](https://arxiv.org/abs/2605.18739),
[LTX-2](https://arxiv.org/abs/2601.03233),
[OmniForcing](https://github.com/OmniForcing/OmniForcing), and
[Omni-LiveAvatar](https://github.com/Aoko955/Omni-LiveAvatar) as research references.
Recheck release status before adopting them; joint audio/video models are not
automatically interchangeable with a fixed-audio avatar renderer.

The implementation status above distinguishes completed code/tests from remaining
integration and qualification. GPU runs, cloud provisioning, training, and public
deployment remain separate execution steps. Preserve the existing instance
lifecycle preference: the user handles instance termination.
