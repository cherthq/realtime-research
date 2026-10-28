# Realtime avatar demo

**New machine or teammate? Start with [Recreate the demo](../docs/recreate-demo.md).**
The code, dependency constraints and setup commands are public; the required
portrait/audio pack is transferred separately with `realtime.scripts.demo_assets`.

An experimental two-panel video-call application with OpenAI voice, a persistent
AvatarForever worker on Lambda, and self-hosted LiveKit media transport. Generated
video is directly 640x480. The [plan](../plan.md) describes its architecture and
the [qualification record](QUALIFICATION.md) records measured results and failures.

Real voice/avatar browser calls have passed for all three prepared characters.
Optional transformer-block compilation reached 38.62 generated FPS in a ten-minute
paced worker replay on one H100 80GB. GPU throughput is separate from full browser
response latency, delivery reliability and perceptual lip-sync quality.

The UI and backend tests have passed in Ubuntu. Three optional transport checks
also passed in a separate real-LiveKit run; remote browser checks cover startup,
interruption and connection recovery. Automated media tests use synthetic inputs.
Physical microphone/speaker checks and public-network deployment remain separate.
The diagnostic renderer is explicitly labeled and remains distinct from generated
AvatarForever motion.

`GET /api/models` includes each adapter's native geometry, PCM format, first and
following chunk sizes, buffering requirements, conditioning and reset behavior.
AvatarForever is `streaming_experimental`; this status does not assert that every
performance target passes. It uses only arrived chunks, but a frame can depend on
later audio inside its chunk. The first chunk needs 1.0s of audio and subsequent
chunks need 1.28s. New architectures must implement their own compatible backend;
changing a model ID alone is not an adapter implementation.

## Local media server: no LiveKit account needed

From the repository root, use Python 3.11 or newer:

```powershell
py -3.11 -m venv .venv-realtime
.venv-realtime/Scripts/python.exe -m pip install -r realtime/requirements.txt
.venv-realtime/Scripts/python.exe realtime/scripts/livekit_local.py run
```

On Linux:

```bash
python3.11 -m venv .venv-realtime
.venv-realtime/bin/python -m pip install -r realtime/requirements.txt
.venv-realtime/bin/python realtime/scripts/livekit_local.py run
```

The installer downloads the pinned official LiveKit 1.13.11 release, checks its
SHA-256 digest, and extracts only its executable into ignored `.cache/livekit/`.
It supports Windows and Linux on x64 and ARM64. `install` downloads without
starting the server. `run` stays in the foreground; Ctrl+C stops it.

The server binds to `127.0.0.1:7880`. Its local development connection is
`ws://127.0.0.1:7880`, with API key `devkey` and secret `secret`. These values are
only for local development. This port is a signaling endpoint, not the web app.

Copy `realtime/.env.example` to `realtime/.env` when configuring the backend. Keep real
OpenAI and LiveKit signing secrets there, never in browser code or Git. A LiveKit
Cloud subscription is unnecessary for self-hosting. The local media server does
not require an OpenAI key or a Lambda GPU.

The initial remote demo uses the SSH profile below. Public website deployment
requires HTTPS/WSS, authentication, the server's public address and TURN. The
local command binds HTTP signaling to loopback, but LiveKit's media listeners
bind other interfaces; keep them blocked to public ingress. Do not expose the
development credentials publicly.
See [LiveKit's deployment documentation](https://docs.livekit.io/transport/self-hosting/deployment/).

## Build and start the diagnostic application

The call screen has two 4:3 tiles: your local camera and the avatar. Camera,
microphone, call, and interrupt controls sit underneath. Open **Settings** for
avatar/model/voice selection, **Test reply**, and optional connection details.
On mobile the tiles stack vertically. Camera preview is opt-in, stays in the
browser, and stops on call end or page exit; it is never published to LiveKit.
The header and tab icon use Chert's [official logo](https://www.trychert.com/chert.png),
resized to 128×128 and bundled locally at `web/public/chert-logo.png`.

The layout follows the supplied two-panel reference and the restrained call
controls in [Google Meet's layout guide](https://support.google.com/meet/answer/10550593?hl=en).
Capture and cleanup follow the browser's
[getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
and [track stop](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/stop) APIs.

With the media server running, build the web application:

```powershell
cd realtime/web
npm ci
npm run build
cd ../..
.venv-realtime/Scripts/python.exe -m realtime.app
```

On Linux, use `.venv-realtime/bin/python -m realtime.app`. Open
<http://127.0.0.1:8080>. The application needs the prepared portraits and fixed
audio under `data/eval/facetime-zero-shot-v1/`; these generated assets are ignored
by Git. Their preparation is documented in
[the baseline protocol](../docs/zero-shot-baseline.md).

### Windows browser with services in WSL

If the page hangs on Windows localhost or WSLg opens unusable windows, use the
local bridge from the repository root:

```powershell
.venv-realtime/Scripts/python.exe -m realtime.scripts.windows_demo
```

Or double-click `realtime/start-demo.cmd`. Open <http://127.0.0.1:18800> in
Windows Chrome or Edge; keep the launcher running. Ctrl+C stops its services.
The launcher requires Ubuntu in WSL, the built frontend, prepared assets, and
a WSL Python environment with `realtime/requirements.txt` installed. It defaults
to `~/.cache/chert/realtime-venv/bin/python`; pass `--wsl-python /absolute/path/to/python`
and `--distro NAME` for a different environment. It does not install dependencies.

The bridge buffers complete HTTP responses into one write and uses 100ms
sequential polling for call control. WebRTC connects directly to WSL's private
address over TCP. The launcher rediscovers that address each time, generates
private media credentials, and starts its own API/media processes on ports
18801–18803. It does not change browser or firewall settings. This route is for
local development; do not forward these ports publicly. WSL logs and credentials
stay under `~/.cache/chert/windows-demo/`, outside Git. Normal hosts continue to
use SSE; Lambda/team access should use the SSH profile below.

For the same Windows networking issue with a Lambda backend, prepare its private
profile with `--client-address <WSL-private-IPv4>` and a distinct port trio such as
`--api-port 19881 --signal-port 19882 --media-port 19883`. Run the SSH client inside
WSL and forward those three ports on that private WSL address to the matching
Lambda loopback ports. Keep the SSH key in WSL with mode 600 and verify the host
key normally. Then start the Windows HTTP bridge only:

```powershell
.venv-realtime/Scripts/python.exe -m realtime.scripts.windows_demo --upstream <WSL-private-IPv4>:19881 --port 18810
```

Open `http://127.0.0.1:18810`. Browser media reaches WSL's SSH forwards; the backend's
media connection stays on Lambda loopback. The profile enables polling for this
route. WSL's address can change after a restart; update the profile and forwards
together. This does not open cloud firewall ports. The private profile contains
the access code; do not commit it.

The diagnostic controls exercise portrait selection, a single active session,
fixed reply playback, cancellation, and optional microphone interruption. Model inference FPS is not measured by the diagnostic renderer, and server
timing is not browser playback latency. AvatarForever stays disabled by default;
its experimental adapter requires explicit configuration on a prepared GPU host.
No cloud account, OpenAI key, or GPU is required for diagnostic mode.

## Character voices and live conversation

Voice replies use `max_output_tokens: "inf"` (the provider's maximum available
output), with no application token ceiling or fixed word-count instruction.
A provider `incomplete` outcome (such as `max_output_tokens`) ends that reply,
drains its received audio, and keeps the call available for the next turn. It
never triggers an automatic retry. Actual provider failures still end the call.
Live voice can buffer up to 300 seconds of unplayed PCM audio (28.8 MB), allowing
long replies that arrive faster than playback. This is a CPU backlog guard, not a
reply-duration limit; consumed audio frees capacity and interruption clears the
backlog. Video/GPU queues remain bounded separately. Fixed replay inputs retain
their 20-second bound. Session reports include
bounded `voice_diagnostics` status/reason codes and token counts, without reply
text, microphone audio, provider error messages, or API keys.

The **Live voice** mode streams microphone input to the configured Realtime model
and routes its reply audio through the selected avatar worker. With the diagnostic
renderer selected it still uses simulated video. Real provider calls are included
in the qualification record. Add `OPENAI_API_KEY` privately to `realtime/.env`
when ready, and restart the backend; never put the key in frontend variables.

Prepare the library and match the documented synthetic references explicitly:

```powershell
.venv-realtime/Scripts/python.exe -m realtime.scripts.prepare_voices --cast --freeze-pack data/eval/facetime-avatar-voices-v2
```

This command makes paid API requests. It generates actual Realtime voice samples,
describes their sound using a model review, and casts each fictional image from
the resulting catalog. Descriptions are model-reviewed, not human-reviewed.
Preview or override the result in the UI. A real/unknown portrait requires a
supplied preference; it is not sent to image-based gender classification.
Assignments remain fixed until an explicit rematch. Changed images/catalogs
invalidate automatic reuse instead of silently selecting another voice.
With the catalog prepared, starting a live call automatically creates a missing
synthetic-character match before reserving the GPU. A saved match is reused, and
an explicit voice selection skips casting. Invalidated matches still require
**Rematch**; unknown/real-person provenance still requires a supplied preference.
Preparation time is stored separately in `voice_selection.preparation_ms`.

The frozen v2 pack shares waveforms across identical profiles, includes exact
PCM silence, validates generated transcripts against the scripts, and records
image/audio hashes and provider metadata. It preserves v1 assets. Re-running
with unchanged inputs resumes without regenerating saved waveforms. Changed
inputs require a new destination. If preparation is interrupted, inspect the
`.preparing` lock and confirm no preparation process is running before removing
that file and resuming.

Live replies have bounded input/output queues. Interruption invalidates late
packets and mutes browser output immediately on a local click or server cancellation
event. It remains muted until a frame from a newer response is displayed. Frame IDs
travel with the actual video; the browser acknowledges them from
`requestVideoFrameCallback`, rather than guessing progress from HTTP delivery.
Audio travels through a Web Audio gain gate, so autoplay recovery cannot unmute
an interrupted response. The audio context and its graph are released on call end.
Both published tracks use the same `avatar` stream name. The browser requests an
80ms video jitter-buffer target to reduce video leading the audio path; this is
a browser hint, not a fixed delay or a guarantee on other networks/devices.

The pinned LiveKit 2.23 metadata path enables legacy encoded streams in Chromium
but does not connect audio receivers. `playback.ts` supplies an identity transform
for those receivers so received audio actually decodes. This compatibility path
is scoped to this app's non-E2EE room; revisit it when upgrading the SDK. It does
not modify installed dependencies. See the SDK's
[metadata receiver implementation](https://github.com/livekit/client-sdk-js/blob/v2.23.0/src/frameMetadata/FrameMetadataManager.ts)
and the browser's [jitter buffer target](https://developer.mozilla.org/en-US/docs/Web/API/RTCRtpReceiver/jitterBufferTarget).

Voice-history truncation uses the last displayed-frame position reported with
audio playback enabled, bounded by server publication progress. The server can
wait up to 200ms for the cancellation receipt after cancelling output. Missing
feedback retains zero potentially unheard audio instead of pretending it played.
This is a **browser video/audio-enabled proxy**, not a physical-speaker measurement.
Frame metadata requires the pinned SDK versions and a compatible browser; the
local Chromium tests exercise it. Foreground playback is tested; background-tab
and device-specific playback policies still need qualification.

Cancellation tests cover repeated responses, server-driven interruption and a
600ms delayed HTTP acknowledgement. Reports under ignored
`runs/realtime/browser-tests/` separate the browser mute operation from the server
acknowledgement round trip. Neither field is labeled acoustic stop latency. The
frame and cancellation flow follows [LiveKit frame metadata](https://docs.livekit.io/transport/media/frame-metadata/)
and [OpenAI interruption/truncation guidance](https://developers.openai.com/api/docs/guides/realtime-conversations#interruption-and-truncation).

An opt-in test reads the real post-gate audio analyser and compares tone onset to
video compositor callbacks. The sync fixture starts at exact 25 FPS boundaries
(0.52s and 2.0s). A separate four-second tone tests local and server interruption
without confusing cancellation with a short beep ending naturally. The latest
local private-TCP sample recorded ten markers with 34.8ms maximum/p95 absolute
offset, and audio-graph stop times of 32ms and 18.2ms. This small Linux Chromium
sample is not a remote-network, physical-speaker or avatar lip-sync qualification.
RAF sampling adds up to one display interval; the report records analyser and
audio-context latency information. The qualification record also includes the
remote Lambda samples and their failed interruption target. Physical output still
needs testing.

Session
reports go to ignored `runs/realtime/sessions/`; microphone audio and conversation
transcripts are not recorded. The call limit is ten minutes with a thirty-second
disconnect grace period.
Ending a call starts one shared cleanup operation. An HTTP disconnect does not
cancel it, and repeated end requests await the same work. The single-session
slot stays busy until worker teardown, room deletion and report handling finish.
Tests inject cancelled requests and a media-queue error to verify that GPU and
media cleanup still run. Worker IPC and individual resource releases keep their
own deadlines; there is no outer ten-second timeout that frees a draining slot.

## AvatarForever on Lambda

Run these only on the intended Linux GPU machine. The separate GPU environment
has been tested on an H100 80GB with Python 3.11 and PyTorch 2.7.1/CUDA 12.8.
See [qualification results](QUALIFICATION.md) for the tested scope and limits.

```bash
bash infra/setup-avatarforever.sh
.venv-avatarforever/bin/hf auth login
.venv-avatarforever/bin/python scripts/download_baseline_weights.py avatarforever --root /path/to/persistent/checkpoints
.venv-avatarforever/bin/python -m realtime.avatarforever \
  --pack data/eval/facetime-avatar-voices-v2 \
  --weights /path/to/persistent/checkpoints \
  --output runs/avatarforever/greeting-ref01 --portrait ref01 --scenario s01_greeting
```

Accept Gemma's upstream access terms through Hugging Face before downloading the
gated text encoder. The downloader pins both weight repositories and retains
available license/README files. See the separate code, checkpoint, backbone and
Gemma entries in [the component record](QUALIFICATION.md#component-terms).

Use `--dry-run` to inspect the exact native command after preparing the v2 pack.
The runner enforces direct 640x480, seed 0, native 25 FPS and `8n+1` conditioning
frames. It removes final temporal padding and muxes the original audio without
resizing video. Geometry, duration, hashes and decodability are checked. Cold
process timing includes loading/compilation and is not a realtime benchmark.
Repeat with `s05_pause_resume` and a fresh output directory.
An experimental incremental AvatarForever process and publisher are implemented;
native GPU execution and real voice/browser calls at direct 640x480 have passed.
Streaming-boundary equivalence to upstream full-file inference, perceptual
lip-sync quality and physical-device timing remain separate qualification work.

The GPU process retains loaded weights between calls. Each call replaces its
reference conditioning and clears audio/latent history before reusing the model.
Changing to the diagnostic renderer unloads the idle model. Interruptions invalidate
queued and in-flight results; an unresponsive worker is retired rather than reused.
Health and model endpoints distinguish configured, loading, loaded and leased
states. A loaded model is still experimentally qualified, not certified realtime.

## Private Lambda demo over SSH

Use a Linux host and keep the repository, assets and checkpoints on persistent
storage. Install the CPU requirements in `.venv-realtime` and build the frontend
as above; the GPU environment remains separate. Transfer the three prepared
portraits and v1 inputs under `data/eval/` (ignored by Git). Privately configure
`realtime/.env`, then run the voice-preparation command above when API access is
ready. For the actual avatar, set `AVATARFOREVER_ENABLED=true` and absolute
`AVATARFOREVER_PYTHON` / `AVATARFOREVER_WEIGHTS` paths after the GPU smoke tests.

Keep public ingress limited to SSH. The initial profile uses ICE/TCP through the
tunnel and needs no public media ports, domain or LiveKit Cloud account. LiveKit's
TCP media listener binds all interfaces, so the cloud/host firewall must block
public access to that port; the helper does not change firewall rules. This is a
private test profile, not the later public UDP/TURN deployment.

On Lambda, from the repository root:

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_host prepare
.venv-realtime/bin/python -m realtime.scripts.demo_host check
.venv-realtime/bin/python -m realtime.scripts.demo_host serve
```

`prepare` writes new random LiveKit credentials and an access code to ignored
`.cache/realtime-host/`, with private Linux permissions. It refuses to overwrite
an existing profile. `check` validates configuration without printing secrets;
it does not certify GPU availability or the firewall. `serve` starts the two
services, checks startup, supervises them and stops only its own children. The
profile overrides development LiveKit credentials from `.env`; OpenAI settings
and model paths still come from `.env`. Logs stay in the private profile folder.
Keep `serve` in an attached terminal or a persistent terminal session such as tmux.

For an Ubuntu user service that survives SSH disconnects, first stop any manually
started `serve` instance, then run from the repository root:

```bash
sudo loginctl enable-linger "$USER"
systemd-run --user --unit=chert-demo --property="WorkingDirectory=$PWD" \
  --property=Restart=on-failure --property=RestartSec=5 --property=TimeoutStopSec=45 \
  "$PWD/.venv-realtime/bin/python" -m realtime.scripts.demo_host serve
systemctl --user status chert-demo
```

This is a transient service; run the command again after a host reboot. Use
`systemctl --user stop chert-demo` to stop it. This does not terminate Lambda.
Keep the generated profile to preserve the access code across service restarts.
Browser login tickets are held in memory: unlock again after restarting the API.

On the laptop, first stop any local demo occupying the same ports:

```powershell
.venv-realtime/Scripts/python.exe -m realtime.scripts.demo_host tunnel --host ubuntu@YOUR_LAMBDA_IP --identity C:\Users\YOUR_NAME\.ssh\id_ed25519
```

This forwards **8080 (web/API), 7880 (signaling), and 7881 (media)**. It preserves
SSH host-key verification and fails if a local port is occupied. Use `--dry-run`
to inspect the command without connecting. Alternate ports must match between
`prepare` and `tunnel`; both accept `--api-port`, `--signal-port`, `--media-port`.

Open <http://127.0.0.1:8080> and enter `access_code` from the private server
`profile.json`. Read it privately, not through a Git commit or chat message.
The browser stores a revocable, HttpOnly session cookie rather than the access
code. Images, previews, API calls and event streams use that same cookie.
**Settings → Lock demo** ends the call and revokes this browser's access.

Stop the remote `serve` command with Ctrl+C to stop its API and LiveKit. Closing
the laptop's tunnel only disconnects the browser; it does not stop the remote
services or the billable Lambda instance. Only the user terminates the instance.

The TCP-only profile passed local Linux browser media tests, including selected
ICE candidate verification. Supervisor shutdown was also checked: all three test
listeners closed while the separate local demo remained running. The actual
Windows-to-Lambda SSH path, OpenAI calls and H100 generation have also passed
three-character smoke tests; see the qualification record for timing limits. If a page loads
but media does not, check all three tunnel forwards and server logs before
changing the model or opening public ports.

## Streaming replay and evaluation

After native GPU smoke tests and the frozen v2 pack are ready, the CPU environment
can drive the separate resident GPU process. It submits only arrived 20ms PCM
packets, preserves final partial chunks and records exact frame coverage. Saved
videos retain native 640x480 frames; the final video frame can extend less than
40ms beyond the audio. No resize, interpolation or audible padding is added.

```bash
# One paced greeting; add --dry-run to validate inputs without starting the GPU.
.venv-realtime/bin/python -m realtime.evaluate \
  --pack data/eval/facetime-avatar-voices-v2 --weights /path/to/persistent/checkpoints \
  --output runs/avatarforever/paced-greeting --save-video

# The 18-output pilot: three portraits, six fixed scenarios, seed 0.
.venv-realtime/bin/python -m realtime.evaluate \
  --pack data/eval/facetime-avatar-voices-v2 --weights /path/to/persistent/checkpoints \
  --portrait all --scenario all --output runs/avatarforever/pilot-18 --save-video

# One first turn plus 30 reused-model turns for warm timing.
.venv-realtime/bin/python -m realtime.evaluate \
  --pack data/eval/facetime-avatar-voices-v2 --weights /path/to/persistent/checkpoints \
  --repeat 31 --output runs/avatarforever/warm-30

# Optional compiled blocks: compile/warm before replay, then measure warm work.
.venv-realtime/bin/python -m realtime.evaluate \
  --pack data/eval/facetime-avatar-voices-v2 --weights /path/to/persistent/checkpoints \
  --compile-blocks --load-timeout 900 --output runs/avatarforever/compiled-greeting
```

Use a fresh output directory each time. Repeat a greeting with `--mode throughput`
to measure compute capacity separately from paced input. Run
`--scenario s05_pause_resume --duration-seconds 60` first; after inspecting its
results, use `--duration-seconds 600` in another directory for continuity. Duration
tests repeat that frozen waveform and explicitly record the repetition. They do
not synthesize a new conversation or silently resume the old 108-output batch.

Reports include input/code/weight hashes, voice selection, cold load, first decoded
chunk, p50/p95 warm timing, useful generated frames, exact duplicate-frame hashes,
compute realtime factor, pending audio, and GPU-process memory samples. GPU peaks
cover the process lifetime and PyTorch allocations, not every allocation by the
driver or other processes; host memory is Linux peak RSS. Optional videos include
hashes, probe data and decode/geometry/frame-count/audio-duration checks. Paced
overload fails explicitly. Failed replays retain partial decoded-chunk metrics
and the underlying failure type. Existing output folders are not overwritten.

The replay reports do **not** measure browser audible latency, physical-device A/V
offset, identity quality or model lip-sync. Those collectors and real-hardware
qualification remain separate work. A completed replay is not a realtime pass.

During a brief control-connection outage, the browser gates audio/video and retries
the same session for up to 30 seconds. It resumes only after current session status
and a valid new frame arrive; the selected voice remains attached to that session.
An interrupted event stream falls back to bounded status polling. Authentication
expiry ends the call immediately, and an outage beyond the grace period requires
a new call. Media reconnect and status recovery hold playback independently.

Regional compilation is optional. To enable it for a hosted demo, set
`AVATARFOREVER_COMPILE_BLOCKS=true` and `GPU_LOAD_TIMEOUT_SECONDS=900` privately
in `realtime/.env`, restart the idle service, then prewarm on that host:

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_host warm \
  --directory .cache/realtime-host --timeout 945
```

Prewarming refuses to interrupt an existing call, loads/compiles with silent dummy
input, releases its temporary session, and leaves the GPU weights resident. Run it
before opening a browser call so compilation is not on the user's response path.
The report records compile preparation separately from warm generation. This
option keeps BF16 weights, native dimensions and denoising steps; floating-point
fusion can change pixels slightly. Use the eager default for the original
execution baseline. Read [qualification](QUALIFICATION.md) for measured limits.

## Validation

The paid end-to-end check is explicitly opt-in and requires a configured remote
instance. It feeds a synthetic WAV through a fake browser microphone, verifies
real provider speech and AvatarForever frames, then interrupts and ends the call:

```powershell
$env:RUN_REAL_AVATAR = '1'
$env:REAL_AVATAR_MIC_WAV = 'D:\path\to\synthetic-utterance-with-trailing-silence.wav'
$env:REAL_AVATAR_ID = 'ref01' # Also ref02 or ref03; each report is kept separately.
$env:DEMO_TEST_URL = 'http://127.0.0.1:18810'
# Load DEMO_TEST_ACCESS_CODE privately from the SSH profile if access is enabled.
cd realtime/web
npm run test:e2e -- real-avatar.spec.ts
```

This check does not use the physical microphone. It writes an ignored screenshot
and metrics under `runs/realtime/real-avatar-browser/`. It does not establish
physical-speaker latency, model lip-sync quality, or ten-minute stability.

For a warm conversational latency baseline, use one frozen synthetic mono WAV
containing a short prompt (the prepared prompt is “Please say hello in three
words.”). The opt-in collector sends it through a synthetic browser microphone
31 times in one call, excludes the first turn, and records the same browser clock
for input completion, output audio-graph onset and video-compositor presentation:

```powershell
$env:RUN_WARM_VOICE_BASELINE = '1'
$env:RUN_BROWSER_TIMING = '1'
$env:WARM_VOICE_INPUT_WAV = 'D:\path\to\frozen-synthetic-prompt.wav'
$env:WARM_VOICE_OUTPUT = 'D:\path\to\new-latency-run'
# Reuse DEMO_TEST_URL and private DEMO_TEST_ACCESS_CODE from the hosted test above.
npm run test:e2e -- warm-voice.spec.ts
```

This consumes provider requests and GPU time. It stores waveform hashes, trim
boundaries, per-turn timings and failures, without recording a physical microphone
or storing generated speech/transcripts. Output directories must be new. Input
silence is trimmed with recorded 20ms RMS windows; the test rejects an early
response or a substantially delayed input-end callback. Audio is observed through
the actual post-gate graph. Video uses a bounded performance mark at playback
opening, so status-request latency does not contaminate presentation timing.
Neither measurement establishes physical speaker delay or perceptual lip-sync.
`WARM_VOICE_TURNS=3` is a short collector check. `WARM_VOICE_MODEL=mock` explicitly
labels a voice/transport calibration and **does not qualify avatar generation**;
the default model is AvatarForever.

Set `WARM_VOICE_SESSION_SECONDS=600` to spread the 31 turns across one ten-minute
call and observe the automatic call limit. This additionally records periodic
queue/memory samples and browser frame continuity, verifies native geometry, and
checks that cleanup releases the session. It is a conversational soak with idle
intervals, separate from the uninterrupted ten-minute worker replay. The latency
target is still checked; a completed stability run can fail that separate target.

Live metrics include `voice_timing`: intervals from server receipt of the provider's
speech-stop, response-created and first-audio events to first media publication.
All intervals use one monotonic server clock. They exclude upstream VAD delay and
downstream browser playback, and must not be labeled end-to-end latency. No audio
or transcript is stored by this instrumentation.

The GPU publisher keeps both media tracks flowing during pauses with held images
and silence. These frames are reported separately from generated motion; idle
silence neither extends a completed response nor advances its playback history.
Read the qualification record for both completed measurements and failed runs.

```powershell
.venv-realtime/Scripts/python.exe -m pytest realtime/tests -q
$env:RUN_LIVEKIT_TESTS = '1'
.venv-realtime/Scripts/python.exe -m pytest realtime/tests/test_transport.py -q
cd realtime/web
npm run test:e2e

# Autoplay, microphone denial, temporary/persistent control loss,
# expired credentials, and SSE-to-poll recovery (no paid inference).
npm run test:e2e -- recovery.spec.ts

# Optional real output-graph timing; plays diagnostic tones.
$env:RUN_BROWSER_TIMING = '1'
npm run test:e2e -- transport-timing.spec.ts
```

The transport test requires the local LiveKit server; browser tests also require
the built application on port 8080 and Microsoft Edge by default. Set
`PLAYWRIGHT_CHANNEL=chromium` for an installed Playwright Chromium on Linux.
On Linux, use `RUN_BROWSER_TIMING=1 npm run test:e2e -- transport-timing.spec.ts`.
The timing test asserts <=80ms p95 absolute graph/video offset and <300ms graph
stop time, and writes ignored `runs/realtime/browser-tests/transport-timing.json`.
It removes Playwright's default audio mute for that run. To test a private profile,
set `DEMO_TEST_URL` and load `DEMO_TEST_ACCESS_CODE` privately from its profile;
never check that value into scripts or documentation.
A successful HTTP health
check alone does not verify WebSocket signaling or media delivery.

On some Windows setups, native loopback responses or WSLg browser windows may
not work reliably. The Windows launcher above provides a private WSL bridge.
Windows Chrome browser checks passed through this route:
two-panel 640×480 calls, test audio, microphone publication, interruption,
cleanup/reconnection, mobile layout, and camera permission handling. The private
access-code test is skipped on this unauthenticated local profile. A separate
Windows timing check also passes: ten markers measured 69.9ms p95 absolute
audio-graph/video-compositor offset, with local/server interruption at 3.3/112.4ms
in that run. These are diagnostic measurements, not physical-speaker latency,
real provider access, or GPU avatar inference qualification.

For the Linux Python 3.11 application server, install `requirements.txt` with
`-c realtime/constraints-linux-py311.txt` from the repository root. These constraints
record the deployed server versions. Unit tests generate temporary synthetic
media and isolate private `.env` settings, so they run without an asset handoff,
OpenAI key, LiveKit service or GPU. Transport tests remain explicit opt-in.

The checked-in `requirements-lock-windows.txt` records the Python 3.11 development
environment; `requirements-lock-linux.txt` records the Python 3.14 WSL test
environment, including browser test dependencies. The GPU environment remains
Python 3.11 and writes its own resolved lock during setup. Do not interchange
platform-specific locks blindly.
