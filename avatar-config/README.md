# Standalone avatar settings

A separate dashboard for editing a prompt and driving image, viewing deployment
history, and opening the existing browser or FaceTime call.

```sh
cd avatar-config
npm run dev
```

Open http://127.0.0.1:4198. Requires Node 22+, with no package installation.

Choose/drop a JPEG or PNG and optionally expand **Adjust crop**. The still preview
is normalized to 640×480. **Save locally** stores the prompt and image together in
IndexedDB with stale-tab conflict detection. Local saves do not change live calls.

**Run local preview** opens the existing backend at http://127.0.0.1:18837.
**Call on FaceTime** uses the configured number with Apple's native `facetime:`
video-call handler. Calls use the deployed settings, not the saved local draft.

## Private deployment records

Fresh checkouts contain no private images, deployment records, credentials, or
callable FaceTime number. Supply an ignored `local/deployments.json` to populate
the dashboard. Use the shape of `dist/deployments.json` with these fields:

- `previewUrl`: loopback HTTP URL for the existing web call.
- `facetimeNumber`: E.164 phone number or `null` to disable the link.
- `recordedAt`: ISO date, or `null` when no deployments exist.
- Optional `verifiedAt`: ISO date of the latest read-only runtime verification.
- `current`: deployment record or `null`.
- `previous`: array of older deployment records.

Each record has `id`, `date` (ISO date), `detail`, `prompt`, `greeting`, and an
`image` object with `src`, `label`, and `note`. `src` is a PNG/JPEG base64 data URL
or `null` for an unavailable image. Records are read-only snapshots, not a live
subscription or a complete deployment ledger. Prompts can be expanded to read
in full. Browser drafts never alter these records.

Keep `local/` private and excluded from Git. The loopback server only serves its
explicit allowlist; it never serves credential files. CSP permits same-origin
snapshot reads and blocks external fetch/WebSocket requests.

## Existing runtime connection

Use an existing SSH tunnel to the backend, or the optional `npm run preview:connect`
helper. The latter uses an already-authorized Lambda Jupyter workspace through its
[terminal API](https://jupyter-server-terminals.readthedocs.io/en/latest/api.html).
It needs ignored `local/connection.json` with `base` (HTTPS workspace URL) and
`token` (existing credential). Protect this directory with mode 0700 and the file
with mode 0600. Never publish or commit it.

The helper creates one temporary terminal running an in-memory TCP relay to the
existing backend on 127.0.0.1:18837. Its local listener is also loopback-only. It
installs no packages, writes no remote source/configuration, and does not restart
services or create another avatar backend. Ctrl-C or SIGTERM closes the listener
and its own terminal. After abrupt termination, its terminal ID can be found in
`local/tunnel-state.json` for cleanup.

## Checks and integration status

Run `npm run check` and `npm test`. Browser verification covers saved drafts,
image decoding/cropping, conflicting tabs, deployment cards, mobile layout, and
opening the existing call page. No test call is placed by these checks.

Phase 2 is described in [the integration plan](../docs/line-avatar-configuration-plan.md).
The dashboard remains independent of the Chert dashboard/control plane.

## Phase 2 connection

The optional integration adds **Publish**, **Edit** on the current deployment,
and **Restore** on previous versions. **Save locally** still keeps a browser draft.
Live controls appear only after an authenticated runtime read succeeds. A failed
connection returns to the labeled recorded snapshot; it never simulates a publish.

After the runtime integration is installed, put its separate configuration token
in ignored mode-0600 `local/runtime.json` as `{"token":"<configuration token>"}`.
The loopback dashboard server forwards only `/api/configuration` to the existing
preview connection's `/api/avatar-config`. It checks Host/Origin, limits request
size and keeps credentials server-side. No arbitrary upstream URL is accepted.

The editor loads the current deployment on first use when no local draft exists.
Subsequent refreshes preserve drafts. **Edit** requires saving unsaved edits first.
A prompt-only publication preserves an unchanged deployed image byte-for-byte.
Image switching remains gated until the actual trained renderer passes acceptance.
The existing automatic FaceTime greeting is separate and remains unchanged.

Runtime installation, maintenance and rollback are documented in
[`avatar_config_runtime/README.md`](../avatar_config_runtime/README.md).
The loopback server is for development. Production hosting uses the separate
Vercel project described below. No Chert dashboard, database or control plane is
modified by this integration.

The initial live integration was installed after an approved idle restart. Baseline
prompt/image equality, authentication, stale writes, the image gate and the Ready
web preview were verified. Recorded pre-integration history remains visible as
read-only cards. Image changes are rejected in the dashboard before transmission
until the runtime reports that renderer switching has passed acceptance.

## Hosted dashboard

Production: https://realtime-avatar-dashboard.vercel.app/ (Vercel sign-in required).
This is the independent `realtime-avatar-dashboard` project in Gary's personal
Vercel workspace, rooted at `avatar-config/`, running Node 24.

Vercel Authentication must cover **all deployments**, including production.
The server functions proxy only the existing runtime's `/api/avatar-config`.
The existing Lambda Cloudflare connector routes that exact path to port 18837;
its other workspace routes remain unchanged. No extra avatar/GPU service runs.

Production environment variables:

- `AVATAR_RUNTIME_URL`: HTTPS URL ending in `/api/avatar-config`.
- `AVATAR_RUNTIME_TOKEN`: sensitive, scoped configuration credential only.
- `AVATAR_DEPLOYMENT_RECORDS`: sensitive recorded metadata with image SHA-256
  values; image data is hydrated from verified runtime history during the build.

`npm run build` produces eight public assets and a private function-only history
file. The broad workspace credential is never sent to Vercel. Prompts, images and
configuration credentials must never be committed or placed in static assets.
Builds fail if recorded images cannot be verified against runtime history.

The hosted configuration UI works independently of the laptop. **Run local
preview** still opens the existing loopback call page and needs the local preview
connection; **Call on FaceTime** uses the native FaceTime handler on the device.
Browser drafts are origin-specific and do not migrate from localhost.

## Call logs

The **Call logs** button opens `/calls.html`. Calls after recording is enabled are
listed with channel, connected duration, status and transcript. Downloads include
mixed audio, separate caller/agent WAV files, a side-by-side MP4 with both voices,
and a plain-text transcript. Video is captured at 15 fps (960 × 360 composite);
this does not change the live call renderer. Caller video appears only while the
camera track is available. Existing GPT-Live transcripts are archived without
additional transcription API requests.

Enable `record_calls: true` in the existing private runtime configuration. Files
and the SQLite ledger live in its private `calls/` subdirectory on the Lambda
host. Earlier unrecorded calls cannot be reconstructed. Files are retained without
automatic deletion. At 5 GiB remaining, recording stops with a visible error while
calls continue. Operator backup/storage expansion is needed for indefinite archival;
this host-local storage is not an independent backup.

List/detail endpoints require the existing server credential. After Vercel sign-in,
downloads use a five-minute HMAC ticket scoped to one call and file, support HTTP
Range, and bypass serverless response-size limits. No recording enters Git or the
static build. Recording failures and interrupted shutdowns remain visible in the
ledger; partial recordings are never labeled complete.

The optional recorder uses the existing LiveKit project and installed FFmpeg,
with a subscribe-only room participant and two CPU encoding threads. It does not
start a second GPU or require a new OpenAI key. FaceTime requests carry a channel
header; the deployed companion's authenticated greeting also identifies FaceTime.
