# Private runtime configuration

This opt-in layer serves one existing GPT-Live deployment. It does not create a
GPU backend, change the public avatar catalog, or integrate the Chert dashboard.
Without `runs/avatar-config/runtime.json`, existing call behavior is unchanged.

## Installation contract

The private serving package differs from repository main. Apply the small
`agent.py`/`preview.py` integration changes to the **captured deployed files**;
do not replace the live package with this repository's copies. In particular,
preserve the existing greeting endpoint/implementation and private cleanup fixes.
The session report must use `session.worker.catalog` to hash the pinned image.

Install this Python package beside the existing `realtime` package. Dependencies
(FastAPI, Pillow, NumPy) are already part of the runtime. Configure a private
mode-0600 `runs/avatar-config/runtime.json`:

```json
{
  "avatar_id": "<existing private avatar ID>",
  "admin_token": "<separate random configuration token, at least 32 characters>",
  "image_switch_verified": false
}
```

The deployment and avatar ID are selected by the host operator, never supplied
by the browser as an authorization decision. The configuration token grants
access only to this deployment. It is not the preview token or Jupyter token.
The initial version copies the existing instructions and image byte-for-byte.
No prior deployment dates or unverified historical versions are fabricated.

The trial port exposes authenticated `GET` and `POST /api/avatar-config` through
a separate ASGI handler, independent of the preview's access token and static
mount. This endpoint is absent from the baseline port and public gateway. Browser
Origins are rejected; a server-side dashboard proxy holds the token. Do not add
these routes to public gateway allowlists or expose this loopback endpoint.

POST accepts `prompt`, `image` (PNG/JPEG data URL, or null to keep the current
image), and `expectedRevision`. Restore accepts only `restoreId` and
`expectedRevision`. Missing/invalid tokens return 401; stale revisions return 409.
A disconnected publisher may still commit; refresh before retrying. Reusing an old
revision cannot activate twice. The dashboard never displays the token.

## Call and publication behavior

SQLite atomically activates an immutable prompt/image pair; images are private,
hash-addressed files. Each new trial call pins its version before constructing
its worker. A worker-local catalog and instance-local voice instructions prevent
changes from affecting an active call or other catalog consumers. Existing
start receipts recover that same call after publication. The per-call version ID
stays in process memory and is not added to public snapshots or room metadata.

Prompt-only publications can happen during a call. A different image requires
an idle admission lock and an already loaded renderer. Preparation opens the
candidate, renders a frame, then opens/renders the previous image again using the
same GPU process. Activation follows successful preparation. Failure leaves the
active version unchanged. Failed GPU IPC is retired by the existing runtime;
the configuration layer never cold-loads replacement weights.

`image_switch_verified` must remain false until the **actual private trained
renderer** passes A → B → A with captured frames, unchanged model process, clean
per-call reset, failed-image recovery, and a final known-good baseline. Unit tests
and image decoding are not substitutes for that acceptance test. Enabling this
flag requires restarting the existing service. Keep it false for the initial
installation; prompt changes and restore of the same image remain available.

Existing FaceTime automatic greeting is intentionally unchanged. If its wording
contains the previous persona's name, changing the prompt does not rewrite that
separate greeting. Changing the greeting remains outside this feature.

## Maintenance and rollback

Before installation, capture exact source bytes and hashes of all touched live
files, the baseline image/prompt hashes, supervisor pair state and the original
launch environment privately. Never print or commit the environment or secrets.
Stage and compile the candidate against those captures. Abort if any live source
hash changes between capture and installation.

Coordinate an idle window before restarting. Recheck both call admission and GPU
lease immediately before the restart. Use the existing supervisor's backend
restart path, preserving its avatar-first / companion-second startup order. Do
not stop the supervisor itself: its shutdown handler stops both services.

For rollback: restore all captured source files byte-for-byte, remove/disable the
new opt-in runtime configuration, then restart only the supervised backend during
an idle window. Keep the version store and image assets for recovery; do not delete
published data. The old package does not import the added runtime layer. Verify
health, original image hash, 640×480 output, model/voice, idle GPU state and the
existing preview Ready screen. Test calls require their own coordinated test step.

## Verification

Run `python -m pytest avatar_config_runtime/tests gpt_live_trial/test_preview.py
 gpt_live_trial/test_agent.py`. Tests use generated images and mock media workers;
they do not contact Lambda, OpenAI or LiveKit. Integration coverage includes both
call ports, independent admin authentication, actual call constructor binding,
active-call isolation, start-response recovery across publication, restore,
concurrent/stale updates, failed preparation and disconnected publishers.
