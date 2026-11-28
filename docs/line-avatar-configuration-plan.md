# Standalone avatar configuration

## Scope

This is a separate application in `avatar-config/`, not part of the Chert
dashboard. Do not modify the Chert dashboard, authentication, shared-line control
plane, billing, or native bridge as part of this project. Reuse the existing
Dun Li Lambda avatar runtime; do not provision another GPU or duplicate backend.

The user approved the standalone UI, authorized read-only runtime inspection and
a local preview connection, and on November 28 approved committing the reviewed
work remotely and proceeding to Phase 2.

## Reviewed Phase 1

- Two fields: prompt and driving image. No name field or dashboard navigation.
- Browser-local atomic saves with revision conflict detection.
- JPEG/PNG validation, crop controls, normalized 640×480 still preview.
- Current/previous deployment cards with image and expandable full prompt.
- Links to the existing web call and native FaceTime video-call handler.
- Private deployment snapshots and connection credentials stay in ignored `local/`.
- The real image was verified against the runtime's served image byte-for-byte.
- The browser call page was verified Ready without starting a call.
- Local server on port 4198; existing backend connection on port 18837.

The original proposal to build inside the Chert dashboard is superseded. Its
removed worktree is not an implementation target. Historical local planning is
retained privately rather than published with this independent application.

## Phase 2: configuration over the existing runtime

1. Capture the exact deployed serving package, launch configuration, current
   prompt/image pair and rollback procedure. Preserve the known difference between
   repository greeting behavior and the deployed greeting; do not silently change it.
2. Add immutable configuration versions and private image storage, with atomic
   activation and optimistic concurrency. Bootstrap from the exact live pair.
3. Bind one configuration version to each new call. Keep active calls untouched,
   including through publication and rollback. A lost start response must recover
   the original call/version rather than allocate another session.
4. Pass the pinned prompt and image into the existing GPT-Live/AvatarForever
   worker. Reuse loaded weights and the existing single-call GPU admission lock.
5. Connect the standalone UI to authenticated configuration APIs. Keep the two-field
   editor; show real current/previous deployments and a clear publish action.
6. Validate the actual private trained renderer's image reset path before enabling
   arbitrary uploads. Test A → B → A, failure recovery, stale writes, active-call
   isolation, and rollback. Do not equate a valid image with renderer readiness.
7. Deploy only the necessary integration to the existing runtime while idle, with
   a captured rollback. Host the standalone dashboard separately; no Chert UI or
   shared database changes are implied by approval for this project.

## Constraints

Private prompts and images must not enter the public avatar catalog or room
metadata. Public clients must not choose arbitrary private versions. Authenticated
configuration writes must reject malformed images, oversized data, stale revision
updates, and arbitrary filesystem paths. Store image assets by immutable hash and
verify them on use. Keep raw credentials and private assets out of Git and logs.

Preserve GPT-Live 1.1, Marin, AvatarForever, 640×480 output, Fit/Fill and existing
call-start recovery. Source-image dimensions and output-video geometry are separate.
A new call must either use one complete version or fail clearly, never combine an
old image with a new prompt. Deployment rollback must restore the whole pair.

## Integration entry points

- `gpt_live_trial/agent.py`: per-call instructions.
- `gpt_live_trial/preview.py`: GPT-Live worker routing and shared backend lifecycle.
- `realtime/app.py`: session admission, session snapshots, start/recovery routes.
- `realtime/start_receipts.py`: idempotent call-start recovery.
- `realtime/worker.py`, `realtime/gpu_pool.py`: reference-image acquisition/reset.
- `realtime/public_gateway.py`: public/private boundary.
- Private Lambda serving extensions and trained renderer: inspect the actual
  implementation; public source alone does not describe the deployed path.

The running deployment's private paths and checksums are recorded in ignored
local verification artifacts. No runtime write has been performed as of the
Phase 1 commit. Phase 2 implementation follows this checkpoint.

## Phase 2 implementation checkpoint — November 28

Implemented locally on `codex/avatar-runtime-configuration`:

- Private SQLite versions, hash-verified images, atomic publish/restore and CAS.
- Opt-in trial-only authenticated API; a separate narrow admin token.
- Per-call catalog/instructions binding with existing admission/start receipts.
- Warm-renderer preparation, idle-only image switching and acceptance gate.
- Dashboard Publish/Edit/Restore, real history reads and a same-origin local proxy.
- Local integration browser verification of publish and whole-version restore.

The actual private package has additional greeting and cleanup changes absent from
main. A six-file candidate was staged against captured live source, preserving
those changes; original bytes and a hash manifest are kept in ignored local files.
The user approved the next idle window. The six-file integration is now installed
and the existing supervised backend/companion pair is healthy after an idle
restart. The original prompt, image, greeting and output geometry were preserved.
Next: run the trained renderer acceptance sequence, then enable image changes.
Separate Vercel hosting and authenticated Cloudflare connectivity are now installed
and verified; see the hosted dashboard checkpoint below.

## Initial live rollout verification

- Captured source/launch backups before installing; all six installed hashes match.
- Existing supervisor retained the avatar-first startup order; both services ready.
- Current prompt matches the recorded live prompt exactly; image bytes/hash match.
- Output remains 640×480, 25 native / 24 publish fps, using GPT-Live 1.1.
- Configuration API: unauthorized 401, stale update 409, image-change gate 409.
- Validation left the imported baseline at revision 1; no persona was published.
- Standalone dashboard now reads live configuration; older recorded cards remain.
- Web preview reloaded to Ready. No test call was placed.
- 46 focused runtime tests and 16 dashboard tests pass; browser console is clean.

Image switching stays disabled in both the backend and dashboard publication
path pending actual private-renderer acceptance.

## Hosted dashboard checkpoint — November 28

- Production: https://realtime-avatar-dashboard.vercel.app/.
- Independent Vercel project in Gary's personal workspace; all deployments require
  Vercel Authentication. Anonymous page and API requests redirect to sign-in.
- Server-only configuration proxy uses the narrow runtime credential. The existing
  Cloudflare connector routes only `/api/avatar-config` to the existing backend.
- Existing workspace route and avatar service remain healthy. The connector reload
  needed its full shutdown window; final checks use local health plus external
  verification, rather than the host's unreliable public self-check.
- Authenticated hosted configuration and historical metadata return 200; all three
  displayed driving images decode correctly. Browser console has no errors.
- Runtime remains idle at revision 1 with its original prompt, greeting and image.
  No test call or persona publication was performed.
- 20 dashboard tests and syntax checks pass, including private build-artifact checks.
- Hosted dashboard is independent of the laptop. The explicitly local preview
  button still needs its local connection; native FaceTime uses the device handler.
