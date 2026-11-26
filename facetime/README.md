# Call the Chert avatar on FaceTime

Run the named agent and Chert connector on the same Linux host as the working GPT-Live preview. The connector takes inbound calls on an assigned Chert line. The bridge waits for the FaceTime participant before it starts a Brick Room session on the private preview. A standby room does not occupy the GPU.

The agent subscribes to the caller's audio and, when the line provides it, camera track. It republishes them into the private avatar room. It sends the avatar's audio and video back to FaceTime under one participant identity. This extra LiveKit hop lets the browser and FaceTime share the same GPU admission and speech pipeline. A browser session using the GPU can prevent a FaceTime call from starting.

Once room metadata signals `chert_bridge_ready: true`, the agent requests a short greeting through the authenticated preview session. This prevents FaceTime's ten-second silent-audio timeout while keeping standby rooms silent and browser sessions caller-first. Deploy the FaceTime files and `gpt_live_trial` together; an older backend does not have the greeting endpoint. Repeated readiness signals and HTTP retries do not produce additional greetings.

## Set up

1. Start the [GPT-Live preview](../gpt_live_trial/README.md) on loopback port 18837. Verify that the provider reports GPT-Live and that the avatar worker is ready. Use a publicly reachable LiveKit Cloud URL.
2. Install Node 24 and run `npm ci` in this directory. On Linux, create a private `.env.local` with mode 600. Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` to the existing preview's LiveKit project. Keep the voice provider key in the preview backend.
3. Run `npx --no-install chert login --device` and approve the intended project. Start the worker with `node --env-file=.env.local agent.mjs start`. Keep the process running.
4. In a second terminal here, run `npx --no-install chert doctor --agent chert-avatar-live --env-file .env.local`. The named agent should join the test room. The starter's OpenAI environment check is separate from this worker's backend connection.
5. Run `npx --no-install chert dev --agent chert-avatar-live --env-file .env.local`. Wait for the connected and warm-room messages. Leave the process running for the test window. If the line is already in use, resolve ownership first.
6. On a shared trial, text the join code shown in the project dashboard from the testing iPhone. After the confirmation, start a FaceTime Video call to the assigned number and speak. Confirm both directions, video motion, a follow-up question and clean hangup. The caller's camera is a separately provisioned line capability.

To release the line, stop `chert dev` with Ctrl+C and wait for its release confirmation, then stop the named worker. The browser preview runs independently. `chert logout` revokes the connector login. Keep Chert's private state out of Git.

## Fill the phone screen

Chert Internal Testing uses the per-project `video_framing: "fill"` setting. It was applied in production on November 26, 2026 UTC and verified in `facetime_shared_line_settings`. [The deployment SQL](config/internal-testing-fill.sql) records the exact project and line, skips the write when already configured, and checks the saved result. Run it explicitly through an authorized administrative SQL connection to the database named in the file; it is not an automatic schema migration or an agent startup step.

New calls pick up this setting. Fill preserves the image's proportions and crops the edges to cover the viewport; the native bridge uses a centered 9:16 viewport once FaceTime portrait orientation is confirmed. FaceTime's controls remain part of its own interface. To return to Fit, use `facetime_update_customer_line` with the same project and line and `{"video_framing":"fit"}`. Keep this setting scoped to the project, since the line is shared.

Fill changes composition, not source resolution. The inspected live avatar session rendered at 512 × 384. A 9:16 center crop uses about 216 × 384 source pixels before enlargement, so a phone-filling image can look softer than the smaller landscape view. The forwarding agent preserves received frame dimensions; increasing only the final camera canvas cannot recover missing source detail. Higher render resolution, a suitable reference image, and measured encoding improvements should be evaluated against GPU memory, response latency, and quality on a real handset before deployment.

The current geometry configuration supports `AVATAR_RESOLUTION=512x384` and `640x480`. The latter renders 56% more pixels and is a candidate for a quality benchmark, not a verified production improvement. Geometry is process-wide and requires a coordinated backend/renderer restart while idle. 720p, 1080p, and native portrait rendering require further implementation and validation.

[FaceTime quickstart](https://docs.trychert.com/facetime/quickstart) · [Connect a named agent](https://docs.trychert.com/facetime-cli/quickstart) · [Media capabilities](https://docs.trychert.com/facetime/audio-video)
