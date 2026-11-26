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

[FaceTime quickstart](https://docs.trychert.com/facetime/quickstart) · [Connect a named agent](https://docs.trychert.com/facetime-cli/quickstart) · [Media capabilities](https://docs.trychert.com/facetime/audio-video)
