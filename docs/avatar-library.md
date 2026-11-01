# Demo avatar library

The default catalog contains exactly the ten images in
[`realtime/demo-avatars/`](../realtime/demo-avatars/README.md). Its manifest fixes
each image checksum, display name and ElevenLabs stock voice. No private import
or prepared voice cache is required.

Choose a portrait under **Settings → Avatar & voice** before calling. The stock
voice is selected automatically; the voice menu can choose another bundled stock
voice. These are creative casting choices, not cloned voices of the people shown.
The public catalog does not publish private voice previews or assignments.

Private research imports remain supported through
`python -m realtime.scripts.prepare_avatar_library /private/manifest.json` and
`AVATAR_CATALOG=research`. The importer requires 4:3 images and saves assets and
profiles under ignored data directories. That mode also enables the old benchmark
references and must never be used with the public gateway.
