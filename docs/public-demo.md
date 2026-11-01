# Publish on an HTTPS domain

GitHub Pages serves static pages; it cannot run the Python agent or CUDA worker.
Host the app and restricted gateway on the GPU machine, and embed that HTTPS
origin in your website or link to it directly. LiveKit Cloud transports media.

Complete [local setup](recreate-demo.md) and keep `AVATAR_CATALOG=demo`.
The public gateway refuses research mode. Keep the app bound to loopback and
use the existing private profile; never put its access code into the website.

## Backend

Start the gateway on the GPU host, replacing the example origin:

```bash
.venv-realtime/bin/python -m realtime.public_gateway --origin https://demo.example.com
```

Run it as a persistent service in deployment. It listens on loopback port 18820
and reads the private profile at `.cache/realtime-host/profile.json` by default.
Use `--directory` if the app has a different profile location.

Install Caddy from its [official instructions](https://caddyserver.com/docs/install),
set `CHERT_DEMO_DOMAIN=demo.example.com` in its service environment, and load
[`infra/Caddyfile.demo`](../infra/Caddyfile.demo). Point a DNS **A** record for
`demo` to the GPU host's public IPv4 address, and allow inbound TCP 80/443 in the
cloud firewall. Caddy obtains and renews the HTTPS certificate automatically.
Do not expose ports 18820, the private app port, the CUDA worker, or Qwen directly.

The gateway:

- Exposes only ten bundled portraits and stock voices; blocks voice-casting/admin routes.
- Keeps the application access code server-side and strips upstream cookies.
- Rejects unexpected hosts, cross-origin API requests and unsupported routes.
- Requires a session-specific key for call control, enforced by the application.
- Limits starts to three per minute and thirty per hour globally; one GPU call at a time.
- Keeps the ten-minute call limit and uses polling for bounded HTTP responses.

These limits control a small public demo, not production abuse prevention or
multi-user scale. Rate counters reset when the gateway restarts. For a sustained
public launch, add visitor identity, a queue and provider spending limits.

## Website

An iframe must explicitly allow microphone and camera:

```html
<iframe src="https://demo.example.com"
  title="Chert video call"
  allow="camera; microphone; autoplay; fullscreen"
  allowfullscreen referrerpolicy="no-referrer"></iframe>
```

Both pages must use HTTPS. Update `frame-ancestors` in `realtime/public_gateway.py`
for your own website; the included deployment allows dunlichan.com. The iframe
URL contains no secrets. A direct link is a useful fallback if a browser blocks
device permissions in an embedded page.

Before sharing, verify `/api/avatars` contains only `call01` through `call10`,
private avatar URLs return 404, no provider key is present in browser assets,
and a complete browser call works with interruption. Check the Cloud dashboard's
usage while the demo is public. Shut down the cloud instance separately when done.
