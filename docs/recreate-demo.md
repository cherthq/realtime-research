# Recreate the interactive avatar demo

This guide starts from a clean checkout and ends with a private browser call:
microphone → OpenAI voice → AvatarForever on a GPU → synchronized video/audio.
The browser shows your camera beside the generated avatar at native 640×480.
LiveKit is self-hosted; a LiveKit Cloud account is not required.

The tested GPU configuration is **Ubuntu x86_64, H100 80 GB, Python 3.11,
PyTorch 2.7.1/CUDA 12.8**. This is an experimental single-call demo. See
[qualification](../realtime/QUALIFICATION.md) for measured performance and limits.
Do not infer GPU capacity from steady-state memory: model loading peaked near
74 GB of allocated GPU memory in the recorded run.

## What you need

| Item | Where it comes from |
| --- | --- |
| Application and setup code | This repository, at the revision provided by the maintainer |
| Three reference PNGs and six replay WAVs | Separate `chert-demo-assets.zip` from the maintainer; Git does not contain them |
| AvatarForever and Gemma weights | Pinned download script below; approximately 70 GB, plus download/cache space |
| Hugging Face account/token | Your account, with access to the gated Gemma repository |
| OpenAI API key | Your account, kept privately on the server |
| Git, Python 3.11 with venv, FFmpeg, Node.js 22.12+ and npm | Installed on the GPU host; verify versions before setup |
| SSH access and a browser with microphone permission | Your laptop; localhost browser URL provides a secure context for device access |

Use at least 150 GB of available storage for weights, environments and temporary
downloads. On Lambda, attach a filesystem in the instance's region at launch if
you want to retain weights across instance termination. Set a private SSH-only
firewall policy; this guide does not expose the web/media ports to the internet.

The asset ZIP is **not** a full instance backup. Never publish or share a backup
containing `.env`, `.cache/realtime-host/profile.json`, access codes or SSH keys.

## 1. Clone and install the application environment

On the GPU host:

```bash
git clone https://github.com/cherthq/realtime-research.git
cd realtime-research
# If the handoff is on a branch or tag, check out that revision now.
git rev-parse HEAD
python3.11 --version
node --version
npm --version
ffmpeg -version
nvidia-smi

python3.11 -m venv .venv-realtime
.venv-realtime/bin/python -m pip install -r realtime/requirements.txt -c realtime/constraints-linux-py311.txt
.venv-realtime/bin/python -m pip check
```

The CPU application and GPU model have separate Python environments. Do not
install the model dependencies into `.venv-realtime`.

Build the frontend:

```bash
cd realtime/web
npm ci
npm run build
cd ../..
```

The frontend build is ignored by Git and must be generated after cloning. The
Python server serves this build; a separate Vite development server is unnecessary.

## 2. Import the demo media

Ask the maintainer for `chert-demo-assets.zip` and its SHA-256 digest. Copy it to
the host, then compare its digest with the one supplied separately:

```bash
sha256sum /path/to/chert-demo-assets.zip
.venv-realtime/bin/python -m realtime.scripts.demo_assets import --archive /path/to/chert-demo-assets.zip
.venv-realtime/bin/python -m realtime.scripts.demo_assets check
```

Import verifies the contents and refuses to overwrite an existing destination.
The default destination is `data/eval/facetime-zero-shot-v1/`. For an existing
valid installation, use `check` rather than importing over it. Use `--assets
/absolute/new/path` for another location and set `ASSETS_DIR` in the server's
`.env` to that same directory.

This pack supplies the reference images and the **Test reply** fixtures. Live
conversation generates new voice audio. The optional 18-case v2 benchmark pack
and previous experiment results are not required to start the interactive demo.
This minimal handoff omits the manifests needed by the offline baseline runners;
request the full frozen benchmark pack separately if reproducing those runs.
New portraits or regenerated speech are different evaluation inputs; see the
[baseline protocol](zero-shot-baseline.md) if creating a new set.

## 3. Install the GPU environment and retrieve weights

```bash
bash infra/setup-avatarforever.sh
.venv-avatarforever/bin/python -m pip check
.venv-avatarforever/bin/hf auth login
```

First accept/request access to
[google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
with the account associated with that token. Enter the token interactively; do
not put it in a command, commit, or shared document.

Choose a checkpoint root on your storage volume, replacing this example:

```bash
export CHERT_CHECKPOINTS=/lambda/nfs/YOUR_FILESYSTEM/checkpoints
.venv-avatarforever/bin/python scripts/download_baseline_weights.py avatarforever --root "$CHERT_CHECKPOINTS"
```

The root must contain:

```text
checkpoints/
  avatarforever-revisions.json
  AvatarForever/avatarforever-ltx-2.3-22b.safetensors
  gemma-3-12b-it-qat-q4_0-unquantized/...
```

The source, AvatarForever weights and Gemma revision are pinned. The downloader
can reuse existing downloads. Keep `avatarforever-revisions.json` with the weights.
The root is the directory **above** the two model directories, not a checkpoint
file. Review [component terms](../realtime/QUALIFICATION.md#component-terms).

## 4. Configure the private server

Copy the template once, then edit it privately:

```bash
cp -n realtime/.env.example realtime/.env
chmod 600 realtime/.env
```

Fill in your key and absolute paths. Use the actual repository and checkpoint
locations; `.env` paths below are examples, not shell-variable expressions:

```dotenv
OPENAI_API_KEY=YOUR_PRIVATE_KEY
OPENAI_REALTIME_MODEL=gpt-realtime
OPENAI_CASTING_MODEL=gpt-5.6-terra
AVATARFOREVER_ENABLED=true
AVATARFOREVER_PYTHON=/home/ubuntu/realtime-research/.venv-avatarforever/bin/python
AVATARFOREVER_WEIGHTS=/lambda/nfs/YOUR_FILESYSTEM/checkpoints
AVATARFOREVER_COMPILE_BLOCKS=true
GPU_LOAD_TIMEOUT_SECONDS=900
GPU_CHUNK_TIMEOUT_SECONDS=60
SESSION_LIMIT_SECONDS=600
```

`OPENAI_CASTING_MODEL` must be accessible to your account and support image input
and structured Responses output. If this configured model is unavailable, select
an appropriate accessible model, or use an explicitly chosen prepared voice.
The service does not silently substitute a voice or avatar renderer.

The SSH profile in the next step supplies fresh LiveKit credentials and the
browser access code, overriding the template's local development credentials.
Do not copy somebody else's signing secrets or access code.

Prepare the voice previews and match them to the three fictional characters:

```bash
.venv-realtime/bin/python -m realtime.scripts.prepare_voices --voices cedar marin alloy --cast
```

This makes paid API requests. Listen to the previews in Settings before using the
demo. To avoid automatic casting, prepare the voices without `--cast`, then choose
a voice explicitly in Settings before starting a call. `--freeze-pack` is an
optional benchmark operation, not a requirement for live conversation.

## 5. Choose one private browser route

Pick **A** first. Use **B** if Windows localhost/media behavior requires the WSL
bridge. Choose before generating a profile: `prepare` intentionally refuses to
overwrite an existing one. If changing routes, stop the old service and use a
different `--directory`, passing it consistently to `check`, `serve` and `warm`.

### A. Direct SSH forwarding

On the GPU host:

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_host prepare
.venv-realtime/bin/python -m realtime.scripts.demo_host check
.venv-realtime/bin/python -m realtime.scripts.demo_host serve
```

Keep this terminal open, or run `serve` inside tmux. For a user service, use the
[systemd instructions](../realtime/README.md#private-lambda-demo-over-ssh).
The server creates private `.cache/realtime-host/profile.json` and `livekit.yaml`.

On your laptop, in a separate terminal:

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:8080:127.0.0.1:8080 -L 127.0.0.1:7880:127.0.0.1:7880 -L 127.0.0.1:7881:127.0.0.1:7881 ubuntu@YOUR_GPU_IP
```

Add `-i /path/to/your/private/key` if needed. Verify the SSH host key normally.
All three forwarded ports are required: API, signaling, and TCP media.
Open **http://127.0.0.1:8080/** after warm-up in step 6.

### B. Windows browser through a private WSL bridge

Find your WSL Ubuntu IPv4 address (`hostname -I` inside WSL). Use that private
address consistently below; it can change when WSL restarts. Do not use the
Lambda public IP in `--client-address`.

On the GPU host, for a fresh profile:

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_host prepare --client-address YOUR_WSL_IPV4 --api-port 19881 --signal-port 19882 --media-port 19883
.venv-realtime/bin/python -m realtime.scripts.demo_host check
.venv-realtime/bin/python -m realtime.scripts.demo_host serve
```

Inside WSL, keep this SSH forwarding process open:

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L YOUR_WSL_IPV4:19881:127.0.0.1:19881 -L YOUR_WSL_IPV4:19882:127.0.0.1:19882 -L YOUR_WSL_IPV4:19883:127.0.0.1:19883 ubuntu@YOUR_GPU_IP
```

Use your private SSH key in WSL with appropriate permissions. On Windows, clone
the same revision and run from its root in PowerShell:

```powershell
py -3.11 -m venv .venv-realtime
.venv-realtime/Scripts/python.exe -m pip install -r realtime/requirements.txt
.venv-realtime/Scripts/python.exe -m realtime.scripts.windows_demo --upstream YOUR_WSL_IPV4:19881 --port 18810
```

Open **http://127.0.0.1:18810/** in Windows Chrome or Edge. This command starts
only the local HTTP bridge. Keep the bridge and WSL SSH process open. The API,
voice connection and GPU model run on the server. No OpenAI key belongs on the
laptop for this route.

## 6. Warm up and test an actual call

With `serve` running, use a second SSH terminal on the GPU host:

```bash
cd /path/to/realtime-research
.venv-realtime/bin/python -m realtime.scripts.demo_host warm --directory .cache/realtime-host --timeout 945
```

The recorded H100 first load/compile took about 3½ minutes. Wait for `loaded: true`.
Warm-up refuses to interrupt an active call and releases its temporary call slot.
This checks model preparation; it does not establish browser media delivery.

Read `access_code` privately from `.cache/realtime-host/profile.json`, enter it
in the browser, then:

1. Allow microphone/camera access. The camera preview stays on your laptop.
2. In Settings choose **AvatarForever** and **Live voice**, then a character.
3. Start the call, ask a question, and confirm both audible speech and generated
   movement. The diagnostic renderer is a separate option and is not GPU output.
4. Interrupt a reply by speaking or pressing Interrupt. End the call and start
   another to confirm the session slot is released.

Voice replies have no application token or word-count cap; the provider's own
maximum remains. Incomplete replies no longer terminate the call. There is a
separate 10-minute demo call duration by default (`SESSION_LIMIT_SECONDS`).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing images or replay audio | Run `demo_assets check`; verify `ASSETS_DIR` points to the imported pack |
| Voice profile unavailable | Run `prepare_voices`; choose an explicit prepared voice if casting is unavailable |
| Hugging Face 401/403 | Accept Gemma access with the same account used by `hf auth login` |
| Model disabled / missing weights | Check the private `.env`, the checkpoint root and both model directories; restart the server after edits |
| First call waits for minutes | Run `warm` first; compilation is separate from response latency |
| Page loads but no media | Forward all three ports; check browser permission and the SSH/WSL address, then inspect `.cache/realtime-host/` logs |
| Port already in use | Stop the specific old launcher, or choose matching alternate ports; do not kill unrelated services |
| Reply shortened or provider failure | Session reports in `runs/realtime/sessions/` include `voice_diagnostics`; inspect the status/reason code |

GPU OOM, disabled model and provider errors are reported explicitly. Do not use
lower-quality fallback output as evidence that AvatarForever is working.

## Checks before publishing or handing off

```bash
.venv-realtime/bin/python -m pip install ruff==0.14.1
.venv-realtime/bin/python -m ruff check realtime scripts --select E9,F
.venv-realtime/bin/python -m pytest realtime/tests -q
.venv-realtime/bin/python scripts/check_public_tree.py
cd realtime/web
npm ci
npm run build
cd ../..
```

CPU tests generate temporary fixtures and isolate deployment settings. They need
no private media, API key or GPU. Three real LiveKit integration tests remain
opt-in with `RUN_LIVEKIT_TESTS=1` and a local LiveKit server. Browser media and a
fresh GPU installation require their own checks; a passing build is not proof
of live inference. GitHub Actions runs the CPU tests and frontend build only.

To create the **asset-only** handoff on a machine with the prepared v1 inputs:

```bash
.venv-realtime/bin/python -m realtime.scripts.demo_assets export --output data/handoff/chert-demo-assets.zip
```

The command prints the SHA-256 digest and exports only the nine configured media
files plus provenance/checksums. It excludes `.env`, voice assignments, access
profiles, recordings and weights. Share the ZIP and digest separately from Git.
The source tree does not include this generated archive.

Review `git status --short` and the diff before committing. Include `realtime/`,
its referenced configs, setup scripts, documentation and package lockfile. Do
not force-add ignored folders. `check_public_tree.py` checks tracked and
non-ignored files, including accidentally tracked credentials/media patterns;
it reports filenames, never matched credential values.

## Stopping and resuming

End the browser call. Stop `serve` with Ctrl+C, or stop its user service, and close
the laptop tunnel/bridge. These actions do not terminate a Lambda instance.
Terminate compute through the Lambda console when finished. Save repository
changes and required results first; retain the attached filesystem if keeping
the weights. On a new instance in the same region, attach that filesystem, clone
the desired revision, recreate the environments and profile, and warm up again.

The implementation can be recreated from code, assets and model pins. Exact
stochastic voice responses, pixel output and latency are not guaranteed across
provider updates, hardware changes or compiler versions.
