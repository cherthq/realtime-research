# Lambda / WSL setup

Use Linux for model execution. Locally: WSL Ubuntu, `/mnt/d/realtime-research`.
On Lambda: clone this workspace into `~/realtime-research`, with model sources,
data and weights on a persistent volume when available. This foundation does not
provision or bill cloud resources.

1. Add the host/key using `ssh_config.example`, then `ssh chert-lambda`.
2. Install Git and FFmpeg on the GPU host if absent.
3. Transfer the current workspace (including uncommitted foundation files):

   ```bash
   # Run in WSL, with rsync installed and the SSH host configured in WSL.
   rsync -av --exclude=.git --exclude=vendor --exclude=data --exclude=checkpoints --exclude=runs --exclude='.env*' --exclude='*.pem' --exclude='*.key' /mnt/d/realtime-research/ chert-lambda:~/realtime-research/
   ```

4. Run `python3 scripts/workspace.py sources all` on the host.
5. Create a separate Python environment per model, using the exact Python,
   PyTorch/CUDA and attention versions in its pinned upstream README. These
   versions differ across models. Do not install all requirements into one env.
6. In that environment run `python scripts/workspace.py doctor --require-gpu`.
7. Follow each pinned README's checkpoint instructions. Download into the paths
   its inference script expects; canonical persistent storage can live under
   `checkpoints/` with explicit symlinks inside the vendor repo. Record HF revision
   and downloaded filenames in the run metadata before using the checkpoint.
8. Run the unmodified upstream inference example before writing any adapter.

Links: [Live Avatar](https://github.com/Alibaba-Quark/LiveAvatar),
[FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk),
[HunyuanVideo-Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar).

GPU RAM requirements vary with precision, frame count and resolution. Source
checkout checks do not establish inference capacity or fine-tuning capacity.
Capture `nvidia-smi`, Python/PyTorch versions and `pip freeze` with each run.

Training readiness is a later gate: inspect available training code, loss and
checkpoint formats before selecting a LoRA/full fine-tuning implementation.
