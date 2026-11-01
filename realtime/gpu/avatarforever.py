"""Experimental bounded streaming adaptation of the pinned AvatarForever source.

Uses upstream's model, relative-position selection, per-chunk feature cache and
Euler steps. Retains a sink chunk and the most recent chunk. Audio is encoded
from arrived PCM with bounded left context and 280 ms of arrived lookahead;
VAE decoding retains four latents.
These streaming boundaries differ from upstream full-file audio encoding and
full-clip decoding. GPU equivalence/quality/performance are NOT yet established.
Upstream remains under its own LTX-2 Community License in vendor/avatarforever.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import math
import resource
import threading
import time

import numpy as np
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import VideoConditionByLatentIndex
from ltx_core.model.audio_vae import encode_audio
from ltx_core.model.transformer.ar_feature_cache import ARFeatureCache
from ltx_core.model.transformer.attention import (
    Attention,
    AttentionFunction,
    PytorchAttention,
    XFormersAttention,
    memory_efficient_attention,
)
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.video_vae import decode_video
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.a2vid_distilled import A2VidDistilledPipeline
from ltx_pipelines.utils.autoregressive import _select_ar_chunk_state
from ltx_pipelines.utils.helpers import (
    apply_video_channel_condition_to_state,
    noise_audio_state,
    noise_video_state,
    post_process_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import load_image_conditioning

from realtime.contracts import AudioChunk, FRAME_SAMPLES, HEIGHT, SAMPLE_RATE, WIDTH
from realtime.gpu.prompt_cache import FixedPromptCache
from realtime.listening import ListeningControl, ONSET_STRENGTH, QUIET_PROMPT
from realtime.streaming import GeneratedBatch, chunk_frames, LOOKAHEAD_SAMPLES

PROMPT = (
    "A person speaking naturally to the camera in an upper-body video call. "
    "The lips and jaw articulate the supplied speech clearly, with natural "
    "blinking, subtle head motion and conversational expressions. "
    "Stable identity and background, continuous motion in a single shot."
)
# The opt-in listening profile changes fixed text contexts without resetting
# visual history. It is experimental, not a guarantee of closed lips in silence.


class GenerationCancelled(Exception):
    pass


class AvatarForeverBackend:
    """Single-owner GPU object; bounded arrived PCM, no network/file audio access."""

    def __init__(
        self,
        weights: Path,
        image: Path,
        *,
        compile_blocks: bool = False,
        cache_prompt: bool = False,
        listening_profile: str = "off",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("AvatarForever requires a CUDA GPU")
        self.listening = ListeningControl(listening_profile)
        self.warm_profile = None
        self.pipeline = A2VidDistilledPipeline(
            distilled_checkpoint_path=str(
                weights / "AvatarForever/avatarforever-ltx-2.3-22b.safetensors"
            ),
            spatial_upsampler_path=None,
            gemma_root=str(weights / "gemma-3-12b-it-qat-q4_0-unquantized"),
            loras=[],
        )
        self.device, self.dtype = self.pipeline.device, self.pipeline.dtype
        extra = self.pipeline._channel_condition_extra_config(
            first_frame_channel_condition_latent=None,
            stage2_first_frame_channel_condition_latent=None,
            init="zero",
            mode="gated",
            fallback_channel_dim=self.pipeline.pipeline_components.video_latent_channels,
        )
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=self.dtype, cache_enabled=False),
        ):
            self.modules = self.pipeline._get_fast_modules(
                stage_mode="one-stage", transformer_extra_config=extra
            )
            self.video_context, self.audio_context = (
                self.pipeline._encode_prompt_context(
                    prompt=PROMPT,
                    images=[],
                    enhance_prompt=False,
                    text_encoder=self.modules.text_encoder,
                    embeddings_processor=self.modules.embeddings_processor,
                )
            )
            self.prompt_contexts = {"default": (self.video_context, self.audio_context)}
            if listening_profile != "off":
                quiet = self.pipeline._encode_prompt_context(
                    prompt=QUIET_PROMPT,
                    images=[],
                    enhance_prompt=False,
                    text_encoder=self.modules.text_encoder,
                    embeddings_processor=self.modules.embeddings_processor,
                )
                speaking = (self.video_context, self.audio_context)
                onset = tuple(
                    torch.lerp(q.float(), s.float(), ONSET_STRENGTH).to(q.dtype)
                    for q, s in zip(quiet, speaking)
                )
                self.prompt_contexts = {
                    "quiet": quiet,
                    "onset": onset,
                    "speaking": speaking,
                }
        # Text is fixed for this session. Its large encoder need not occupy GPU
        # memory during video generation; the diffusion/VAE modules stay resident.
        self.modules.text_encoder.to("cpu")
        self.modules.embeddings_processor.to("cpu")
        torch.cuda.empty_cache()
        self.sigmas = torch.tensor(
            [1.0, 0.98125, 0.909375, 0.421875, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        self.compiled_blocks = 0
        self.compile_warmup_seconds = None
        self.last_stage_ms = {}
        self.stage_events = []
        self.prompt_cache = None
        self.open_session(image)
        if cache_prompt:
            self.prompt_cache = FixedPromptCache(
                [
                    module
                    for module in self.modules.transformer.modules()
                    if isinstance(module, BasicAVTransformerBlock)
                ],
                len(self.sigmas) - 1,
                self.listening.profiles,
            )
            self._warm_history_shapes()
            self.prompt_cache.freeze()
        if compile_blocks:
            self._compile_and_warm()

    def _compile_and_warm(self):
        """Compile repeated blocks without changing weights, attention or steps.

        Resolve upstream's callable enum to its exact implementation because
        Dynamo 2.7 cannot trace enum calls. Share stateless callables so object-ID
        guards do not cause a separate graph for every transformer layer.
        """
        targets = {
            AttentionFunction.PYTORCH: PytorchAttention(),
            AttentionFunction.XFORMERS: XFormersAttention(),
        }
        targets[AttentionFunction.DEFAULT] = targets[
            AttentionFunction.XFORMERS
            if memory_efficient_attention is not None
            else AttentionFunction.PYTORCH
        ]
        for module in self.modules.transformer.modules():
            if isinstance(module, Attention):
                if module.attention_function not in targets:
                    raise ValueError(
                        "Compilation is unqualified for this attention implementation"
                    )
                module.attention_function = targets[module.attention_function]
        for module in self.modules.transformer.modules():
            if isinstance(module, BasicAVTransformerBlock):
                module.compile(fullgraph=True, dynamic=True)
                self.compiled_blocks += 1
        if not self.compiled_blocks:
            raise RuntimeError("No transformer blocks found to compile")
        started = time.monotonic()
        self._warm_history_shapes()
        self.compile_warmup_seconds = time.monotonic() - started

    def _warm_history_shapes(self):
        # Warm all three bounded-history shapes before announcing readiness.
        # The dummy audio is local silence, never published or put in a session.
        try:
            for profile in self.listening.profiles:
                self.reset()
                self.warm_profile = profile
                for _ in range(3):
                    count = chunk_frames(self.position) * FRAME_SAMPLES
                    self.generate(
                        AudioChunk(
                            "compile-warmup",
                            1,
                            self.position,
                            np.zeros(count, dtype=np.int16),
                            count,
                            np.zeros(LOOKAHEAD_SAMPLES, dtype=np.int16),
                        ),
                        threading.Event(),
                    )
        finally:
            self.warm_profile = None
            self.reset()

    @torch.inference_mode()
    def open_session(self, image: Path):
        self.reset()
        # All character-specific state is replaced before this session is ready.
        self.reference = None
        with torch.autocast("cuda", dtype=self.dtype):
            pixels = load_image_conditioning(
                str(image), HEIGHT, WIDTH, self.dtype, self.device, crf=0
            )
            self.reference = (
                self.modules.video_encoder(pixels)[:1].detach().contiguous()
            )
        torch.cuda.synchronize()

    def reset(self):
        self.listening.reset()
        self.sink = None
        self.previous = None
        self.audio_left = np.zeros(0, dtype=np.int16)
        self.position = 0
        self.chunk_index = 0
        self.generator = torch.Generator(device=self.device).manual_seed(0)

    def close_session(self):
        self.reset()
        self.reference = None

    def info(self):
        return {
            "adapter": "avatarforever_bounded_v1",
            "geometry": {"width": WIDTH, "height": HEIGHT, "fps": 25},
            "qualification": "streaming_experimental",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "first_chunk_frames": 25,
            "following_chunk_frames": 32,
            "maximum_video_history_latents": 8,
            "audio_left_context_seconds": 1.28,
            "audio_lookahead_seconds": 0.28,
            "audio_conditioning_steps_per_chunk": 32,
            "precision": str(self.dtype),
            "seed": 0,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "text_encoder_offloaded": True,
            "listening_control": self.listening.info(),
            "execution": "compiled_blocks" if self.compiled_blocks else "eager",
            "compiled_blocks": self.compiled_blocks,
            "fixed_prompt_cache": self.prompt_cache is not None,
            "fixed_prompt_cache_bytes": self.prompt_cache.bytes
            if self.prompt_cache is not None
            else 0,
            "attention_backend": "xformers"
            if memory_efficient_attention is not None
            else "pytorch",
            "compile_warmup_seconds": self.compile_warmup_seconds,
            "denoising_sigmas": self.sigmas.tolist(),
            "memory": self.metrics(),
        }

    def metrics(self):
        # Peaks intentionally span the process lifetime, including model loading;
        # never reset them between turns and hide an earlier allocation spike.
        return {
            "gpu_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(self.device),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "scope": "GPU subprocess lifetime; PyTorch allocations and Linux peak RSS",
            "last_stage_ms": self.last_stage_ms,
            "listening_control": self.listening.info(),
        }

    def _stage(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.stage_events.append((name, event))

    @torch.inference_mode()
    def generate(self, audio: AudioChunk, cancelled: threading.Event) -> GeneratedBatch:
        def check():
            if cancelled.is_set():
                raise GenerationCancelled()

        check()
        expected = chunk_frames(audio.sample_start)
        if (
            audio.sample_start != self.position
            or len(audio.pcm) != expected * FRAME_SAMPLES
        ):
            raise ValueError("Noncontiguous or invalid native audio chunk")
        torch.cuda.synchronize()
        started = time.monotonic()
        self.stage_events = []
        self._stage("start")
        with torch.autocast("cuda", dtype=self.dtype):
            cue = self.listening.preview(audio)
            profile = self.warm_profile or cue.profile
            result = self._generate(audio, expected, check, profile)
        self._stage("readback_and_history")
        torch.cuda.synchronize()
        self.last_stage_ms = {
            name: round(previous.elapsed_time(event), 3)
            for (_, previous), (name, event) in zip(
                self.stage_events, self.stage_events[1:]
            )
        }
        check()
        self.listening.commit(cue)
        self.position += audio.valid_samples
        return GeneratedBatch(
            audio,
            result[: math.ceil(audio.valid_samples / FRAME_SAMPLES)],
            time.monotonic() - started,
        )

    def _generate(self, audio, frame_count, check, profile="default"):
        video_context, audio_context = self.prompt_contexts[profile]
        # Audio latents run at 25 Hz. Keep the already-produced history features
        # fixed; encode only a bounded prefix plus the newly arrived PCM.
        if len(audio.lookahead) != LOOKAHEAD_SAMPLES:
            raise ValueError("AvatarForever requires bounded audio lookahead")
        pcm = np.concatenate((self.audio_left, audio.pcm, audio.lookahead))
        waveform = (
            torch.from_numpy(pcm.astype(np.float32) / 32768)
            .to(self.device)[None, None]
            .repeat(1, 2, 1)
        )
        encoded = encode_audio(Audio(waveform, SAMPLE_RATE), self.modules.audio_encoder)
        # Upstream assigns 32 audio steps to four video latents. The initial
        # four latents decode to 25 frames, so later chunks need seven arrived
        # future steps. Assigning 25/32 audio steps instead weakened lip motion.
        offset = len(self.audio_left) // FRAME_SAMPLES + (7 if self.position else 0)
        current_audio = encoded[:, :, offset : offset + 32].contiguous()
        if current_audio.shape[2] != 32:
            raise ValueError("Audio encoder did not cover the requested PCM interval")
        self._stage("audio_encode")
        check()
        history = [] if self.sink is None else [self.sink]
        if self.chunk_index > 1:
            history.append(self.previous)
        latent_count = 4 * (len(history) + 1)
        shape = VideoPixelShape(1, (latent_count - 1) * 8 + 1, HEIGHT, WIDTH, 25)
        noiser = GaussianNoiser(generator=self.generator)
        conditions = (
            [VideoConditionByLatentIndex(self.reference, 1.0, 0)] if not history else []
        )
        video, video_tools = noise_video_state(
            output_shape=shape,
            noiser=noiser,
            conditionings=conditions,
            components=self.pipeline.pipeline_components,
            dtype=self.dtype,
            device=self.device,
        )
        tokens_per_frame = video.latent.shape[1] // latent_count
        tokens_per_chunk = tokens_per_frame * 4
        for index, (latents, _) in enumerate(history):
            tokens = self.pipeline.pipeline_components.video_patchifier.patchify(
                latents
            )
            begin, end = index * tokens_per_chunk, (index + 1) * tokens_per_chunk
            video.latent[:, begin:end] = tokens
            video.clean_latent[:, begin:end] = tokens
            video.denoise_mask[:, begin:end] = 0
        video = apply_video_channel_condition_to_state(
            video_state=video,
            video_tools=video_tools,
            channel_condition=self.reference.repeat(1, 1, latent_count, 1, 1),
        )
        audio_features = torch.cat([*(h[1] for h in history), current_audio], dim=2)
        if audio_features.shape[2] != 32 * (len(history) + 1):
            raise ValueError("Bounded audio/video history became misaligned")
        sound, _ = noise_audio_state(
            output_shape=shape._replace(frames=audio_features.shape[2]),
            noiser=noiser,
            conditionings=[],
            components=self.pipeline.pipeline_components,
            dtype=self.dtype,
            device=self.device,
            noise_scale=0.0,
            initial_latent=audio_features,
        )
        sound = replace(sound, denoise_mask=torch.zeros_like(sound.denoise_mask))
        video_ranges = [
            (i * tokens_per_chunk, (i + 1) * tokens_per_chunk)
            for i in range(len(history) + 1)
        ]
        audio_sizes = [h[1].shape[2] for h in history] + [current_audio.shape[2]]
        audio_ranges, cursor = [], 0
        for size in audio_sizes:
            audio_ranges.append((cursor, cursor + size))
            cursor += size
        selected = list(range(len(history) + 1))
        cache = (
            ARFeatureCache()
        )  # upstream cache is per denoised chunk, across its steps
        denoise = simple_denoising_func(
            video_context=video_context,
            audio_context=audio_context,
            transformer=self.modules.transformer,
        )
        stepper = EulerDiffusionStep()
        self._stage("state_prepare")
        for step in range(len(self.sigmas) - 1):
            check()
            if self.prompt_cache is not None:
                self.prompt_cache.select(step, profile)
            window, current_slice = _select_ar_chunk_state(
                video,
                chunk_token_ranges=video_ranges,
                selected_chunk_indices=selected,
                current_chunk_idx=len(history),
                relative_positions=True,
                channel_condition_current_chunk_only=True,
                video_tokens_per_frame=tokens_per_frame,
                current_temporal_index=latent_count - 1,
            )
            sound_window, sound_slice = _select_ar_chunk_state(
                sound,
                chunk_token_ranges=audio_ranges,
                selected_chunk_indices=selected,
                current_chunk_idx=len(history),
                relative_positions=True,
            )
            prediction, _ = denoise(
                window,
                sound_window,
                self.sigmas,
                step,
                ar_feature_cache=cache,
                video_current_slice=current_slice,
                audio_current_slice=sound_slice,
            )
            prediction = post_process_latent(
                prediction, window.denoise_mask, window.clean_latent
            )
            updated = stepper.step(window.latent, prediction, self.sigmas, step)
            updated = post_process_latent(
                updated, window.denoise_mask, window.clean_latent
            )
            video.latent[:, -tokens_per_chunk:] = updated[:, current_slice]
        check()
        self._stage("denoise")
        latents = (
            video_tools.unpatchify(video_tools.clear_conditioning(video))
            .latent[:, :, -4:]
            .contiguous()
        )
        # Decode only contiguous recent history, never sink + a disjoint past.
        decode_latents = (
            torch.cat((self.previous[0], latents), dim=2) if self.previous else latents
        )
        frames = torch.cat(
            list(
                decode_video(
                    decode_latents, self.modules.video_decoder, generator=self.generator
                )
            )
        )
        frames = frames[-frame_count:]
        if tuple(frames.shape) != (frame_count, HEIGHT, WIDTH, 3):
            raise ValueError("Native VAE output violated the 4:3 contract")
        check()
        self._stage("video_decode")
        result = frames.cpu().numpy()
        record = (latents.detach().clone(), current_audio.detach().clone())
        if self.sink is None:
            self.sink = record
        self.previous = record
        self.audio_left = audio.pcm[-32 * FRAME_SAMPLES :].copy()
        self.chunk_index += 1
        return result

    def close(self):
        self.reset()
        self.pipeline.clear_fast_cache()
