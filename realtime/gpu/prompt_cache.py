"""Per-profile, per-sigma K/V projections for immutable text contexts.

This caches exact projection outputs, not latent history or approximate attention.
It is inference-only and must be rebuilt if prompt, weights, dtype or sigmas change.
The backend captures all steps over three history lengths before compilation.
"""

import torch


class FixedPromptProjection(torch.nn.Module):
    def __init__(self, source, steps, profiles=("default",)):
        super().__init__()
        self.source = source
        self.steps = steps
        self.step = 0
        self.profiles = tuple(profiles)
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Prompt profiles must be nonempty and unique")
        self.profile = self.profiles[0]
        self.contexts = {}
        self.outputs = {}
        self.frozen = False
        self.active = None

    def select(self, step, profile="default"):
        if not 0 <= step < self.steps:
            raise ValueError("Prompt projection step is outside the captured schedule")
        self.step = step
        if profile not in self.profiles:
            raise ValueError("Unknown prompt projection profile")
        self.profile = profile
        if self.frozen:
            self.active = self.outputs[(profile, step)]

    def forward(self, context):
        if self.frozen:
            return self.active
        output = self.source(context)
        key = (self.profile, self.step)
        if key in self.contexts:
            if not torch.equal(context, self.contexts[key]):
                raise ValueError("Prompt context varies with streaming history")
        else:
            self.contexts[key] = context.detach().clone()
            self.outputs[key] = output.detach().clone()
        return output

    def freeze(self):
        expected = {
            (profile, step) for profile in self.profiles for step in range(self.steps)
        }
        if set(self.outputs) != expected:
            raise ValueError("Prompt projection schedule is incomplete")
        self.contexts.clear()
        self.frozen = True
        self.select(0, self.profiles[0])


class FixedPromptCache:
    def __init__(self, blocks, steps, profiles=("default",)):
        self.projections = []
        for block in blocks:
            # Only text cross-attention. Never cache audio-conditioned or
            # self-attention projections, which change with each chunk.
            for name in ("attn2", "audio_attn2"):
                attention = getattr(block, name, None)
                if attention is None:
                    continue
                for attr in ("to_k", "to_v"):
                    projection = FixedPromptProjection(
                        getattr(attention, attr), steps, profiles
                    )
                    setattr(attention, attr, projection)
                    self.projections.append(projection)
        if not self.projections:
            raise ValueError("No text projections found for the fixed prompt")

    def select(self, step, profile="default"):
        for projection in self.projections:
            projection.select(step, profile)

    def freeze(self):
        for projection in self.projections:
            projection.freeze()

    @property
    def bytes(self):
        return sum(
            value.numel() * value.element_size()
            for projection in self.projections
            for value in projection.outputs.values()
        )
