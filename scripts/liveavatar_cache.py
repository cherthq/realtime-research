"""Capacity-only correction for the pinned Live Avatar conditioning cache."""

from functools import wraps
import inspect


def grow_conditioning_cache(cache, required):
    """Retain cached values/device/dtype; never truncate conditioning tokens."""
    if required <= 0:
        return False
    key, value = cache["cond_k"], cache["cond_v"]
    if key.shape != value.shape or len(key.shape) != 4:
        raise ValueError("Unexpected conditioning-cache layout")
    capacity = key.shape[1]
    if required <= capacity:
        return False
    if required > 16384:
        raise ValueError(f"Unexpected conditioning-token count: {required}")
    shape = (key.shape[0], required, *key.shape[2:])
    next_key, next_value = key.new_zeros(shape), value.new_zeros(shape)
    next_key[:, :capacity] = key
    next_value[:, :capacity] = value
    cache["cond_k"], cache["cond_v"] = next_key, next_value
    return True


def install_cache_guard(attention_class):
    original = attention_class.forward
    expected = [
        "self",
        "x",
        "seq_lens",
        "grid_sizes",
        "freqs",
        "block_mask",
        "kv_cache",
        "current_start",
        "current_end",
        "sp_size",
        "seg_idx",
        "freqs_cond",
    ]
    if list(inspect.signature(original).parameters) != expected:
        raise ValueError("Pinned Live Avatar attention signature changed")

    @wraps(original)
    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        current_end=0,
        sp_size=None,
        seg_idx=None,
        freqs_cond=None,
    ):
        if kv_cache is not None and seg_idx is not None:
            # Match upstream's conditioning-prefill branch. Temporal attention,
            # offsets, weights and the actual attention math remain unchanged.
            if seg_idx[1] == seg_idx[0] and seg_idx[2] > seg_idx[1]:
                required = int(seg_idx[2] - seg_idx[1])
                if grow_conditioning_cache(kv_cache, required):
                    print(
                        f"Conditioning cache expanded to {required} tokens", flush=True
                    )
        return original(
            self,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            current_end,
            sp_size,
            seg_idx,
            freqs_cond,
        )

    attention_class.forward = forward
