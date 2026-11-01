"""Run in the AvatarForever environment; the HTTP-only environment needs no torch."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from realtime.gpu.prompt_cache import FixedPromptCache, FixedPromptProjection  # noqa: E402


def test_cache_preserves_per_step_projections_and_never_wraps_self_attention():
    text = SimpleNamespace(to_k=torch.nn.Linear(3, 5), to_v=torch.nn.Linear(3, 5))
    dynamic = torch.nn.Linear(3, 5)
    block = SimpleNamespace(attn2=text, attn1=dynamic)
    cache = FixedPromptCache([block], 4)
    contexts = [torch.randn(1, 7, 3) for _ in range(4)]
    expected = []
    with torch.inference_mode():
        for step, context in enumerate(contexts):
            cache.select(step)
            expected.append((text.to_k(context).clone(), text.to_v(context).clone()))
        # Verify the fixed context over another streaming-history length.
        for step, context in enumerate(contexts):
            cache.select(step)
            text.to_k(context)
            text.to_v(context)
        cache.freeze()

        # Original matrix operations must not execute during cached generation.
        def unexpected(*args):
            raise AssertionError("Projection was recomputed")

        text.to_k.source.forward = unexpected
        text.to_v.source.forward = unexpected
        for step in [3, 0, 2, 1, 0]:
            cache.select(step)
            assert torch.equal(text.to_k(contexts[step]), expected[step][0])
            assert torch.equal(text.to_v(contexts[step]), expected[step][1])
    assert block.attn1 is dynamic
    assert cache.bytes == sum(
        t.numel() * t.element_size() for pair in expected for t in pair
    )


def test_cache_rejects_incomplete_schedule_and_context_changes_during_capture():
    projection = FixedPromptProjection(torch.nn.Linear(3, 5), 4)
    context = torch.ones(1, 7, 3)
    with torch.inference_mode():
        projection(context)
        with pytest.raises(ValueError, match="context varies"):
            projection(context + 1)
        with pytest.raises(ValueError, match="incomplete"):
            projection.freeze()
        for step in [-1, 4]:
            with pytest.raises(ValueError, match="outside"):
                projection.select(step)


def test_profiles_keep_distinct_outputs_and_require_complete_capture():
    profiles = ("quiet", "onset", "speaking")
    source = torch.nn.Linear(3, 5)
    projection = FixedPromptProjection(source, 2, profiles)
    expected = {}
    with torch.inference_mode():
        for index, profile in enumerate(profiles):
            for step in range(2):
                context = torch.full((1, 7, 3), float(index * 10 + step))
                projection.select(step, profile)
                expected[profile, step] = source(context).clone()
                assert torch.equal(projection(context), expected[profile, step])
            if profile != profiles[-1]:
                with pytest.raises(ValueError, match="incomplete"):
                    projection.freeze()
        projection.freeze()
        for profile in ["onset", "quiet", "speaking", "quiet", "onset"]:
            for step in [1, 0]:
                projection.select(step, profile)
                assert torch.equal(projection(torch.empty(0)), expected[profile, step])
        with pytest.raises(ValueError, match="Unknown"):
            projection.select(0, "missing")


def test_compiled_projection_observes_profile_switch_without_reusing_old_output():
    projection = FixedPromptProjection(torch.nn.Linear(3, 5), 2, ("quiet", "speaking"))
    context = torch.ones(1, 7, 3)
    expected = {}
    with torch.inference_mode():
        for profile, multiplier in [("quiet", 1), ("speaking", 3)]:
            for step in range(2):
                projection.select(step, profile)
                expected[profile, step] = projection(
                    context * (multiplier + step)
                ).clone()
        projection.freeze()
        compiled = torch.compile(
            projection, backend="eager", fullgraph=True, dynamic=True
        )
        for profile in ["speaking", "quiet", "speaking", "quiet"]:
            for step in [1, 0]:
                projection.select(step, profile)
                assert torch.equal(compiled(context), expected[profile, step])
