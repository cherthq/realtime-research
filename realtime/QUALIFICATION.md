# Tested scope and limitations

The demo is experimental and supports one active call per GPU. The working
deployment uses one H100 PCIe 80 GB, BF16 weights, four denoising steps, native
512×384 video, a 24 FPS publication cap, LiveKit Cloud and ElevenLabs speech.
The ten-minute call limit remains in place.

Short synthetic-input browser checks have exercised speech, captions, interruption,
web lookup, spoken search acknowledgement and subsequent replies. These checks
do not establish naturalness, physical speaker timing or multi-user capacity.

The public-demo cleanup passed 231 backend tests from an exported clean checkout
(six opt-in tests skipped), eleven browser UI checks, and a real stock-voice
call through the local connection. The latter received 512×384 video and a
completed spoken reply with no browser errors. Public HTTPS media delivery
remains separate from that local-connection result.

Long-call visual drift remains unresolved. A twelve-minute replay showed late
scene/color degradation and false color. Decoder comparisons did not remove it.
Experimental cache/scene-prompt changes have not passed visual qualification and
are not included in the deployed runtime. Earlier FPS measurements are not
evidence that long calls are visually stable.

The optional `onset_boost` listening profile reduces idle mouth motion and uses
a stronger speaking cue for the first two audible chunks. It can still produce
idle mouth motion or imperfect speech onset. See [implementation details](../docs/listening-profile.md).

Public HTTPS, device permissions and actual media delivery must be checked for
each deployment. Unit tests and a successful page load are not a completed call.
The repository contains no API keys, generated recordings or private voice clones.

## Component terms

These are records of the inspected upstream files, not a blanket commercial-use
clearance. Preserve source notices and review each component independently.

| Component | Recorded terms/source |
| --- | --- |
| AvatarForever source | Pinned repository `LICENSE` is **LTX-2 Community License Agreement**, dated January 5, 2026; it is not MIT/Apache. [Pinned license](https://github.com/leeruibin/avatarforever/blob/de4deb6a55c9567a17dad58d67390fe089993630/LICENSE). |
| AvatarForever checkpoint | The inspected pinned repository contains the checkpoint and Git attributes but no separate model card/license. Do not infer unrestricted weight rights from availability. [Pinned weights](https://huggingface.co/LetsThink/AvatarForever/tree/622ea82482c09e7fb2d8e29cb7ff1d009c820f05). |
| LTX-2.3 backbone | The released checkpoint is named `avatarforever-ltx-2.3-22b.safetensors`; the vendored source carries the community terms above. Separate original-backbone weight terms are not established by that filename. |
| Gemma text encoder | Pinned README declares `license: gemma` and base model `google/gemma-3-12b-it`. Access was granted through Hugging Face. [Pinned model card](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/blob/68f7ee4fbd59087436ada77ed2d62f373fdd4482/README.md). |

Cancellation uses LiveKit's documented targeted data packets and validates the
sender, topic and epoch. Reliable delivery is best effort, so HTTP status still
recovers missed notices. [LiveKit data-packet reference](https://docs.livekit.io/transport/data/packets/).
