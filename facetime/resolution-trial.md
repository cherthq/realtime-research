# 640 × 480 deployment trial — November 26, 2026

The private FaceTime/browser backend was changed from `AVATAR_RESOLUTION=512x384`
to `640x480` and left enabled after validation. It uses the trained two-step
AvatarForever sampler on an H100 PCIe, with 25 FPS native generation and a
24 FPS publication cap. These measurements describe that private deployment;
its trained weights are not included in this repository.

## Reproduce the setting

Source [the resolution configuration](config/avatar-resolution.sh) into the
backend's launch environment **before starting Python**. For example, from the
repository root:

```sh
. facetime/config/avatar-resolution.sh
# Start the prepared backend with its existing factory and private credentials.
```

Geometry is imported at process startup, so setting it only in a configuration
file loaded later is insufficient. Preserve the variable in the process
supervisor's restart environment. The deployed supervisor retained 640 × 480
through a managed restart during verification.

Restart only while the shared backend has no active call. On this host, stop
the companion model before cold-loading the avatar, warm the avatar first,
then start the companion. Loading both together can exhaust GPU memory.
To roll back, set `AVATAR_RESOLUTION=512x384` in the supervisor's launch
environment and repeat the idle restart and warmup sequence.

## Measured comparison

The same prerecorded articulation scenario passed through two LiveKit hops
using the deployed `forwardTracks` implementation at each resolution.

| Measurement | 512 × 384 | 640 × 480 |
| --- | ---: | ---: |
| Received frame dimensions | 512 × 384 | 640 × 480 |
| Model generation throughput | 43.78 FPS | 32.76 FPS |
| GPU round-trip throughput | 42.91 FPS | 31.97 FPS |
| Play request to first audible PCM | 1,318 ms | 2,347 ms |
| Maximum received video-frame gap | 118 ms | 166 ms |
| Speech-buffer underrun frames | 0 | 0 |

The larger frame contains 56.25% more pixels. Generation sustained more than
25 FPS in this short test, but audio arrival took about one second longer.
These timings exclude live speech recognition/model response time and physical
FaceTime handset playback. They are single short replays, not a latency SLA or
proof of improved perceptual quality.

Steady GPU usage after the higher-resolution test, including the companion,
was 73,709 MiB used with 7,371 MiB free.

## Live voice verification

An existing voice-provider authentication failure was resolved by updating the
private credential and restarting the backend. Credentials remain outside Git.
The subsequent GPT-Live 1.1 named-agent test received 622 video frames at
640 × 480 and audible output 5,648 ms after bridge readiness, within the
ten-second startup deadline. The maximum received video-frame gap was 110 ms.
The temporary test room was removed and the backend returned idle.

That November 26 test included the automatic FaceTime greeting. The greeting
was subsequently reverted on `main`; restoring this resolution configuration
does not restore it. The startup timing above is historical and does not
validate silent-start behavior with the greeting removed.

This verifies the avatar and both LiveKit forwarding hops. A physical iPhone
call was not placed during this trial; handset appearance and perceived latency
still require a phone-side comparison. The project's Fill setting remains a
separate per-project configuration.
