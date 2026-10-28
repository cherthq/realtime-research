# Private demo qualification

Measured 2026-10-28. The interactive private demo works; this record separates
functional verification from performance targets. Results apply to one H100
80GB, direct 640x480 at native 25 FPS, and the pinned releases in
`configs/models.json`. They do not establish a production service SLA.

## Recorded evidence

Generated media, detailed reports and private deployment configuration stay
under ignored `runs/`, `data/` and `.cache/`. The qualification run is
`runs/avatarforever/qualification-20261028-v2/`; inputs are the frozen
`data/eval/facetime-avatar-voices-v2/` pack. Seed is 0.

| Check | Result and scope |
| --- | --- |
| Three real voice/avatar calls | Passed with synthetic microphone input, effective provider voice verification, browser audio/video, interruption and cleanup. Reports: `runs/realtime/real-avatar-browser/`. |
| Final updated-adapter smoke | Ref01 passed after cancellation/telemetry changes: 57 distinct generated frames, 28.77 compute FPS, no reported stalls, matched Cedar voice, clean end. One preceding attempt ended before generation; its server report is preserved, so uninterrupted network reliability is not claimed. |
| 18 image/audio cases | All completed, 206.5365 seconds total. Native geometry, frame count, complete master audio, hashes and FFmpeg decodability checked. No exactly repeated RGB frames. |
| Pilot generation | 24.21–26.63 generated FPS. This is GPU generation time, not delivered browser FPS. |
| 30 warm short turns | 31 consecutive replays, first excluded from warm statistics. First arrived packet to first decoded chunk: p50 1.786s, p95 1.811s. Generation 25.25–25.83 FPS. Maximum pending input 1.44s. |
| 60-second paced replay | 1,500 distinct decoded frames; compute 59.53s, elapsed replay 66.45s. Compute factor 0.992 leaves little headroom; elapsed time also includes IPC and encoding. |
| Ten-minute continuous paced replay | **Failed:** input exceeded the bounded 20s pending-audio queue. The process exited after 314s including model loading; the requested 600s was not completed. The original failure/logs are preserved. The application recovered and prewarmed successfully. |
| Throughput replay | All six ref01/v2 scenarios completed without paced release. 68.84s audio took 68.44s GPU compute and 71.88s total replay. Generation ranged 24.40–26.40 FPS; complete-path factor 1.044 is slower than realtime even without review-video encoding. |
| Remote browser recovery | Passed blocked-autoplay recovery, microphone denial/retry and a new session after control-connection failure. Reconnect here is a clean new call, not transparent voice-session resumption. |
| Remote flash/beep sample before WebRTC cancellation notices | Ten markers: 42ms p95 absolute audio-graph/video offset. One marker fragmented; raw edges retained. Local interrupt 4.5ms; HTTP-triggered remote interrupt 748.9ms, exceeding the request-to-stop target. |
| Current cancellation path | Immediate local gate plus room-scoped WebRTC notices; status polling remains fallback. Remote retest: 49.9ms p95 graph/video offset, local stop 8.4ms, HTTP-triggered remote stop 489.3ms. The combined timing test still fails its 300ms request-to-stop threshold; this is not a physical or speech-detection measurement. |
| Backend regression/integration checks | Full suite: 72 passed, 3 optional transport checks skipped. All three real-LiveKit checks passed in the separate 12-test affected-code run. |
| Failure-report regression | Seven evaluator checks passed after adding partial metrics to failed runs, including a renderer that fails after one completed chunk. |
| Current browser recovery checks | Seven checks passed on the actual Windows-to-Lambda private TCP route: blocked autoplay, ending during media setup, microphone denial/retry, temporary status failure recovering the same call, persistent failure ending after 30s, immediate rejection of expired credentials, and SSE failure falling back to polling. Six recovery checks and the separate startup-race check also passed locally. |
| Component profile | Synchronized CUDA profiling of a greeting attributes steady full chunks to about 1.07s diffusion, 0.15s video decoding and 0.003s audio encoding. This changes scheduling and is a diagnostic breakdown, not a new end-to-end throughput result. |
| Regional compilation experiment | Same weights, BF16, seed, attention implementation and geometry. Warm full chunks took about 0.83s versus 1.26s eager (~39 versus ~25 generated FPS). First compile is excluded from these timings. RGB MAE stayed below 1/255 across the greeting; this numerical comparison does not prove perceptual equivalence or sustained realtime delivery. |
| Compiled 60-second paced worker replay | Passed: 1,500 distinct frames, 39.29s compute, 61.60s wall time including input pacing/drain, 38.17 generated FPS, 1.28s peak pending audio, first decoded chunk at 1.573s. Cold load took 212.84s including 157.64s compile warm-up. Same public subprocess/IPC path as calls; no browser timing or perceptual lip-sync measured. |
| Compiled ten-minute paced worker replay | **Passed:** all 28,800,000 input samples and 15,000 distinct frames covered; zero exact duplicates, 388.45s compute (factor 0.647), 600.90s wall including pacing/drain, 38.62 generated FPS, first decoded chunk at 1.569s, peak pending audio 1.28s. Same 20s overload bound as the earlier failed eager run. Final allocated GPU memory 39.61GB; lifetime peak 73.59GB includes loading. |
| Warm voice/transport calibration | 31 scripted turns in one remote call; first excluded. Thirty warm turns: p50 1.842s, p95 2.332s from synthetic input completion to both observed audio/video onset. This used the explicitly labeled **diagnostic renderer**, so it isolates voice/transport and does not qualify AvatarForever latency. |
| First compiled live latency attempt | **Failed after 17 completed turns:** the next response showed a late video onset but no detected output audio within 30s. GPU generation reported 37.38 FPS and zero renderer stalls. This is a delivery failure, not a successful 30-warm-turn baseline. Report retained at `runs/realtime/warm-voice-avatarforever-20261028/`. |
| Compiled live latency after idle continuity fix | **All 31 turns completed**, first excluded: 30 warm turns, p50 **2.653s**, p95 **3.256s**, zero reported renderer stalls and no missing audio/video onset in this sample. **The <=3s p95 target failed.** The 239.72s call generated 1,848 distinct frames at 34.81 compute FPS; 4,108 held frames were counted separately. Frozen synthetic prompt, Cedar voice, private SSH/TCP route. Report: `runs/realtime/warm-voice-avatarforever-idle-20261028/`. |
| Final compiled character calls | All three passed after the idle continuity change: real provider voice, generated 640x480 media, the same call/voice recovered after an injected control outage, interruption and cleanup. Effective voices: Cedar/Marin/Cedar. Current reports: `runs/realtime/real-avatar-browser/ref01`, `ref02`, `ref03`; earlier eager reports retained in `real-avatar-browser-before-compiled-20261028`. |
| Updated regression checks | 75 backend tests passed, three optional transport checks skipped. All seven browser recovery/startup tests passed against the remote private TCP service. |

The 18 pilot contact sheets preserve head/shoulders/torso and recognizable
reference appearance in the sampled start/middle/end frames. Still frames do
not verify temporal smoothness or mouth/audio alignment. Perceptual lip-sync
has not been scored. No aggregate quality score is invented.

The browser displays generated and held-frame counts separately. Distinct-frame
statistics mean exact RGB hashes, not perceptually distinct motion. They exclude
held idle images; GPU compute FPS excludes network, voice generation and playback.

Following the failed live latency attempt, the GPU publisher now sends paired
held frames and silence continuously while listening or buffering. Previously it
sent an occasional held frame without audio between replies. Idle media does not
count as generated motion or response audio, and a fixed response-tail deadline
prevents continuing silence from keeping a completed reply active. Nine affected
checks passed, including all three real-LiveKit checks and idle audio/video
continuity with a synthetic GPU process; the full backend suite remained at
75 passed and three optional skips. The subsequent 31-turn browser run completed
without the missing-onset failure; this single successful run does not establish
long-term network reliability or prove the sole cause of the earlier failure.

The frozen v2 pack retains its original voice-profile hashes. The live catalog
was subsequently expanded to four actual recorded previews; Cedar's description
was revised using a second recorded review of the same preview. That review and
the prior profile are retained privately. Final live casting remains Cedar for
ref01/ref03 and Marin for ref02. This metadata revision does not overwrite v2
or imply that its catalog hash matches the newer live catalog.

## Remaining gates

- The eager path failed sustained overload. The compiled path now passes the
  ten-minute paced subprocess/IPC replay with bounded input and memory. That
  establishes continuous decoded-chunk throughput on this H100 configuration,
  not a ten-minute browser/physical-device conversation or public service SLA.
- Remote request-to-stop remains above 300ms in the recorded test. It includes
  network travel; it is not speech-detection latency or physical-speaker timing.
- The scripted browser collector measured 3.256s warm p95 from synthetic speech
  completion to observed output audio/video onset, exceeding the 3s target.
  Its 30 warm turns used one prompt and one character; it is not the 1.811s
  decoded-chunk measurement or a physical-speaker timing result.
- Background-tab/mobile-device behavior, physical-speaker audible-stop and
  acoustic flash/beep timing remain unverified. Output-graph samples do not
  measure DAC, Bluetooth or speaker delay.
- Public HTTPS/WSS and TURN routing have not been deployed or verified. The
  working demo uses a private SSH/TCP route. No claim is made about public NAT
  traversal or multi-user capacity.
- Compare the incremental audio/VAE boundaries against the upstream full-file
  renderer before treating quality as equivalent.

Keep the model labeled experimental until the relevant targets pass. A completed
render is not evidence of sustained realtime delivery.

The compile experiment is `compile-comparison-v3-20261028`. Earlier attempts are
retained as failures: an untraceable callable enum, then repeated guards for
separate attention-callable objects. The working option resolves the same
stateless attention implementation once and compiles repeated transformer blocks.
It follows PyTorch's [regional compilation recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
and [2.7 compile API](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html).
It is explicit opt-in, records compile warm-up separately and fails if compilation
cannot complete; it does not silently change model size, precision or resolution.

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
