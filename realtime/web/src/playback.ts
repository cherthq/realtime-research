import { getBrowser, type RemoteAudioTrack, type RemoteVideoTrack } from 'livekit-client';

export type PlaybackReceipt = {
  frame_id: number | null;
  audio_playing: boolean;
  completed_frame_id?: number;
  cancel_epoch?: number;
  mute_ms?: number;
};

/** Gates browser output by response epoch and acknowledges displayed frame IDs.
 * The receipt is an A/V playback proxy, never a physical-speaker measurement.
 */
export class BrowserPlayback {
  private minimumEpoch = 0;
  private lastCancel = -1;
  private lastFrame: number | null = null;
  private lastReport = 0;
  private completedFrame: number | undefined;
  private reportPending = false;
  private callback = 0;
  private stopped = false;
  private interruptSequence = 0;
  private pendingInterrupt = 0;
  private holds = new Set<'status' | 'media'>();
  private open = false;
  private video: HTMLVideoElement | null = null;
  private cover: HTMLCanvasElement | null = null;
  private coverReady = false;
  private sampleWindow = new Float32Array(128);
  private presentedFrames = 0;
  private lastPresentedAt = 0;
  private videoWaiters = new Set<(error?: Error) => void>();
  private audio = new Map<
    HTMLMediaElement,
    { track: RemoteAudioTrack; gate: GainNode; meter: AnalyserNode; readiness: AnalyserNode }
  >();
  private audioReceivers = new WeakSet<RTCRtpReceiver>();

  constructor(
    private send: (receipt: PlaybackReceipt) => Promise<unknown>,
    private visible: (value: boolean) => void,
    private blocked: () => void,
    private error: (message: string) => void,
    private context: AudioContext,
    private continuousVideo = false,
  ) {}

  private playing() {
    return this.open && this.context.state === 'running' && this.audio.size > 0;
  }

  get videoDelayed() {
    return (
      !this.stopped && this.presentedFrames > 0 && performance.now() - this.lastPresentedAt > 500
    );
  }

  /** Readiness is decoded video, not room membership or speech starting. */
  waitForVideo(timeoutMs = 15000): Promise<void> {
    if (this.stopped) return Promise.reject(new Error('Call ended before video was ready.'));
    if (this.presentedFrames >= 3) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const done = (error?: Error) => {
        clearTimeout(timer);
        this.videoWaiters.delete(done);
        if (error) reject(error);
        else resolve();
      };
      const timer = setTimeout(
        () => done(new Error('Avatar video did not start. End the call and reconnect.')),
        timeoutMs,
      );
      this.videoWaiters.add(done);
    });
  }

  attachAudio(track: RemoteAudioTrack) {
    this.forwardEncodedAudio(track);
    if (this.stopped) return null;
    const gate = this.context.createGain(),
      meter = this.context.createAnalyser(),
      readiness = this.context.createAnalyser();
    gate.gain.value = this.open ? 1 : 0;
    meter.fftSize = 128;
    readiness.fftSize = 128;
    // LiveKit webAudioMix owns source -> plugins -> output. Its audio elements
    // remain silent, so startAudio/autoplay recovery cannot bypass this gate.
    // Observe decoded PCM before our output gate. Otherwise a closed gate
    // makes audio readiness indistinguishable from missing audio.
    track.setWebAudioPlugins([readiness, gate, meter]);
    const el = track.attach();
    this.audio.set(el, { track, gate, meter, readiness });
    el.dataset.playbackOpen = String(this.open);
    void el.play().catch(this.blocked);
    return el;
  }

  private forwardEncodedAudio(track: RemoteAudioTrack) {
    // LiveKit 2.23 enables legacy insertable streams for frame metadata in
    // Chromium, but only wires VIDEO receivers. Unconnected AUDIO streams then
    // receive RTP without decoding it. Supply the missing identity transform.
    // This room has no E2EE; never apply this workaround to an encrypted room.
    const browser = getBrowser();
    const chromium = browser?.name === 'Chrome' && browser.os !== 'iOS';
    const scriptTransform = 'RTCRtpScriptTransform' in window && !chromium;
    const receiver = track.receiver as
      | (RTCRtpReceiver & {
          createEncodedStreams?: () => { readable: ReadableStream; writable: WritableStream };
        })
      | undefined;
    if (scriptTransform || !receiver?.createEncodedStreams || this.audioReceivers.has(receiver))
      return;
    this.audioReceivers.add(receiver);
    try {
      const { readable, writable } = receiver.createEncodedStreams();
      void readable.pipeTo(writable).catch(() => {
        if (!this.stopped) this.error('Audio decoding stopped. End the call and reconnect.');
      });
    } catch {
      this.error('Audio decoding could not start. End the call and reconnect.');
    }
  }

  detachAudio(el: HTMLMediaElement) {
    const nodes = this.audio.get(el);
    if (nodes) {
      nodes.gate.gain.value = 0;
      nodes.track.setWebAudioPlugins([]);
      nodes.gate.disconnect();
      nodes.meter.disconnect();
      nodes.readiness.disconnect();
    }
    this.audio.delete(el);
  }

  attachVideo(track: RemoteVideoTrack, element: HTMLVideoElement, cover?: HTMLCanvasElement) {
    // Generated video otherwise presents ahead of Opus decoding + audio output.
    // Request an 80ms video jitter-buffer target (two 25 FPS frames). This is a browser
    // hint, not a sync guarantee; qualify again on the real remote route/device.
    if (track.receiver && 'jitterBufferTarget' in track.receiver) {
      (
        track.receiver as RTCRtpReceiver & { jitterBufferTarget: number | null }
      ).jitterBufferTarget = 80;
    }
    if (typeof element.requestVideoFrameCallback !== 'function') {
      this.error('Frame timing is unavailable in this browser. Try a current Chromium browser.');
      return;
    }
    if (this.video && this.callback) this.video.cancelVideoFrameCallback(this.callback);
    this.video = element;
    this.cover = cover ?? null;
    let missing = 0;
    const next = (_now: number, frame: VideoFrameCallbackMetadata) => {
      if (this.stopped) return;
      this.lastPresentedAt = performance.now();
      this.callback = element.requestVideoFrameCallback(next);
      // Keep the video element decoding under the last generated frame during
      // cancellation. Never swap in the portrait or pause the live decoder.
      if (this.continuousVideo) this.visible(true);
      if (++this.presentedFrames === 3) {
        for (const done of this.videoWaiters) done();
      }
      const metadata =
        frame.rtpTimestamp === undefined
          ? undefined
          : track.lookupFrameMetadata({ rtpTimestamp: frame.rtpTimestamp });
      if (!metadata) {
        if (++missing === 75)
          this.error(
            'Frame timing is unavailable in this browser. Try a current Chromium browser.',
          );
        return;
      }
      missing = 0;
      const ident = metadata.frameId,
        epoch = Math.floor(ident / 65536) - 1,
        position = ident % 65536;
      if (this.pendingInterrupt || (position > 0 && epoch < this.minimumEpoch)) {
        this.freezeVideo();
        return;
      }
      if (position === 0 && epoch < this.lastCancel) return;
      if (
        position === 0 &&
        this.lastFrame !== null &&
        this.lastFrame % 65536 > 0 &&
        this.playing()
      ) {
        this.completedFrame = this.lastFrame;
      }
      this.lastFrame = ident;
      if (position > 0 && !this.pendingInterrupt && !this.holds.size) {
        this.visible(true);
        if (!this.open) {
          if (!this.audio.size || !this.decodedAudioReady()) {
            this.freezeVideo();
            return;
          }
          this.open = true;
          for (const [el, nodes] of this.audio) {
            nodes.gate.gain.setValueAtTime(1, this.context.currentTime);
            el.dataset.playbackOpen = 'true';
          }
          // Bounded, audio-free timing evidence for the optional browser latency
          // collector. This is a compositor timestamp, not physical output.
          performance.clearMarks('chert-playback-start');
          performance.mark('chert-playback-start', {
            detail: { frameId: ident, expectedDisplayTime: frame.expectedDisplayTime },
          });
        }
      }
      if (position > 0 && !this.playing()) this.freezeVideo();
      else if (!this.holds.size) this.rememberVideo();
      if (!this.reportPending && performance.now() - this.lastReport >= 100) {
        this.lastReport = performance.now();
        this.reportPending = true;
        const completed = this.completedFrame;
        void this.send({
          frame_id: ident,
          audio_playing: this.playing(),
          completed_frame_id: completed,
        })
          .then(() => {
            if (this.completedFrame === completed) this.completedFrame = undefined;
          })
          .catch(() => {})
          .finally(() => {
            this.reportPending = false;
          });
      }
    };
    this.callback = element.requestVideoFrameCallback(next);
  }

  private decodedAudioReady() {
    if (this.context.state !== 'running') return false;
    for (const { readiness } of this.audio.values()) {
      readiness.getFloatTimeDomainData(this.sampleWindow);
      if (this.sampleWindow.some((value) => Math.abs(value) > 0.00003)) return true;
    }
    return false;
  }

  private rememberVideo() {
    if (!this.cover || !this.video?.videoWidth) return;
    const context = this.cover.getContext('2d');
    if (!context) return;
    context.drawImage(this.video, 0, 0, this.cover.width, this.cover.height);
    this.coverReady = true;
    this.cover.hidden = true;
  }

  private freezeVideo() {
    if (this.continuousVideo && this.cover && this.coverReady) this.cover.hidden = false;
  }

  private mute() {
    this.open = false;
    this.completedFrame = undefined;
    for (const [el, nodes] of this.audio) {
      nodes.gate.gain.cancelScheduledValues(this.context.currentTime);
      nodes.gate.gain.setValueAtTime(0, this.context.currentTime);
      el.dataset.playbackOpen = 'false';
    }
    if (!this.continuousVideo) this.visible(false);
  }

  hold(reason: 'status' | 'media') {
    this.holds.add(reason);
    this.mute();
    this.freezeVideo();
  }

  resume(reason: 'status' | 'media') {
    // A fresh frame, not a successful HTTP request or autoplay click, opens
    // playback. Independent holds cannot accidentally clear one another.
    this.holds.delete(reason);
  }

  interruptLocally() {
    this.pendingInterrupt = ++this.interruptSequence;
    const playing = this.playing();
    this.minimumEpoch = Math.max(
      this.minimumEpoch,
      this.lastFrame === null ? 1 : Math.floor(this.lastFrame / 65536),
    );
    this.mute();
    this.freezeVideo();
    void this.send({ frame_id: this.lastFrame, audio_playing: playing }).catch(() => {});
    return this.pendingInterrupt;
  }

  interruptConfirmed(sequence: number, epoch: number) {
    this.cancelled(epoch);
    // The last displayed frame may still belong to an earlier response. Keep
    // output closed until this request tells us the authoritative cancelled epoch.
    if (sequence === this.pendingInterrupt) this.pendingInterrupt = 0;
  }

  cancelled(epoch: number) {
    if (epoch <= this.lastCancel) return;
    const mark = performance.now(),
      playing = this.playing();
    this.lastCancel = epoch;
    this.minimumEpoch = Math.max(this.minimumEpoch, epoch + 1);
    // A late status/data notice for an older response must not chop the newer
    // response that is already displayed. Local interruption still gates it.
    if (this.lastFrame === null || Math.floor(this.lastFrame / 65536) - 1 <= epoch) {
      this.mute();
      if (this.lastFrame !== null && this.lastFrame % 65536 > 0) this.freezeVideo();
    }
    void this.send({
      frame_id: this.lastFrame,
      audio_playing: playing,
      cancel_epoch: epoch,
      mute_ms: performance.now() - mark,
    }).catch(() => {});
  }

  close() {
    this.stopped = true;
    for (const done of this.videoWaiters) done(new Error('Call ended before video was ready.'));
    this.mute();
    if (this.video && this.callback) this.video.cancelVideoFrameCallback(this.callback);
    for (const [el, nodes] of this.audio) {
      // Detach the SDK source before removing our gate from its plugin chain.
      nodes.track.detach(el);
      this.detachAudio(el);
    }
    this.video = null;
    if (this.cover) this.cover.hidden = true;
    this.cover = null;
    this.coverReady = false;
  }
}
