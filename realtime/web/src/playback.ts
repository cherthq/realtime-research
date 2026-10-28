import { getBrowser, type RemoteAudioTrack, type RemoteVideoTrack } from 'livekit-client';

export type PlaybackReceipt = {
  frame_id: number | null;
  audio_playing: boolean;
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
  private reportPending = false;
  private callback = 0;
  private stopped = false;
  private interruptSequence = 0;
  private pendingInterrupt = 0;
  private holds = new Set<'status' | 'media'>();
  private open = false;
  private video: HTMLVideoElement | null = null;
  private audio = new Map<
    HTMLMediaElement,
    { track: RemoteAudioTrack; gate: GainNode; meter: AnalyserNode }
  >();
  private audioReceivers = new WeakSet<RTCRtpReceiver>();

  constructor(
    private send: (receipt: PlaybackReceipt) => Promise<unknown>,
    private visible: (value: boolean) => void,
    private blocked: () => void,
    private error: (message: string) => void,
    private context: AudioContext,
  ) {}

  private playing() {
    return this.open && this.context.state === 'running' && this.audio.size > 0;
  }

  attachAudio(track: RemoteAudioTrack) {
    this.forwardEncodedAudio(track);
    if (this.stopped) return null;
    const gate = this.context.createGain(),
      meter = this.context.createAnalyser();
    gate.gain.value = this.open ? 1 : 0;
    meter.fftSize = 128;
    // LiveKit webAudioMix owns source -> plugins -> output. Its audio elements
    // remain silent, so startAudio/autoplay recovery cannot bypass this gate.
    track.setWebAudioPlugins([gate, meter]);
    const el = track.attach();
    this.audio.set(el, { track, gate, meter });
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
    }
    this.audio.delete(el);
  }

  attachVideo(track: RemoteVideoTrack, element: HTMLVideoElement) {
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
    let missing = 0;
    const next = (_now: number, frame: VideoFrameCallbackMetadata) => {
      if (this.stopped) return;
      this.callback = element.requestVideoFrameCallback(next);
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
      if (epoch < this.minimumEpoch) return; // late old-response video cannot reopen audio
      this.lastFrame = ident;
      if (position > 0 && !this.pendingInterrupt && !this.holds.size) {
        this.visible(true);
        if (!this.open) {
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
      if (!this.reportPending && performance.now() - this.lastReport >= 100) {
        this.lastReport = performance.now();
        this.reportPending = true;
        void this.send({ frame_id: ident, audio_playing: this.playing() })
          .catch(() => {})
          .finally(() => {
            this.reportPending = false;
          });
      }
    };
    this.callback = element.requestVideoFrameCallback(next);
  }

  private mute() {
    this.open = false;
    for (const [el, nodes] of this.audio) {
      nodes.gate.gain.cancelScheduledValues(this.context.currentTime);
      nodes.gate.gain.setValueAtTime(0, this.context.currentTime);
      el.dataset.playbackOpen = 'false';
    }
    this.visible(false);
  }

  hold(reason: 'status' | 'media') {
    this.holds.add(reason);
    this.mute();
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
    if (this.lastFrame === null || Math.floor(this.lastFrame / 65536) - 1 <= epoch) this.mute();
    void this.send({
      frame_id: this.lastFrame,
      audio_playing: playing,
      cancel_epoch: epoch,
      mute_ms: performance.now() - mark,
    }).catch(() => {});
  }

  close() {
    this.stopped = true;
    this.mute();
    if (this.video && this.callback) this.video.cancelVideoFrameCallback(this.callback);
    for (const [el, nodes] of this.audio) {
      // Detach the SDK source before removing our gate from its plugin chain.
      nodes.track.detach(el);
      this.detachAudio(el);
    }
    this.video = null;
  }
}
