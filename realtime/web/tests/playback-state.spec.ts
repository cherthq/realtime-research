import { test, expect } from '@playwright/test';
import type { RemoteVideoTrack } from 'livekit-client';
import { BrowserPlayback } from '../src/playback';

function fixture(continuous = true) {
  const visibility: boolean[] = [];
  let callback: VideoFrameRequestCallback | undefined;
  let metadata: { frameId: number } | undefined;
  let frame = 0;
  let copied = 0;
  const cover = {
    hidden: true,
    width: 640,
    height: 480,
    getContext: () => ({ drawImage: () => copied++ }),
  } as unknown as HTMLCanvasElement;
  const element = {
    videoWidth: 640,
    requestVideoFrameCallback(next: VideoFrameRequestCallback) {
      callback = next;
      return 1;
    },
    cancelVideoFrameCallback() {
      callback = undefined;
    },
  } as unknown as HTMLVideoElement;
  const track = { lookupFrameMetadata: () => metadata } as unknown as RemoteVideoTrack;
  const playback = new BrowserPlayback(
    async () => {},
    (value) => visibility.push(value),
    () => {},
    (message) => {
      throw new Error(message);
    },
    { state: 'running', currentTime: 0 } as AudioContext,
    continuous,
  );
  playback.attachVideo(track, element, cover);
  return {
    playback,
    visibility,
    cover,
    copied: () => copied,
    present(frameId?: number) {
      metadata = frameId === undefined ? undefined : { frameId };
      callback?.(frame * 40, { rtpTimestamp: ++frame } as VideoFrameCallbackMetadata);
    },
  };
}

test('continuous listening becomes ready without speech or timing metadata', async () => {
  const f = fixture();
  let ready = false;
  const waiting = f.playback.waitForVideo().then(() => {
    ready = true;
  });
  f.present();
  f.present();
  await Promise.resolve();
  expect(ready).toBe(false);
  f.present();
  await waiting;
  expect(f.visibility).toEqual([true, true, true]);
  await f.playback.waitForVideo();
  f.playback.close();
});

test('speech cancellation and recovery never hide continuous video', () => {
  const f = fixture();
  f.present(65536 + 1);
  f.playback.hold('status');
  f.playback.hold('media');
  const sequence = f.playback.interruptLocally();
  f.playback.interruptConfirmed(sequence, 0);
  f.present(65536 + 2); // Underlying decoder runs, but the held frame covers stale video.
  f.present(); // Missing speech timing metadata cannot hide the video either.
  f.playback.resume('status');
  f.playback.resume('media');
  f.present(2 * 65536); // Next epoch's listening frame.
  expect(f.visibility.length).toBeGreaterThan(3);
  expect(f.visibility.every(Boolean)).toBe(true);
  f.playback.close();
});

test('interruption holds the generated frame until fresh listening, never stale speaking', () => {
  const f = fixture();
  f.present(2 * 65536);
  f.present(2 * 65536 + 1);
  expect(f.copied()).toBe(1);
  f.playback.cancelled(2);
  expect(f.cover.hidden).toBe(false);
  f.present(2 * 65536 + 2);
  expect(f.copied()).toBe(1);
  expect(f.cover.hidden).toBe(false);
  f.present(3 * 65536); // Fresh silent video uses the cancellation epoch itself.
  expect(f.cover.hidden).toBe(true);
  expect(f.copied()).toBe(2);
  f.playback.close();
});

test('starting a reply does not freeze already silent listening video', () => {
  const f = fixture();
  f.present(65536);
  f.playback.cancelled(1);
  expect(f.cover.hidden).toBe(true);
  f.present(65536); // Older idle packet may arrive before the new epoch.
  expect(f.cover.hidden).toBe(true);
  f.present(3 * 65536);
  expect(f.cover.hidden).toBe(true);
  f.playback.close();
});

test('speaking video waits for decoded audio; mute/late frames cannot reopen it', () => {
  const f = fixture();
  let amplitude = 0;
  const gains: number[] = [];
  const el = { dataset: {} } as HTMLMediaElement;
  (f.playback as any).audio.set(el, {
    track: { detach() {}, setWebAudioPlugins() {} },
    gate: {
      gain: { setValueAtTime: (n: number) => gains.push(n), cancelScheduledValues() {} },
      disconnect() {},
    },
    meter: { disconnect() {} },
    readiness: {
      getFloatTimeDomainData: (data: Float32Array) => data.fill(amplitude),
      disconnect() {},
    },
  });
  f.present(65536);
  f.present(65536 + 1);
  expect(f.cover.hidden).toBe(false);
  expect(gains).not.toContain(1);
  amplitude = 0.05;
  f.present(65536 + 2);
  expect(f.cover.hidden).toBe(true);
  expect(gains.at(-1)).toBe(1);
  const seq = f.playback.interruptLocally();
  expect(gains.at(-1)).toBe(0);
  f.present(65536 + 3);
  expect(gains.at(-1)).toBe(0);
  expect(f.cover.hidden).toBe(false);
  f.playback.interruptConfirmed(seq, 1);
  f.present(2 * 65536);
  expect(f.cover.hidden).toBe(true);
  expect(gains.at(-1)).toBe(0);
  f.playback.close();
});

test('ending setup releases the pending video readiness wait', async () => {
  const f = fixture();
  const failed = expect(f.playback.waitForVideo()).rejects.toThrow('Call ended');
  f.playback.close();
  await failed;
  await expect(f.playback.waitForVideo()).rejects.toThrow('Call ended');
});

test('missing video times out instead of falsely completing connection', async () => {
  const f = fixture();
  await expect(f.playback.waitForVideo(10)).rejects.toThrow('Avatar video did not start');
  f.playback.close();
});

test('non-continuous playback retains its speech visibility gate', () => {
  const f = fixture(false);
  f.present(65536 + 1);
  f.playback.cancelled(0);
  expect(f.visibility).toEqual([true, false]);
  f.playback.close();
});
