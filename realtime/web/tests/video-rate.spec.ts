import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

// Prepared GPU opt-in. Fixed replay audio; no microphone or voice-provider call.
test('WebRTC video is capped at 24 FPS without retiming native audio', async ({ page }) => {
  test.skip(process.env.RUN_AVATAR_RATE_TEST !== '1', 'Requires a prepared avatar GPU');
  test.setTimeout(120000);
  await page.addInitScript(() => {
    const Original = window.RTCPeerConnection;
    (window as any).rateTestPeers = [];
    window.RTCPeerConnection = class extends Original {
      constructor(config?: RTCConfiguration) {
        super(config);
        (window as any).rateTestPeers.push(this);
      }
    };
  });
  let session: any;
  const snapshot = () =>
    page.evaluate(async () => {
      const result: any[] = [];
      for (const peer of (window as any).rateTestPeers as RTCPeerConnection[]) {
        if (peer.connectionState === 'closed') continue;
        const stats = await peer.getStats();
        stats.forEach((s) => {
          if (s.type === 'inbound-rtp') {
            const codec = stats.get(s.codecId);
            result.push({
              id: s.id,
              kind: s.kind,
              timestamp: s.timestamp,
              framesDecoded: s.framesDecoded,
              packetsReceived: s.packetsReceived,
              clockRate: codec?.clockRate,
              frameWidth: s.frameWidth,
              frameHeight: s.frameHeight,
            });
          }
        });
      }
      return result;
    });
  try {
    await page.goto('/');
    await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE!);
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByLabel('Model', { exact: true }).selectOption('avatarforever');
    await page.getByLabel('Conversation', { exact: true }).selectOption('diagnostic');
    await page.getByRole('button', { name: 'Close settings' }).click();
    const created = page.waitForResponse(
      (r) => r.url().endsWith('/api/sessions') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Start call', exact: true }).click();
    session = await (await created).json();
    expect(session.id).toBeTruthy();
    expect(session.metrics.video_max_fps).toBe(24);
    expect(session.metrics.native_model_fps).toBe(25);
    await expect
      .poll(async () => (await snapshot()).find((s) => s.kind === 'video')?.framesDecoded || 0, {
        timeout: 30000,
      })
      .toBeGreaterThan(50);
    const reply = await page.request.post(`/api/sessions/${session.id}/play`, {
      headers: { 'X-Session-Key': session.session_key },
      data: { scenario_id: 's01_greeting' },
    });
    expect(reply.ok()).toBe(true);
    const before = await snapshot();
    await page.evaluate(() => {
      const video = document.querySelector<HTMLVideoElement>(
        'video[aria-label="Generated avatar"]',
      )!;
      const frames: number[] = [];
      (window as any).ratePresentation = { frames, running: true };
      const collect = (_now: number, metadata: VideoFrameCallbackMetadata) => {
        if (!(window as any).ratePresentation.running) return;
        frames.push(metadata.expectedDisplayTime);
        video.requestVideoFrameCallback(collect);
      };
      video.requestVideoFrameCallback(collect);
    });
    await page.waitForTimeout(10000);
    const after = await snapshot();
    const presentation = await page.evaluate(() => {
      const state = (window as any).ratePresentation;
      state.running = false;
      const gaps = state.frames
        .slice(1)
        .map((t: number, i: number) => t - state.frames[i])
        .sort((a: number, b: number) => a - b);
      return {
        frame_intervals: gaps.length,
        median_ms: gaps[Math.floor(gaps.length / 2)],
        p95_ms: gaps[Math.floor(gaps.length * 0.95)],
        max_ms: gaps.at(-1),
      };
    });
    const video = after.find((s) => s.kind === 'video');
    const first = before.find((s) => s.id === video.id);
    const seconds = (video.timestamp - first.timestamp) / 1000;
    const frames = video.framesDecoded - first.framesDecoded;
    expect([video.frameWidth, video.frameHeight]).toEqual([640, 480]);
    expect(frames).toBeGreaterThan(100);
    // Two frames of measurement-boundary tolerance; a 25 FPS stream fails.
    expect(frames).toBeLessThanOrEqual(Math.ceil(24 * seconds) + 2);
    const audio = after.find((s) => s.kind === 'audio');
    expect(audio.clockRate).toBe(48000);
    expect(audio.packetsReceived).toBeGreaterThan(
      before.find((s) => s.id === audio.id).packetsReceived,
    );
    const output = path.resolve('../../runs/realtime/fps24');
    await mkdir(output, { recursive: true });
    await writeFile(
      path.join(output, 'report.json'),
      JSON.stringify(
        {
          status: 'passed',
          configured_max_fps: 24,
          native_model_fps: 25,
          measured_received_fps: frames / seconds,
          frames,
          seconds,
          geometry: [video.frameWidth, video.frameHeight],
          audio_clock_rate: audio.clockRate,
          presentation,
          scope:
            'Receiver stats with fixed audio; does not establish motion continuity or lower response latency.',
        },
        null,
        2,
      ),
    );
  } finally {
    if (session?.id)
      await page.request
        .delete(`/api/sessions/${session.id}`, {
          headers: { 'X-Session-Key': session.session_key },
        })
        .catch(() => {});
  }
});
