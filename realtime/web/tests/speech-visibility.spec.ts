import { test, expect } from '@playwright/test';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/** Real VAD/barge-in regression. Synthetic microphone only; GPU/provider opt-in. */
test('generated video stays visible throughout microphone speech and barge-in', async ({
  page,
}) => {
  test.skip(process.env.RUN_QWEN_DEMO !== '1', 'Requires the prepared GPU demo');
  test.setTimeout(180000);
  const output = path.resolve(
    process.env.QWEN_TEST_OUTPUT || '../../runs/realtime/speech-visibility',
  );
  await mkdir(output, { recursive: true });
  const input = (await readFile(process.env.QWEN_TEST_INPUT_WAV!)).toString('base64');
  await page.addInitScript(() => {
    const state: any = { ctx: null, destination: null, samples: 0, violations: [], phase: 'idle' };
    (window as any).visibilityTest = state;
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (constraints?.audio && !constraints.video) {
        state.ctx ||= new AudioContext({ sampleRate: 48000 });
        state.destination ||= state.ctx.createMediaStreamDestination();
        await state.ctx.resume();
        return state.destination.stream;
      }
      return original(constraints);
    };
  });
  let session: any;
  const report: any = { status: 'running', stages: [] };
  const read = async () => {
    const response = await page.request.get('/api/sessions/' + session.id, {
      headers: { 'x-session-key': session.session_key },
    });
    expect(response.ok()).toBe(true);
    const state = await response.json();
    expect(state.ended).toBe(false);
    expect(state.metrics.error).toBeFalsy();
    return state.metrics;
  };
  const speak = async () => {
    await page.evaluate(async (encoded) => {
      const s = (window as any).visibilityTest;
      s.phase = 'microphone-speech';
      const source = s.ctx.createBufferSource();
      source.buffer = await s.ctx.decodeAudioData(
        Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0)).buffer,
      );
      source.connect(s.destination);
      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start();
      });
      s.phase = 'waiting-for-reply';
    }, input);
  };
  try {
    await page.goto('/');
    await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE!);
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByLabel('Model', { exact: true }).selectOption('avatarforever');
    await page.getByLabel('Conversation', { exact: true }).selectOption('voice_diagnostic');
    await page.getByLabel('Voice', { exact: true }).selectOption('aiden');
    await page.getByRole('button', { name: 'Close settings' }).click();
    const startRequested = Date.now();
    const created = page.waitForResponse(
      (r) => r.url().endsWith('/api/sessions') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Start call', exact: true }).click();
    session = await (await created).json();
    expect(session.id).toBeTruthy();
    const video = page.getByLabel('Generated avatar');
    await expect(video).toHaveClass('visible', { timeout: 60000 });
    await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeVisible();
    report.click_to_video_and_microphone_ready_ms = Date.now() - startRequested;
    await expect(page.locator('.avatar-image')).toHaveCount(0);
    await page.evaluate(() => {
      const s = (window as any).visibilityTest;
      const video = document.querySelector<HTMLVideoElement>(
        'video[aria-label="Generated avatar"]',
      )!;
      s.monitoring = true;
      s.motion = { listening: new Set(), speaking: new Set(), waiting: new Set() };
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 48;
      const context = canvas.getContext('2d', { willReadFrequently: true })!;
      let sampledAt = 0;
      function check() {
        if (!s.monitoring) return;
        s.samples++;
        const hidden =
          !video.classList.contains('visible') || getComputedStyle(video).opacity !== '1';
        const reference = document.querySelector('.avatar-image') !== null;
        if ((hidden || reference || !video.srcObject) && s.violations.length < 100)
          s.violations.push({ phase: s.phase, hidden, reference, attached: !!video.srcObject });
        if (video.readyState >= 2 && performance.now() - sampledAt >= 250) {
          sampledAt = performance.now();
          context.drawImage(video, 0, 0, 64, 48);
          const data = context.getImageData(0, 0, 64, 48).data;
          let hash = 2166136261;
          for (let i = 0; i < data.length; i += 4) hash = Math.imul(hash ^ data[i], 16777619);
          const key =
            s.phase === 'idle'
              ? 'listening'
              : s.phase === 'microphone-speech'
                ? 'speaking'
                : 'waiting';
          s.motion[key].add(hash);
        }
        requestAnimationFrame(check);
      }
      requestAnimationFrame(check);
    });
    const initial = await read();
    expect(initial.continuous_motion).toBe(true);
    report.stages.push({ name: 'listening', metrics: initial });
    // Confirm visible motion BEFORE sending any microphone audio.
    await page.waitForTimeout(4000);
    expect(
      await page.evaluate(() => (window as any).visibilityTest.motion.listening.size),
    ).toBeGreaterThan(3);
    await speak();
    await expect
      .poll(async () => (await read()).interruptions, { timeout: 10000 })
      .toBeGreaterThan(initial.interruptions);
    await expect
      .poll(() => page.locator('.audio-container audio').getAttribute('data-playback-open'), {
        timeout: 45000,
      })
      .toBe('true');
    const responding = await read();
    expect(responding.response_active).toBe(true);
    report.stages.push({ name: 'reply-before-barge-in', metrics: responding });
    // Speak over a real reply: this exercises provider VAD, not the Interrupt button.
    await speak();
    await expect
      .poll(async () => (await read()).interruptions, { timeout: 10000 })
      .toBeGreaterThan(responding.interruptions);
    report.stages.push({ name: 'after-spoken-interruption', metrics: await read() });
    await expect
      .poll(() => page.locator('.audio-container audio').getAttribute('data-playback-open'), {
        timeout: 45000,
      })
      .toBe('true');
    await expect.poll(async () => (await read()).response_active, { timeout: 60000 }).toBe(false);
    const finished = await read();
    await expect
      .poll(async () => (await read()).idle_motion_frames, { timeout: 10000 })
      .toBeGreaterThan(finished.idle_motion_frames + 25);
    report.stages.push({ name: 'returned-to-listening', metrics: await read() });
    report.browser = await page.evaluate(() => {
      const s = (window as any).visibilityTest;
      s.monitoring = false;
      return {
        samples: s.samples,
        violations: s.violations,
        distinct_pixel_samples: Object.fromEntries(
          Object.entries(s.motion).map(([k, v]) => [k, (v as Set<number>).size]),
        ),
      };
    });
    expect(report.browser.samples).toBeGreaterThan(100);
    expect(report.browser.violations).toEqual([]);
    expect(report.browser.distinct_pixel_samples.speaking).toBeGreaterThan(3);
    expect(report.browser.distinct_pixel_samples.waiting).toBeGreaterThan(3);
    await page.screenshot({ path: path.join(output, 'listening.png') });
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed';
    report.error = String(error);
    throw error;
  } finally {
    report.browser ||= await page
      .evaluate(() => {
        const s = (window as any).visibilityTest;
        if (!s) return null;
        s.monitoring = false;
        return { samples: s.samples, violations: s.violations };
      })
      .catch(() => null);
    if (session?.id)
      await page.request
        .delete('/api/sessions/' + session.id, {
          headers: { 'x-session-key': session.session_key },
        })
        .catch(() => {});
    await writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  }
});
