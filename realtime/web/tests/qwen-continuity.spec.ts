import { test, expect } from '@playwright/test';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/** Explicit GPU/provider opt-in. Uses synthetic audio, never the user's microphone. */
test('Qwen speech, continuous idle, interruption and a second reply', async ({ page }) => {
  test.skip(process.env.RUN_QWEN_DEMO !== '1', 'Requires a prepared H100 and provider credentials');
  test.setTimeout(300000);
  const output = path.resolve(
    process.env.QWEN_TEST_OUTPUT || '../../runs/realtime/qwen-continuity',
  );
  await mkdir(output, { recursive: true });
  const input = await readFile(process.env.QWEN_TEST_INPUT_WAV!);
  await page.addInitScript(() => {
    const state: any = {
      ctx: null,
      destination: null,
      gaps: [],
      frames: 0,
      last: null,
      referenceFallbacks: 0,
      monitoring: false,
    };
    (window as any).qwenTest = state;
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
  const report: any = { stages: [], status: 'running' };
  const spokenFrames = (m: any) => m.frames_published - m.idle_motion_frames - m.held_frames;
  let recordingStarted = 0;
  const stage = async (name: string, metrics: unknown) => {
    report.stages.push({ name, recordingSeconds: (Date.now() - recordingStarted) / 1000, metrics });
    await writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    console.log('Call stage:', name);
  };
  const read = async () => {
    const response = await page.request.get('/api/sessions/' + session.id, {
      headers: { 'x-session-key': session.session_key },
      timeout: 10000,
    });
    expect(response.ok()).toBeTruthy();
    const state = await response.json();
    if (state.metrics?.error) throw new Error(state.metrics.error);
    return state;
  };
  const speak = async () => {
    await page.evaluate(async (encoded) => {
      const s = (window as any).qwenTest;
      const buffer = await s.ctx.decodeAudioData(
        Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0)).buffer,
      );
      const source = s.ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(s.destination);
      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start();
      });
    }, input.toString('base64'));
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
    const created = page.waitForResponse(
      (r) => r.url().endsWith('/api/sessions') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Start call', exact: true }).click();
    session = await (await created).json();
    expect(session.id).toBeTruthy();
    await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeVisible({
      timeout: 90000,
    });
    await expect
      .poll(
        () => page.getByLabel('Generated avatar').evaluate((v: HTMLVideoElement) => v.videoWidth),
        { timeout: 30000 },
      )
      .toBe(640);
    await page.evaluate(() => {
      const s = (window as any).qwenTest;
      const video = document.querySelector<HTMLVideoElement>(
        'video[aria-label="Generated avatar"]',
      )!;
      const audio = document.querySelector<HTMLAudioElement>('.audio-container audio');
      const stream = new MediaStream([
        ...(video.srcObject as MediaStream).getVideoTracks(),
        ...((audio?.srcObject as MediaStream | null)?.getAudioTracks() || []),
      ]);
      s.chunks = [];
      s.recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8,opus' });
      s.recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data.size) s.chunks.push(e.data);
      };
      s.recorder.start(1000);
      s.monitoring = true;
      function visible() {
        if (!s.monitoring) return;
        if (!video.classList.contains('visible') || document.querySelector('.avatar-image'))
          s.referenceFallbacks++;
        requestAnimationFrame(visible);
      }
      requestAnimationFrame(visible);
      function frame(now: number) {
        if (s.last !== null) s.gaps.push(now - s.last);
        s.last = now;
        s.frames++;
        video.requestVideoFrameCallback(frame);
      }
      video.requestVideoFrameCallback(frame);
    });
    recordingStarted = Date.now();
    await page.waitForTimeout(5000);
    let state = await read();
    expect(state.metrics.continuous_motion).toBeTruthy();
    expect(state.metrics.idle_motion_frames).toBeGreaterThan(50);
    await stage('initial-idle', state.metrics);
    await expect(page.getByLabel('Generated avatar')).toHaveClass('visible');
    const initialFrames = spokenFrames(state.metrics);
    await speak();
    await expect
      .poll(async () => spokenFrames((await read()).metrics), { timeout: 45000 })
      .toBeGreaterThan(initialFrames + 20);
    await expect
      .poll(async () => (await read()).metrics.response_active, { timeout: 60000 })
      .toBe(false);
    state = await read();
    expect(state.metrics.effective_voice).toBe('aiden');
    expect(state.metrics.voice_mode).toBe('openai_text_qwen_tts');
    await stage('first-reply', state.metrics);
    const idleBefore = state.metrics.idle_motion_frames;
    await page.waitForTimeout(5000);
    expect((await read()).metrics.idle_motion_frames).toBeGreaterThan(idleBefore + 50);
    await stage('between-replies', (await read()).metrics);
    await expect(page.getByLabel('Generated avatar')).toHaveClass('visible');
    const beforeSecond = spokenFrames((await read()).metrics);
    await speak();
    await expect
      .poll(async () => spokenFrames((await read()).metrics), { timeout: 45000 })
      .toBeGreaterThan(beforeSecond + 10);
    await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
    await page.waitForTimeout(2500);
    state = await read();
    expect(state.metrics.response_active).toBe(false);
    await stage('interrupted', state.metrics);
    await expect(page.getByLabel('Generated avatar')).toHaveClass('visible');
    const beforeThird = spokenFrames(state.metrics);
    await speak();
    await expect
      .poll(async () => spokenFrames((await read()).metrics), { timeout: 45000 })
      .toBeGreaterThan(beforeThird + 20);
    await expect
      .poll(async () => (await read()).metrics.response_active, { timeout: 60000 })
      .toBe(false);
    await page.waitForTimeout(4000);
    await stage('recovered-and-idle', (await read()).metrics);
    await page.screenshot({ path: path.join(output, 'demo.png') });
    expect(
      (await read()).metrics.speech_underflow_frames,
      'Speech must not be chopped by late video chunks',
    ).toBe(0);
    expect(await page.evaluate(() => (window as any).qwenTest.referenceFallbacks)).toBe(0);
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed';
    report.error = String(error);
    if (session?.id) report.last = await read().catch(() => null);
    throw error;
  } finally {
    const recording = await page
      .evaluate(async () => {
        const s = (window as any).qwenTest;
        s.monitoring = false;
        if (!s?.recorder || s.recorder.state === 'inactive') return null;
        await new Promise<void>((resolve) => {
          s.recorder.onstop = () => resolve();
          s.recorder.stop();
        });
        const blob = new Blob(s.chunks, { type: 'video/webm' });
        const encoded = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        });
        return {
          encoded: encoded.split(',')[1],
          frames: s.frames,
          maxFrameGapMs: Math.max(...s.gaps),
          referenceFallbacks: s.referenceFallbacks,
        };
      })
      .catch(() => null);
    if (recording) {
      await writeFile(path.join(output, 'call.webm'), Buffer.from(recording.encoded, 'base64'));
      report.browser = {
        frames: recording.frames,
        maxFrameGapMs: recording.maxFrameGapMs,
        referenceFallbacks: recording.referenceFallbacks,
      };
    }
    if (session?.id)
      await page.request
        .delete('/api/sessions/' + session.id, {
          headers: { 'x-session-key': session.session_key },
        })
        .catch(() => {});
    await writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  }
});
