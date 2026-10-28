import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

// Explicit opt-in: consumes GPU time and paid provider requests. The microphone
// is a supplied synthetic WAV, never the operator's physical microphone.
test('real voice reaches AvatarForever and returns interruptible 4:3 media', async ({
  browser,
}) => {
  test.skip(
    process.env.RUN_REAL_AVATAR !== '1',
    'Requires configured GPU/provider and explicit opt-in',
  );
  test.setTimeout(420000);
  if (!process.env.REAL_AVATAR_MIC_WAV)
    throw new Error('Set REAL_AVATAR_MIC_WAV to a synthetic test utterance');
  // Launch separately so the normal diagnostic tests never use paid inputs.
  const actual = await browser.browserType().launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
    ignoreDefaultArgs: ['--mute-audio'],
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--autoplay-policy=no-user-gesture-required',
      `--use-file-for-fake-audio-capture=${path.resolve(process.env.REAL_AVATAR_MIC_WAV)}%noloop`,
    ],
  });
  const context = await actual.newContext({
    baseURL: process.env.DEMO_TEST_URL,
    viewport: { width: 1440, height: 1050 },
  });
  const page = await context.newPage();
  let session: any;
  page.on('response', async (reply) => {
    if (
      reply.url().endsWith('/api/sessions') &&
      reply.request().method() === 'POST' &&
      reply.status() === 201
    )
      session = await reply.json();
  });
  const avatar = process.env.REAL_AVATAR_ID || 'ref01';
  if (!['ref01', 'ref02', 'ref03'].includes(avatar)) throw new Error('Unsupported test avatar');
  const output = `../../runs/realtime/real-avatar-browser/${avatar}`;
  await mkdir(output, { recursive: true });
  const started = Date.now();
  try {
    await page.goto('/');
    if (process.env.DEMO_TEST_ACCESS_CODE) {
      await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE);
      await page.getByRole('button', { name: 'Continue', exact: true }).click();
    }
    await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled();
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page
      .getByRole('button', { name: `Select Character ${avatar.slice(-2)}`, exact: true })
      .click();
    await expect(page.getByLabel('Model', { exact: true })).toHaveValue('avatarforever');
    await expect(page.getByLabel('Conversation', { exact: true })).toHaveValue('voice_diagnostic');
    await page.getByRole('button', { name: 'Close settings' }).click();
    await page.getByRole('button', { name: 'Turn on camera', exact: true }).click();
    await page.getByRole('button', { name: 'Start call' }).click();
    await expect(page.getByRole('button', { name: 'End call' })).toBeVisible({ timeout: 340000 });
    await expect.poll(() => Boolean(session), { timeout: 15000 }).toBe(true);
    const read = async () =>
      (
        await page.request.get(`/api/sessions/${session.id}`, {
          headers: { 'X-Session-Key': session.session_key },
        })
      ).json();
    const connected = Date.now();
    await expect
      .poll(
        async () => {
          const state = await read();
          if (state.ended || state.metrics.error)
            throw new Error(
              `Call ended before generation: ${state.metrics.error || (await page.locator('[role=alert]').allTextContents())}`,
            );
          return state.metrics.generated_frames;
        },
        { timeout: 60000, intervals: [500] },
      )
      .toBeGreaterThan(25);
    const firstMotion = Date.now();
    await page.getByRole('button', { name: 'Mute microphone', exact: true }).click();
    const remote = page.getByLabel('Generated avatar', { exact: true });
    await expect
      .poll(() => remote.evaluate((v: HTMLVideoElement) => [v.videoWidth, v.videoHeight]))
      .toEqual([640, 480]);
    await expect
      .poll(
        () =>
          page
            .locator('audio')
            .evaluateAll((els: HTMLAudioElement[]) =>
              els.some((e) => e.dataset.playbackOpen === 'true' && !e.paused && e.currentTime > 0),
            ),
        { timeout: 15000 },
      )
      .toBe(true);
    await page.screenshot({ path: `${output}/live-call.png`, fullPage: true });
    const before = await read();
    expect(before.metrics.voice_model).toBeTruthy();
    expect(before.metrics.effective_voice).toBe(session.voice_selection.profile.voice);
    expect(before.metrics.error).toBeFalsy();
    expect(before.metrics.distinct_generated_frames).toBeGreaterThan(0);
    expect(before.metrics.distinct_generated_frames).toBeLessThanOrEqual(
      before.metrics.generated_frames,
    );
    // A transient control failure must retain the actual provider session/voice,
    // while stopping browser output until authoritative status returns.
    const sessionPath = `/api/sessions/${session.id}`;
    await page.route('**' + sessionPath, (route) =>
      route.request().method() === 'GET' ? route.abort('internetdisconnected') : route.continue(),
    );
    await expect(page.getByRole('status', { name: 'Connection recovery' })).toBeVisible();
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.every((e) => e.dataset.playbackOpen === 'false'),
          ),
      )
      .toBe(true);
    await page.waitForTimeout(1500);
    await page.unrouteAll({ behavior: 'wait' });
    await expect(page.getByRole('status', { name: 'Connection recovery' })).not.toBeVisible();
    const recovered = await read();
    expect(recovered.id).toBe(session.id);
    expect(recovered.ended).toBe(false);
    expect(recovered.metrics.effective_voice).toBe(before.metrics.effective_voice);
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.locator('summary').filter({ hasText: 'Connection details' }).click();
    await expect(page.getByText('Distinct generated video', { exact: true })).toBeVisible();
    await page.screenshot({ path: `${output}/connection-details.png`, fullPage: true });
    await page.getByRole('button', { name: 'Close settings' }).click();
    const interrupt = Date.now();
    await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.every((e) => e.dataset.playbackOpen === 'false'),
          ),
      )
      .toBe(true);
    const stopped = Date.now();
    await page.getByRole('button', { name: 'End call' }).click();
    await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled({ timeout: 15000 });
    const after = await read();
    await writeFile(
      `${output}/report.json`,
      JSON.stringify(
        {
          status: 'passed',
          geometry: [640, 480],
          setup_ms: connected - started,
          synthetic_utterance_to_observed_motion_ms: firstMotion - connected,
          same_session_control_recovery: true,
          voice_after_recovery: recovered.metrics.effective_voice,
          interrupt_observed_ms: stopped - interrupt,
          before: before.metrics,
          after: after.metrics,
          note: 'Automated synthetic microphone; observed browser media and audio graph, not physical speaker timing.',
        },
        null,
        2,
      ),
    );
  } catch (error) {
    const state = session
      ? await (
          await page.request.get(`/api/sessions/${session.id}`, {
            headers: { 'X-Session-Key': session.session_key },
          })
        )
          .json()
          .catch(() => null)
      : null;
    await writeFile(
      `${output}/failure.json`,
      JSON.stringify(
        {
          error: String(error),
          metrics: state?.metrics,
          alerts: await page.locator('[role=alert]').allTextContents(),
        },
        null,
        2,
      ),
    );
    await page.screenshot({ path: `${output}/failure.png`, fullPage: true });
    throw error;
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    if (session)
      await page.request
        .delete(`/api/sessions/${session.id}`, {
          headers: { 'X-Session-Key': session.session_key },
        })
        .catch(() => {});
    await actual.close();
  }
});
