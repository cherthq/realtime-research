import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import nodePath from 'node:path';

test.beforeEach(async ({ page }) => {
  if (process.env.DEMO_TEST_FORCE_TCP)
    await page.addInitScript(() => {
      const Original = window.RTCPeerConnection;
      (window as any).__demoTestPeerConnections = [];
      window.RTCPeerConnection = class extends Original {
        constructor(config?: RTCConfiguration) {
          super(config);
          (window as any).__demoTestPeerConnections.push(this);
        }
      };
    });
  if (!process.env.DEMO_TEST_ACCESS_CODE) return;
  await page.goto('/');
  await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE);
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start call' })).toBeVisible();
});

test('private demo unlock survives reload and logout revokes access', async ({ page, context }) => {
  test.skip(!process.env.DEMO_TEST_ACCESS_CODE, 'Run against the private host profile');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Start call' })).toBeVisible();
  const cookie = (await context.cookies()).find((cookie) => cookie.name === 'chert_demo_access');
  expect(cookie?.httpOnly).toBe(true);
  expect(cookie?.sameSite).toBe('Strict');
  expect(await page.evaluate(() => document.cookie)).not.toContain('chert_demo_access');
  await settings(page);
  await page.getByRole('button', { name: 'Lock demo', exact: true }).click();
  await expect(page.getByLabel('Access code', { exact: true })).toBeVisible();
  await page.getByLabel('Access code', { exact: true }).fill('incorrect');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Incorrect access code');
  expect((await page.request.get('/api/avatars')).status()).toBe(401);
});

async function settings(page: import('@playwright/test').Page) {
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
}

test('two-panel call: local camera, real remote media, interruption and cleanup', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled();
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await page.getByAltText('Avatar reference').evaluate((image: HTMLImageElement) => image.decode());
  await page.screenshot({ path: '../../runs/realtime/local-demo/call-ready.png', fullPage: true });
  await page.getByRole('button', { name: 'Turn on camera', exact: true }).click();
  const local = page.getByLabel('Your camera', { exact: true });
  const remote = page.getByLabel('Generated avatar', { exact: true });
  await expect
    .poll(() => local.evaluate((element: HTMLVideoElement) => element.videoWidth))
    .toBeGreaterThan(0);
  await local.evaluate((element: HTMLVideoElement) => {
    (window as any).testCameraStream = element.srcObject;
  });
  await page.getByRole('button', { name: 'Start call' }).click();
  try {
    await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
    await expect
      .poll(() =>
        remote.evaluate((element: HTMLVideoElement) => [element.videoWidth, element.videoHeight]),
      )
      .toEqual([640, 480]);
    if (process.env.DEMO_TEST_FORCE_TCP) {
      const protocols = await page.evaluate(async () => {
        const result: string[] = [];
        for (const pc of (window as any).__demoTestPeerConnections as RTCPeerConnection[]) {
          const report = await pc.getStats();
          report.forEach((stat) => {
            if (stat.type === 'transport' && stat.selectedCandidatePairId) {
              const pair = report.get(stat.selectedCandidatePairId);
              result.push(report.get(pair.remoteCandidateId).protocol);
            }
          });
        }
        return result;
      });
      expect(protocols.length).toBeGreaterThan(0);
      expect(new Set(protocols)).toEqual(new Set(['tcp']));
    }
    await settings(page);
    await page.getByLabel('Test reply', { exact: true }).selectOption('s01_greeting');
    await page.getByRole('button', { name: 'Play test reply' }).click();
    await expect(page.getByRole('dialog')).not.toBeVisible();
    await expect(page.getByRole('status')).toContainText('Speaking');
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.some(
              (el) => !el.paused && el.dataset.playbackOpen === 'true' && el.currentTime > 0,
            ),
          ),
      )
      .toBe(true);
    await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.every((el) => el.dataset.playbackOpen === 'false'),
          ),
      )
      .toBe(true);
    await expect(page.getByRole('status')).toContainText('Connected');
    await page.getByRole('button', { name: 'Enable microphone' }).click();
    await expect(page.getByRole('button', { name: 'Mute microphone' })).toBeVisible();
    await settings(page);
    await page.getByText('Connection details', { exact: true }).click();
    await expect.poll(() => page.getByTestId('playback-reports').innerText()).not.toBe('0');
    await expect.poll(() => page.getByTestId('mic-packets').innerText()).not.toBe('0');
    await page.getByRole('button', { name: 'Close settings' }).click();
    await page.screenshot({
      path: '../../runs/realtime/local-demo/call-desktop.png',
      fullPage: true,
    });
  } finally {
    if (await page.getByRole('button', { name: 'End call' }).isVisible())
      await page.getByRole('button', { name: 'End call' }).click();
  }
  await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled();
  expect(
    await page.evaluate(() =>
      (window as any).testCameraStream
        .getTracks()
        .every((track: MediaStreamTrack) => track.readyState === 'ended'),
    ),
  ).toBe(true);
  await expect(local).not.toHaveClass(/visible/);
  await settings(page);
  await page.getByRole('button', { name: 'Select Character 02' }).click();
  await page.getByRole('button', { name: 'Close settings' }).click();
  await page.getByRole('button', { name: 'Start call' }).click();
  await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
  await page.getByRole('button', { name: 'End call' }).click();
  expect(errors).toEqual([]);
});

test('mobile shows both panels with settings hidden', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByAltText('Avatar reference').evaluate((image: HTMLImageElement) => image.decode());
  await expect(page.getByRole('button', { name: 'Start call' })).toBeVisible();
  await expect(page.getByLabel('Video call')).toBeVisible();
  await expect(page.getByRole('dialog')).not.toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: '../../runs/realtime/local-demo/call-mobile.png', fullPage: true });
  await settings(page);
  await expect(page.getByLabel('Model', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
});

test('frame feedback and cancellation survive repeated responses and slow HTTP acknowledgement', async ({
  page,
}) => {
  await page.goto('/');
  const created = page.waitForResponse(
    (reply) => reply.url().endsWith('/api/sessions') && reply.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Start call' }).click();
  const session = await (await created).json();
  const path = `/api/sessions/${session.id}`;
  const headers = { 'X-Session-Key': session.session_key };
  // Control-state check; transport-timing.spec.ts separately measures audio samples.
  const audioEnabled = () =>
    page
      .locator('audio')
      .evaluateAll((els: HTMLAudioElement[]) =>
        els.some((el) => el.dataset.playbackOpen === 'true' && !el.paused),
      );
  const stats: unknown[] = [];
  try {
    await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
    for (let turn = 0; turn < 3; turn++) {
      expect(
        (
          await page.request.post(path + '/play', { headers, data: { scenario_id: 'sync_marker' } })
        ).ok(),
      ).toBe(true);
      await expect.poll(audioEnabled).toBe(true);
      await expect
        .poll(
          async () =>
            (await (await page.request.get(path, { headers })).json()).metrics.browser_playback
              .played_offset_ms,
        )
        .toBeGreaterThan(0);
      if (turn === 1) {
        await page.route('**' + path + '/interrupt', async (route) => {
          const response = await route.fetch();
          await new Promise((resolve) => setTimeout(resolve, 600));
          await route.fulfill({ response });
        });
        // The output gate closes before the delayed HTTP result returns.
        const timing = await page.evaluate(async () => {
          const audio = document.querySelector('audio')!;
          return new Promise<number>((resolve) => {
            const mark = performance.now();
            const observer = new MutationObserver(() => {
              if (audio.dataset.playbackOpen === 'false') {
                observer.disconnect();
                resolve(performance.now() - mark);
              }
            });
            observer.observe(audio, { attributes: true, attributeFilter: ['data-playback-open'] });
            (document.querySelector('[aria-label="Interrupt"]') as HTMLButtonElement).click();
          });
        });
        expect(timing).toBeLessThan(300);
      } else {
        // A server-side event also gates playback without pressing the UI button.
        expect((await page.request.post(path + '/interrupt', { headers })).ok()).toBe(true);
      }
      await expect.poll(audioEnabled).toBe(false);
      await expect
        .poll(
          async () =>
            (await (await page.request.get(path, { headers })).json()).metrics.browser_playback
              .cancel_ack_roundtrip_ms,
        )
        .not.toBeNull();
      const state = await (await page.request.get(path, { headers })).json();
      expect(state.metrics.browser_playback.cancel_ack_roundtrip_ms).toBeLessThan(300);
      expect(state.metrics.browser_playback.offset_source).toBe('browser_video_with_audio_playing');
      stats.push(state.metrics.browser_playback);
    }
    const report = JSON.stringify(
      { scope: 'Local browser frame/mute receipts, not physical-speaker latency', turns: stats },
      null,
      2,
    );
    await test
      .info()
      .attach('browser-cancellation.json', { body: report, contentType: 'application/json' });
    const directory = nodePath.resolve('../../runs/realtime/browser-tests');
    await mkdir(directory, { recursive: true });
    await writeFile(
      nodePath.join(
        directory,
        process.env.DEMO_TEST_FORCE_TCP
          ? 'private-tcp-cancellation.json'
          : 'local-cancellation.json',
      ),
      report,
    );
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    await page.getByRole('button', { name: 'End call' }).click();
  }
});

test('camera denial leaves the call usable and camera retries release their tracks', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    let denied = false;
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (constraints?.video && !denied) {
        denied = true;
        throw new DOMException('Denied', 'NotAllowedError');
      }
      return original(constraints);
    };
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Turn on camera', exact: true }).click();
  await expect(page.getByText('Camera access is blocked.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled();
  await page.getByRole('button', { name: 'Turn on camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Turn camera off' })).toBeVisible();
  await page.getByLabel('Your camera', { exact: true }).evaluate((element: HTMLVideoElement) => {
    (window as any).testCameraStream = element.srcObject;
  });
  await page.getByRole('button', { name: 'Turn camera off' }).click();
  expect(
    await page.evaluate(() =>
      (window as any).testCameraStream
        .getTracks()
        .every((track: MediaStreamTrack) => track.readyState === 'ended'),
    ),
  ).toBe(true);
});

test('cancelling a pending camera request also stops a late permission result', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await original(constraints);
      (window as any).lateCameraStream = stream;
      await new Promise<void>((resolve) => {
        (window as any).releaseCameraPermission = resolve;
      });
      return stream;
    };
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Turn on camera', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => !!(window as any).releaseCameraPermission))
    .toBe(true);
  await page.getByRole('button', { name: 'Cancel camera request' }).click();
  await page.evaluate(() => (window as any).releaseCameraPermission());
  await expect
    .poll(() =>
      page.evaluate(() =>
        (window as any).lateCameraStream
          .getTracks()
          .every((track: MediaStreamTrack) => track.readyState === 'ended'),
      ),
    )
    .toBe(true);
  await expect(page.getByLabel('Your camera', { exact: true })).not.toHaveClass(/visible/);
});
