import { test, expect } from '@playwright/test';

async function startDiagnostic(page: import('@playwright/test').Page, waitForMedia = true) {
  await page.goto('/');
  await expect(
    page
      .getByLabel('Access code', { exact: true })
      .or(page.getByRole('button', { name: 'Start call' })),
  ).toBeVisible();
  if (
    process.env.DEMO_TEST_ACCESS_CODE &&
    (await page.getByLabel('Access code', { exact: true }).isVisible())
  ) {
    await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE);
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
  }
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByLabel('Model', { exact: true }).selectOption('mock');
  await page.getByLabel('Conversation', { exact: true }).selectOption('diagnostic');
  await page.getByRole('button', { name: 'Close settings' }).click();
  const created = page.waitForResponse(
    (r) => r.url().endsWith('/api/sessions') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Start call' }).click();
  const session = await (await created).json();
  await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
  // End is available during connection so users can cancel setup. Wait for
  // the actual media connection before injecting a control/autoplay fault.
  if (waitForMedia) {
    await expect(page.getByRole('button', { name: 'Enable microphone', exact: true })).toBeEnabled({
      timeout: 20000,
    });
    await expect(page.locator('audio')).toHaveCount(1, { timeout: 10000 });
  }
  return { path: `/api/sessions/${session.id}`, headers: { 'X-Session-Key': session.session_key } };
}

test('blocked audio can recover without reopening an interrupted response', async ({ page }) => {
  await page.addInitScript(() => {
    const play = HTMLMediaElement.prototype.play;
    (window as any).blockTestAudio = true;
    HTMLMediaElement.prototype.play = function () {
      return this.tagName === 'AUDIO' && (window as any).blockTestAudio
        ? Promise.reject(new DOMException('Test autoplay restriction', 'NotAllowedError'))
        : play.call(this);
    };
  });
  const session = await startDiagnostic(page);
  try {
    // Delay the actual cancellation request, not just its response. Frames from
    // a response newer than the last displayed frame can arrive during this gap.
    await page.route('**' + session.path + '/interrupt', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      await route.continue();
    });
    await expect(page.getByRole('button', { name: 'Enable sound', exact: true })).toBeVisible();
    expect(
      (
        await page.request.post(session.path + '/play', {
          headers: session.headers,
          data: { scenario_id: 'interrupt_marker' },
        })
      ).ok(),
    ).toBe(true);
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
    await page.evaluate(() => {
      (window as any).blockTestAudio = false;
    });
    await page.getByRole('button', { name: 'Enable sound', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Enable sound', exact: true })).not.toBeVisible();
    expect(
      await page
        .locator('audio')
        .evaluateAll((els: HTMLAudioElement[]) =>
          els.every((e) => e.dataset.playbackOpen === 'false'),
        ),
    ).toBe(true);
    await expect
      .poll(
        async () =>
          (await (await page.request.get(session.path, { headers: session.headers })).json())
            .metrics.interruptions,
      )
      .toBeGreaterThan(0);
    await page.unrouteAll({ behavior: 'wait' });
    expect(
      (
        await page.request.post(session.path + '/play', {
          headers: session.headers,
          data: { scenario_id: 'sync_marker' },
        })
      ).ok(),
    ).toBe(true);
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.some((e) => !e.paused && e.dataset.playbackOpen === 'true'),
          ),
      )
      .toBe(true);
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    await page.request.delete(session.path, { headers: session.headers });
  }
});

test('ending during media setup cannot close a later call', async ({ page }) => {
  await page.addInitScript(() => {
    const original = RTCPeerConnection.prototype.createOffer as (
      options?: RTCOfferOptions,
    ) => Promise<RTCSessionDescriptionInit>;
    let first = true;
    RTCPeerConnection.prototype.createOffer = async function (
      this: RTCPeerConnection,
      options?: RTCOfferOptions,
    ) {
      if (first) {
        first = false;
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
      return original.call(this, options);
    } as typeof RTCPeerConnection.prototype.createOffer;
  });
  const first = await startDiagnostic(page, false);
  try {
    await page.getByRole('button', { name: 'End call' }).click();
    await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled({ timeout: 15000 });
    const second = await startDiagnostic(page);
    try {
      expect(second.path).not.toBe(first.path);
      expect(
        (
          await page.request.post(second.path + '/play', {
            headers: second.headers,
            data: { scenario_id: 'sync_marker' },
          })
        ).ok(),
      ).toBe(true);
      await expect
        .poll(() =>
          page
            .locator('audio')
            .evaluateAll((els: HTMLAudioElement[]) =>
              els.some((e) => e.dataset.playbackOpen === 'true'),
            ),
        )
        .toBe(true);
      await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
    } finally {
      await page.request.delete(second.path, { headers: second.headers });
    }
  } finally {
    await page.request.delete(first.path, { headers: first.headers });
  }
});

test('microphone denial is recoverable within the same call', async ({ page }) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    let first = true;
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (constraints?.audio && first) {
        first = false;
        throw new DOMException('Test microphone denial', 'NotAllowedError');
      }
      return original(constraints);
    };
  });
  const session = await startDiagnostic(page);
  try {
    await page.getByRole('button', { name: 'Enable microphone', exact: true }).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(
      page.getByRole('button', { name: 'Enable microphone', exact: true }),
    ).toBeEnabled();
    await page.getByRole('button', { name: 'Enable microphone', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeVisible();
    await expect
      .poll(
        async () =>
          (await (await page.request.get(session.path, { headers: session.headers })).json())
            .metrics.mic_packets,
      )
      .toBeGreaterThan(0);
  } finally {
    await page.request.delete(session.path, { headers: session.headers });
  }
});

test('brief control loss pauses playback and recovers the same call', async ({ page }) => {
  const session = await startDiagnostic(page);
  try {
    expect(
      (
        await page.request.post(session.path + '/play', {
          headers: session.headers,
          data: { scenario_id: 'sync_marker' },
        })
      ).ok(),
    ).toBe(true);
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.some((e) => e.dataset.playbackOpen === 'true'),
          ),
      )
      .toBe(true);
    await page.route('**' + session.path, (route) =>
      route.request().method() === 'GET' ? route.abort('internetdisconnected') : route.continue(),
    );
    await expect(page.getByRole('status', { name: 'Connection recovery' })).toContainText(
      'Playback is paused',
    );
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.every((e) => e.dataset.playbackOpen === 'false'),
          ),
      )
      .toBe(true);
    await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
    await page.waitForTimeout(1500);
    expect(
      await page
        .locator('audio')
        .evaluateAll((els: HTMLAudioElement[]) =>
          els.every((e) => e.dataset.playbackOpen === 'false'),
        ),
    ).toBe(true);
    // Cancel during the outage. The recovered status must prevent stale frames
    // from opening playback, while a new response can still play on this call.
    await page.request.post(session.path + '/interrupt', { headers: session.headers });
    await page.unrouteAll({ behavior: 'wait' });
    await expect(page.getByRole('status', { name: 'Connection recovery' })).not.toBeVisible();
    const resumed = await (
      await page.request.get(session.path, { headers: session.headers })
    ).json();
    expect(resumed.ended).toBe(false);
    expect(resumed.id).toBe(session.path.split('/').pop());
    expect(
      (
        await page.request.post(session.path + '/play', {
          headers: session.headers,
          data: { scenario_id: 'sync_marker' },
        })
      ).ok(),
    ).toBe(true);
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.some((e) => e.dataset.playbackOpen === 'true'),
          ),
      )
      .toBe(true);
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    await page.request.delete(session.path, { headers: session.headers });
  }
});

test('persistent control loss ends after the grace period and permits a new session', async ({
  page,
}) => {
  test.setTimeout(65000);
  const first = await startDiagnostic(page);
  await page.route('**' + first.path, (route) =>
    route.request().method() === 'GET' ? route.abort('internetdisconnected') : route.continue(),
  );
  try {
    await expect(page.getByRole('status', { name: 'Connection recovery' })).toContainText(
      'Playback is paused',
    );
    await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled({ timeout: 35000 });
    await expect(page.getByRole('alert')).toBeVisible();
    await page.unrouteAll({ behavior: 'wait' });
    const second = await startDiagnostic(page);
    expect(second.path).not.toBe(first.path);
    await page.request.delete(second.path, { headers: second.headers });
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    await page.request.delete(first.path, { headers: first.headers });
  }
});

test('expired control credentials do not receive a reconnect grace period', async ({ page }) => {
  const session = await startDiagnostic(page);
  try {
    await page.route('**' + session.path, (route) =>
      route.request().method() === 'GET'
        ? route.fulfill({
            status: 403,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'Session credentials expired.' }),
          })
        : route.continue(),
    );
    await expect(page.getByRole('button', { name: 'Start call' })).toBeEnabled({ timeout: 8000 });
    await expect(page.getByRole('alert')).toContainText('Session credentials expired.');
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    await page.request.delete(session.path, { headers: session.headers });
  }
});

test('unavailable event stream falls back to polling without replacing the call', async ({
  page,
}) => {
  await page.route('**/api/health', async (route) => {
    const response = await route.fetch(),
      body = await response.json();
    await route.fulfill({ response, json: { ...body, control_transport: 'sse' } });
  });
  await page.route('**/api/sessions/*/events', (route) => route.abort('connectionreset'));
  const session = await startDiagnostic(page);
  try {
    await expect
      .poll(
        async () =>
          (await (await page.request.get(session.path, { headers: session.headers })).json()).ended,
      )
      .toBe(false);
    expect(
      (
        await page.request.post(session.path + '/play', {
          headers: session.headers,
          data: { scenario_id: 'sync_marker' },
        })
      ).ok(),
    ).toBe(true);
    await expect
      .poll(() =>
        page
          .locator('audio')
          .evaluateAll((els: HTMLAudioElement[]) =>
            els.some((e) => e.dataset.playbackOpen === 'true'),
          ),
      )
      .toBe(true);
    await expect(page.getByRole('status', { name: 'Connection recovery' })).not.toBeVisible();
    await expect(page.getByRole('button', { name: 'End call' })).toBeVisible();
  } finally {
    await page.unrouteAll({ behavior: 'wait' });
    await page.request.delete(session.path, { headers: session.headers });
  }
});
