import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';

test.beforeEach(async ({ page }) => {
  const fixtures: Record<string, unknown> = {
    '/api/access': { required: false, authenticated: true },
    '/api/health': {
      livekit: true,
      openai_configured: true,
      camera_vision: true,
      tts_provider: 'elevenlabs',
      tts_configured: true,
    },
    '/api/avatars': [1, 2, 3].map((n) => ({
      id: `ref0${n}`,
      name: `Character 0${n}`,
      image_url: `/test-portrait/${n}`,
      image_available: true,
      casting_status: 'not_cast',
    })),
    '/api/models': [
      {
        id: 'avatarforever',
        name: 'AvatarForever',
        available: true,
        description: 'Streaming avatar',
      },
    ],
    '/api/voices': [],
    '/api/scenarios': [],
  };
  await page.route('**/api/**', (route) =>
    route.fulfill({ json: fixtures[new URL(route.request().url()).pathname] ?? {} }),
  );
  await page.route('**/test-portrait/*', (route) =>
    route.fulfill({
      path: resolve(
        `../../data/eval/facetime-zero-shot-v1/images/ref0${route.request().url().split('/').at(-1)}.png`,
      ),
    }),
  );
});

test('avatar library selects its saved voice, previews overrides, and stays usable on mobile', async ({
  page,
}) => {
  const previews: string[] = [];
  await page.route('**/api/avatars', (route) =>
    route.fulfill({
      json: Array.from({ length: 13 }, (_, n) => ({
        id: `character${n}`,
        name: `Call portrait ${n + 1}`,
        image_url: `/test-portrait/${(n % 3) + 1}`,
        image_available: true,
        voice_profile_id: n % 2 ? 'brightVoice' : 'warmVoice',
        casting_status: 'matched',
      })),
    }),
  );
  await page.route('**/api/voices', (route) =>
    route.fulfill({
      json: [
        {
          id: 'warmVoice',
          name: 'Warm voice',
          preview_available: true,
          review: { presentation: 'neutral' },
        },
        {
          id: 'brightVoice',
          name: 'Bright voice',
          preview_available: true,
          review: { presentation: 'neutral' },
        },
      ],
    }),
  );
  await page.route('**/api/voices/*/preview', (route) => {
    previews.push(new URL(route.request().url()).pathname);
    return route.fulfill({ status: 204 });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  const picker = page.locator('.avatar-choices');
  await expect(picker.getByRole('button')).toHaveCount(13);
  await page.getByRole('button', { name: 'Select Call portrait 12', exact: true }).click();
  await expect(page.getByLabel('Voice', { exact: true })).toHaveValue('');
  await expect(page.getByLabel('Voice', { exact: true }).locator('option:checked')).toHaveText(
    'Matched voice · Bright voice',
  );
  await page.getByRole('button', { name: 'Preview voice', exact: true }).click();
  await expect.poll(() => previews.at(-1)).toBe('/api/voices/brightVoice/preview');
  await page.getByLabel('Voice', { exact: true }).selectOption('warmVoice');
  await page.getByRole('button', { name: 'Preview voice', exact: true }).click();
  await expect.poll(() => previews.at(-1)).toBe('/api/voices/warmVoice/preview');
  await page.getByRole('button', { name: 'Select Call portrait 2', exact: true }).click();
  await expect(page.getByLabel('Voice', { exact: true })).toHaveValue('');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 960 });
    await page.getByRole('button', { name: 'Select Call portrait 13', exact: true }).click();
    const box = (await picker.boundingBox())!;
    expect(box.height).toBeLessThanOrEqual(268);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    await expect(
      page.getByRole('button', { name: 'Select Call portrait 13', exact: true }),
    ).toHaveAttribute('aria-pressed', 'true');
  }
});

test('desktop stage, real camera preview, menus and phone layouts preserve media elements', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Start call', exact: true })).toBeEnabled();
  await expect(page.getByRole('img', { name: 'Avatar reference' })).toBeVisible();
  const avatar = page.getByLabel('Generated avatar', { exact: true });
  const original = await avatar.elementHandle();
  const originalHold = await page.locator('.avatar-hold').elementHandle();
  const main = (await page.locator('.avatar-tile').boundingBox())!;
  const local = (await page.locator('.local-tile').boundingBox())!;
  expect(main.width / main.height).toBeCloseTo(4 / 3);
  expect(local.width).toBeLessThan(main.width / 3);
  expect(local.x).toBeGreaterThan(main.x + main.width / 2);
  expect(local.x + local.width).toBeLessThan(main.x + main.width);
  expect(local.y).toBeGreaterThan(main.y);
  expect(local.y + local.height).toBeLessThan(main.y + main.height);
  await page.getByRole('button', { name: 'Turn camera on', exact: true }).click();
  await expect(page.getByLabel('Your camera', { exact: true })).toHaveClass(/visible/);
  const cameraNode = await page.getByLabel('Your camera', { exact: true }).elementHandle();
  const cameraTrack = await cameraNode!.evaluateHandle(
    (el) => ((el as HTMLVideoElement).srcObject as MediaStream).getVideoTracks()[0],
  );
  await page.getByRole('button', { name: 'Make your camera large' }).click();
  expect((await page.locator('.local-tile').boundingBox())!.width).toBeGreaterThan(
    (await page.locator('.avatar-tile').boundingBox())!.width,
  );
  expect(
    await original!.evaluate(
      (el) => el === document.querySelector('[aria-label="Generated avatar"]'),
    ),
  ).toBe(true);
  expect(await originalHold!.evaluate((el) => el === document.querySelector('.avatar-hold'))).toBe(
    true,
  );
  expect(
    await cameraNode!.evaluate((el) => el === document.querySelector('[aria-label="Your camera"]')),
  ).toBe(true);
  expect(await cameraTrack.evaluate((track) => track.readyState)).toBe('live');
  await page.screenshot({
    path: '../../runs/realtime/call-ui/swapped-desktop.png',
    fullPage: true,
  });
  await page.getByRole('button', { name: 'Make the AI video large' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Make your camera large' })).toBeVisible();
  await expect(page.getByText('Avatar', { exact: true })).not.toBeVisible();
  await page.screenshot({ path: '../../runs/realtime/call-ui/desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Captions', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Captions', exact: true })).toHaveAttribute(
    'aria-pressed',
    'false',
  );
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('checkbox', { name: 'Live captions', exact: true }).check();
  await page.getByLabel('Call timer', { exact: true }).uncheck();
  await expect(page.getByLabel('Camera', { exact: true })).toBeEnabled();
  const cameraId = await page
    .getByLabel('Camera', { exact: true })
    .locator('option')
    .nth(1)
    .getAttribute('value');
  const previousTrack = await page
    .getByLabel('Your camera', { exact: true })
    .evaluateHandle(
      (el) =>
        (el as HTMLVideoElement).srcObject &&
        ((el as HTMLVideoElement).srcObject as MediaStream).getVideoTracks()[0],
    );
  await page.getByLabel('Camera', { exact: true }).selectOption(cameraId!);
  await expect.poll(() => previousTrack.evaluate((track) => track?.readyState)).toBe('ended');
  await expect
    .poll(() =>
      page
        .getByLabel('Your camera', { exact: true })
        .evaluate(
          (el) =>
            ((el as HTMLVideoElement).srcObject as MediaStream).getVideoTracks()[0].readyState,
        ),
    )
    .toBe('live');
  // Compact menu defaults to devices; advanced configuration is still available.
  await page.getByText('Avatar & voice', { exact: true }).click();
  await page.screenshot({ path: '../../runs/realtime/call-ui/settings.png', fullPage: true });
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Settings', exact: true })).toBeFocused();
  await page.getByRole('button', { name: 'Conversation history', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Your conversation' })).toBeVisible();
  await page.keyboard.press('Escape');
  expect(
    await original!.evaluate(
      (el) => el === document.querySelector('[aria-label="Generated avatar"]'),
    ),
  ).toBe(true);
  expect(await originalHold!.evaluate((el) => el === document.querySelector('.avatar-hold'))).toBe(
    true,
  );
  await expect(page.locator('.avatar-hold')).toBeHidden();

  const selectedTrack = await cameraNode!.evaluateHandle(
    (el) => ((el as HTMLVideoElement).srcObject as MediaStream).getVideoTracks()[0],
  );
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    const frame = (await page.locator('.avatar-tile').boundingBox())!;
    const preview = (await page.locator('.local-tile').boundingBox())!;
    expect(preview.x + preview.width).toBeLessThan(frame.x + frame.width);
    expect(preview.y).toBeGreaterThan(frame.y);
    const controls = page.getByRole('group', { name: 'Call controls' });
    expect((await controls.boundingBox())!.y).toBeGreaterThanOrEqual(frame.y + frame.height);
    for (const button of await controls.getByRole('button').all()) {
      const box = (await button.boundingBox())!;
      expect(box.width).toBeGreaterThanOrEqual(44);
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(width);
    }
    expect(
      await page.locator('.avatar-hold').evaluate((el) => getComputedStyle(el).objectFit),
    ).toBe(await avatar.evaluate((el) => getComputedStyle(el).objectFit));
    await page.screenshot({
      path: `../../runs/realtime/call-ui/mobile-${width}.png`,
      fullPage: true,
    });
    await page.getByRole('button', { name: 'Make your camera large' }).click();
    const bigCamera = (await page.locator('.local-tile').boundingBox())!;
    const smallAI = (await page.locator('.avatar-tile').boundingBox())!;
    expect(bigCamera.width).toBeGreaterThan(smallAI.width * 2);
    expect(smallAI.x + smallAI.width).toBeLessThan(bigCamera.x + bigCamera.width);
    expect(await selectedTrack.evaluate((track) => track.readyState)).toBe('live');
    await page.screenshot({
      path: `../../runs/realtime/call-ui/swapped-mobile-${width}.png`,
      fullPage: true,
    });
    await page.getByRole('button', { name: 'Make the AI video large' }).click();
  }
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  expect(
    await page
      .getByRole('dialog', { name: 'Call settings' })
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe('none');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Turn camera off', exact: true }).click();
  await expect(page.getByLabel('Your camera', { exact: true })).not.toHaveClass(/visible/);
});

test('live caption deltas reuse the same bubble and retain only two latest turns', async ({
  page,
}) => {
  await page.goto('/tests/fixtures/conversation.html');
  const rows = [
    { id: 'u1', speaker: 'user', text: 'Can you see', state: 'partial' },
    { id: 'a1', speaker: 'assistant', text: 'Yes, I can see your camera.', state: 'complete' },
  ];
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  const overlay = page.getByRole('log', { name: 'Live captions' });
  const first = await overlay.locator('.live-caption.user').elementHandle();
  rows[0].text = 'Can you see what I am holding?';
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  expect(await first!.evaluate((el) => el.isConnected)).toBe(true);
  await expect(overlay.getByText(rows[0].text)).toBeVisible();
  await page.evaluate(
    (value) => (window as any).fixture.rows(value),
    [...rows, { id: 'u2', speaker: 'user', text: 'Tell me more.', state: 'complete' }],
  );
  await expect(overlay.locator('.live-caption')).toHaveCount(2);
  await expect(overlay.getByText(rows[0].text)).toHaveCount(0);
  // Full history remains available, independent of the compact overlay.
  await expect(
    page.getByRole('region', { name: 'Conversation history' }).locator('.speech-bubble'),
  ).toHaveCount(3);
});

test('switching camera republishes its clone; unavailable devices preserve the working preview', async ({
  page,
}) => {
  await page.goto('/tests/fixtures/conversation.html');
  await page.getByRole('button', { name: 'Toggle camera' }).click();
  await page.evaluate(() => (window as any).fixture.share(true));
  await expect.poll(() => page.evaluate(() => !!(window as any).published)).toBe(true);
  const published = await page.evaluateHandle(() => (window as any).published);
  await page.evaluate(async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    await (window as any).fixture.cameraDevice(
      devices.find((item) => item.kind === 'videoinput')!.deviceId,
    );
  });
  await expect
    .poll(() => published.evaluate((track) => track.mediaStreamTrack.readyState))
    .toBe('ended');
  await expect
    .poll(() => page.evaluate(() => (window as any).published.mediaStreamTrack.readyState))
    .toBe('live');
  await page.evaluate(() => (window as any).fixture.cameraDevice('missing-camera'));
  await expect(page.locator('video')).toHaveClass(/visible/);
  expect(await page.evaluate(() => (window as any).published.mediaStreamTrack.readyState)).toBe(
    'live',
  );
});

test('captions fade after five seconds, snapshot refresh does not revive them, history survives', async ({
  page,
}) => {
  await page.clock.install();
  await page.goto('/tests/fixtures/conversation.html');
  const rows = [
    { id: 'a1', speaker: 'assistant', text: 'A short spoken response.', state: 'complete' },
  ];
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  const overlay = page.getByRole('log', { name: 'Live captions' });
  await expect(overlay.locator('.caption-text')).toHaveText(rows[0].text);
  await expect(overlay.locator('.caption-speaker')).toHaveCount(0);
  await page.clock.runFor(3000);
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  await page.clock.runFor(2020);
  await expect(overlay.locator('.live-caption')).toHaveClass(/leaving/);
  await page.clock.runFor(300);
  await expect(overlay.locator('.live-caption')).toHaveCount(0);
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  await expect(overlay.locator('.live-caption')).toHaveCount(0);
  await expect(
    page.getByRole('region', { name: 'Conversation history' }).getByText(rows[0].text),
  ).toBeVisible();
  rows[0].text += ' More words arrive.';
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  await expect(overlay.locator('.caption-text')).toHaveText(rows[0].text);
  await page.evaluate(() => (window as any).fixture.overlayActive(false));
  await expect(overlay.locator('.live-caption')).toHaveClass(/leaving/);
  await page.clock.runFor(300);
  await expect(overlay.locator('.live-caption')).toHaveCount(0);
  await expect(
    page.getByRole('region', { name: 'Conversation history' }).getByText(rows[0].text),
  ).toBeVisible();
});
