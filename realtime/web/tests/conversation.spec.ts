import { test, expect } from '@playwright/test';

const rows = [
  { id: 'u1', speaker: 'user', text: 'What am I holding?', state: 'complete' },
  {
    id: 'a1',
    speaker: 'assistant',
    text: 'That looks like a tomato. Are you making something for lunch?',
    state: 'complete',
  },
  { id: 'u2', speaker: 'user', text: 'Yeah, I’m thinking of making pasta.', state: 'complete' },
  {
    id: 'a2',
    speaker: 'assistant',
    text: 'Nice. A simple tomato sauce would work well.',
    state: 'interrupted',
  },
];

test('room captions update before the HTTP poll and cannot be overwritten by older snapshots', async ({
  page,
}) => {
  await page.route('**/api/sessions/*/transcripts', (route) =>
    route.fulfill({
      json: {
        revision: 0,
        ended: false,
        utterances: [],
      },
    }),
  );
  await page.goto('/tests/fixtures/conversation.html');
  await page.evaluate(() =>
    (window as any).fixture.session({ id: 'push', session_key: 'fixture-key' }),
  );
  await page.waitForResponse('**/api/sessions/push/transcripts');
  const packet = {
    revision: 4,
    ended: false,
    utterances: [
      { id: 'user:one', speaker: 'user', text: 'Could you tell me', state: 'partial', order: 1 },
    ],
  };
  await page.evaluate((value) => (window as any).fixture.caption(value, 'stranger'), packet);
  await expect(page.locator('.speech-bubble')).toHaveCount(0);
  await page.evaluate((value) => (window as any).fixture.caption(value), packet);
  await expect(
    page
      .getByRole('region', { name: 'Conversation history' })
      .getByText('Could you tell me', { exact: true }),
  ).toBeVisible({ timeout: 400 });
  packet.revision++;
  packet.utterances[0].text = 'Could you tell me about tomatoes?';
  packet.utterances[0].state = 'complete';
  await page.evaluate((value) => (window as any).fixture.caption(value), packet);
  await expect(page.locator('.speech-bubble')).toHaveCount(1);
  await expect(
    page
      .getByRole('region', { name: 'Conversation history' })
      .getByText('Could you tell me about tomatoes?', { exact: true }),
  ).toBeVisible({
    timeout: 400,
  });
  await page.waitForResponse('**/api/sessions/push/transcripts');
  await expect(page.locator('.speech-bubble')).toHaveCount(1);
});

test('call controls sit below the stage and history opens separately', async ({ page }) => {
  const fixtures: Record<string, unknown> = {
    '/api/access': { required: false, authenticated: true },
    '/api/health': {
      livekit: false,
      openai_configured: true,
      active_session: false,
      camera_vision: true,
      tts_provider: 'elevenlabs',
      tts_configured: false,
    },
    '/api/avatars': [
      {
        id: 'ref01',
        name: 'Character 01',
        image_url: '',
        image_available: false,
        casting_status: 'not_cast',
      },
    ],
    '/api/models': [
      { id: 'mock', name: 'Preview', available: true, description: 'Transport fixture' },
    ],
    '/api/scenarios': [],
    '/api/voices': [],
  };
  await page.route('**/api/**', (route) =>
    route.fulfill({ json: fixtures[new URL(route.request().url()).pathname] ?? {} }),
  );
  await page.goto('/');
  const pair = (await page.getByRole('region', { name: 'Video call' }).boundingBox())!;
  const controls = page.getByRole('group', { name: 'Call controls' });
  expect((await controls.boundingBox())!.y).toBeGreaterThanOrEqual(pair.y + pair.height);
  await controls.getByRole('button', { name: 'Conversation history' }).click();
  await expect(page.getByRole('region', { name: 'Conversation history' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(controls.getByRole('button', { name: 'Conversation history' })).toBeFocused();
  await expect(page.locator('.video-pair button')).toHaveCount(1);
  await expect(page.getByRole('button', { name: 'Make your camera large' })).toBeVisible();
  await expect(controls.getByRole('button', { name: 'Settings', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByLabel('Conversation', { exact: true }).selectOption('voice_diagnostic');
  await expect(
    page.getByText('Add the ElevenLabs key and voice ID on the server to enable speech.'),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Match voice', exact: true })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await expect(controls.getByRole('button', { name: 'Settings', exact: true })).toBeFocused();
});

test('one shared panel below both tiles, mobile layout and respectful scroll', async ({ page }) => {
  await page.goto('/tests/fixtures/conversation.html');
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  const panel = page.getByRole('region', { name: 'Conversation history', exact: true });
  const tiles = page.getByRole('region', { name: 'Video call' });
  await expect(panel).toBeVisible();
  expect((await panel.boundingBox())!.y).toBeGreaterThan(
    (await tiles.boundingBox())!.y + (await tiles.boundingBox())!.height,
  );
  const user = await page.locator('.user .speech-bubble').first().boundingBox();
  const assistant = await page.locator('.assistant .speech-bubble').first().boundingBox();
  expect(user!.x).toBeLessThan(assistant!.x);
  await expect(panel.getByText('Interrupted', { exact: true })).toBeVisible();
  const controls = page.getByRole('group', { name: 'Call controls' });
  const panelBox = (await panel.boundingBox())!;
  expect((await controls.boundingBox())!.y).toBeGreaterThanOrEqual(panelBox.y + panelBox.height);
  await page.screenshot({ path: '../../runs/realtime/conversation/desktop.png', fullPage: true });
  const many = Array.from({ length: 40 }, (_, i) => ({ ...rows[i % 4], id: String(i) }));
  await page.evaluate((value) => (window as any).fixture.rows(value), many);
  await page.getByRole('log', { name: 'Call transcript' }).evaluate((element) => {
    element.scrollTop = 0;
    element.dispatchEvent(new Event('scroll'));
  });
  await page.evaluate(
    (value) => (window as any).fixture.rows(value),
    [...many, { ...rows[0], id: 'new' }],
  );
  await expect(page.getByRole('button', { name: 'Latest messages' })).toBeVisible();
  expect(
    await page
      .getByRole('log', { name: 'Call transcript' })
      .evaluate((element) => element.scrollTop),
  ).toBe(0);
  await page.getByRole('button', { name: 'Latest messages' }).click();
  await expect(page.getByRole('button', { name: 'Latest messages' })).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate((value) => (window as any).fixture.rows(value), rows);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  const mobilePanel = (await panel.boundingBox())!;
  expect((await controls.boundingBox())!.y).toBeGreaterThanOrEqual(
    mobilePanel.y + mobilePanel.height,
  );
  await page.screenshot({ path: '../../runs/realtime/conversation/mobile.png', fullPage: true });
});

test('caption snapshots reconcile, survive hangup and reset on next call', async ({ page }) => {
  let revision = 1;
  let content = [rows[0]];
  await page.route('**/api/sessions/*/transcripts', async (route) => {
    expect(route.request().headers()['x-session-key']).toBe('fixture-key');
    await route.fulfill({ json: { revision, ended: false, utterances: content } });
  });
  await page.goto('/tests/fixtures/conversation.html');
  await page.evaluate(() =>
    (window as any).fixture.session({ id: 'one', session_key: 'fixture-key' }),
  );
  await expect(page.locator('.speech-bubble')).toHaveCount(1);
  content = rows;
  revision++;
  await expect(page.locator('.speech-bubble')).toHaveCount(4);
  await page.getByRole('button', { name: 'End call' }).click();
  await expect(page.locator('.speech-bubble')).toHaveCount(4);
  content = [];
  revision = 0;
  await page.evaluate(() =>
    (window as any).fixture.session({ id: 'two', session_key: 'fixture-key' }),
  );
  await expect(page.getByText('Your conversation will appear here.')).toBeVisible();
});

test('camera preview is private until call sharing and releases published clone', async ({
  page,
}) => {
  await page.goto('/tests/fixtures/conversation.html');
  await page.getByRole('button', { name: 'Toggle camera' }).click();
  await expect(page.locator('video')).toHaveClass(/visible/);
  expect(await page.evaluate(() => (window as any).published)).toBeUndefined();
  await page.evaluate(() => (window as any).fixture.share(true));
  await expect(page.getByText('Shared with AI', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'End call' }).click();
  await expect(page.getByText('Shared with AI', { exact: true })).toHaveCount(0);
  await expect
    .poll(() => page.evaluate(() => (window as any).published.mediaStreamTrack.readyState))
    .toBe('ended');
});
