import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';

test('waiting survives refresh, asks before starting, and never uses media while queued', async ({
  page,
}) => {
  let ready = false;
  let joins = 0;
  let starts = 0;
  let leaves = 0;
  let ticket = '';
  let expired = false;
  await page.addInitScript(() => {
    (window as any).mediaRequests = 0;
    navigator.mediaDevices.getUserMedia = async () => {
      (window as any).mediaRequests++;
      throw new Error('Queue must not request media');
    };
  });
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const fixtures: Record<string, unknown> = {
      '/api/access': { required: false, authenticated: true, queue_required: true },
      '/api/health': {
        livekit: true,
        active_session: true,
        openai_configured: true,
        tts_provider: 'elevenlabs',
      },
      '/api/avatars': [
        { id: 'call01', name: 'Call portrait', image_url: '/test-portrait', image_available: true },
      ],
      '/api/models': [{ id: 'avatarforever', name: 'AvatarForever', available: true }],
      '/api/voices': [
        { id: 'stock', review: { description: 'Stock voice', presentation: 'neutral' } },
      ],
      '/api/scenarios': [],
    };
    if (path === '/api/queue') {
      const key = request.headers()['x-queue-key'];
      expect(key).toMatch(/^[a-f0-9]{64}$/);
      expect(request.url()).not.toContain(key);
      if (request.method() === 'POST') {
        joins++;
        ticket = key;
      } else expect(key).toBe(ticket);
      if (request.method() === 'DELETE') {
        leaves++;
        return route.fulfill({ json: { left: true } });
      }
      if (expired) return route.fulfill({ status: 410, json: { detail: 'Expired' } });
      return route.fulfill({
        json: {
          state: ready ? 'ready' : 'waiting',
          position: ready ? 1 : 2,
          waiting: 2,
          turn_expires_in: ready ? 30 : null,
          backend_ready: true,
        },
      });
    }
    if (path === '/api/sessions' && request.method() === 'POST') {
      starts++;
      expect(request.headers()['x-queue-key']).toBe(ticket);
      // No GPU or external service is invoked by this UI test.
      return route.fulfill({ status: 503, json: { detail: 'Test backend unavailable' } });
    }
    return route.fulfill({ json: fixtures[path] ?? {} });
  });
  await page.route('**/test-portrait', (route) =>
    route.fulfill({
      path: resolve('../demo-avatars/call01.png'),
    }),
  );
  await page.goto('/');
  await page.getByRole('button', { name: 'Join queue', exact: true }).click();
  await expect(page.getByText('You’re #2 in line.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Leave queue', exact: true })).toBeVisible();
  expect(starts).toBe(0);
  expect(await page.evaluate(() => (window as any).mediaRequests)).toBe(0);
  await page.reload();
  await expect(page.getByText('You’re #2 in line.')).toBeVisible();
  expect(joins).toBe(1);
  ready = true;
  await expect(page.getByRole('button', { name: 'Start your call', exact: true })).toBeEnabled({
    timeout: 8000,
  });
  expect(starts).toBe(0);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 960 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    const controls = (await page.getByRole('group', { name: 'Call controls' }).boundingBox())!;
    const stage = (await page.getByRole('region', { name: 'Video call' }).boundingBox())!;
    expect(controls.y).toBeGreaterThan(stage.y + stage.height);
  }
  await page.screenshot({ path: '../../.cache/queue-20261101/queue-mobile.png', fullPage: true });
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.screenshot({ path: '../../.cache/queue-20261101/queue-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Start your call', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Test backend unavailable');
  expect(starts).toBe(1);
  await expect(page.getByRole('button', { name: 'Join queue', exact: true })).toBeEnabled();
  expect(await page.evaluate(() => sessionStorage.getItem('chert-queue-ticket'))).toBeNull();
  ready = false;
  await page.getByRole('button', { name: 'Join queue', exact: true }).click();
  await expect(page.getByText('You’re #2 in line.')).toBeVisible();
  await page.getByRole('button', { name: 'Leave queue', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Join queue', exact: true })).toBeEnabled();
  expect(leaves).toBeGreaterThanOrEqual(1);
  await page.getByRole('button', { name: 'Join queue', exact: true }).click();
  await expect(page.getByText('You’re #2 in line.')).toBeVisible();
  expired = true;
  await expect(page.getByText('Your place expired. Join again when you’re ready.')).toBeVisible({
    timeout: 8000,
  });
  expect(starts).toBe(1);
  expect(await page.evaluate(() => (window as any).mediaRequests)).toBe(0);
  // A free slot starts on the initial click; only people who waited must confirm.
  expired = false;
  ready = true;
  await page.getByRole('button', { name: 'Join queue', exact: true }).click();
  await expect.poll(() => starts).toBe(2);
  await expect(page.getByRole('button', { name: 'Join queue', exact: true })).toBeEnabled();
});
