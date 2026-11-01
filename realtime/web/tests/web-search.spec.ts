import { test, expect } from '@playwright/test';

test('checking status clears on interruption; sources stay attached to the played answer', async ({
  page,
}) => {
  await page.route('**/api/sessions/*/transcripts', (route) =>
    route.fulfill({ json: { revision: 0, ended: false, utterances: [] } }),
  );
  await page.goto('/tests/fixtures/conversation.html');
  await page.evaluate(() => (window as any).fixture.session({ id: 'lookup', session_key: 'test' }));
  await page.waitForResponse('**/api/sessions/lookup/transcripts');
  await page.evaluate(() =>
    (window as any).fixture.caption({
      revision: 1,
      ended: false,
      lookup: { state: 'checking' },
      utterances: [],
    }),
  );
  await expect(page.getByText('Checking the web…')).toBeVisible();
  await page.evaluate(() =>
    (window as any).fixture.caption({
      revision: 2,
      ended: false,
      lookup: { state: 'idle' },
      utterances: [],
    }),
  );
  await expect(page.getByText('Checking the web…')).toHaveCount(0);
  await page.evaluate(() =>
    (window as any).fixture.caption({
      revision: 3,
      ended: false,
      lookup: { state: 'idle' },
      utterances: [
        {
          id: 'answer',
          speaker: 'assistant',
          text: 'Here is the current release.',
          state: 'complete',
          order: 2,
          sources: [
            { url: 'https://www.python.org/downloads/', title: 'Python releases' },
            { url: 'javascript:alert(1)', title: 'Bad link' },
          ],
        },
      ],
    }),
  );
  const history = page.getByRole('region', { name: 'Conversation history' });
  const link = history.getByRole('link', { name: '1. python.org' });
  await expect(link).toHaveAttribute('href', 'https://www.python.org/downloads/');
  await expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  await expect(history.getByRole('link')).toHaveCount(1);
  await expect(page.locator('.live-captions').getByRole('link')).toHaveCount(1);
  await page.screenshot({
    path: '../../.cache/web-search-20261101/sources-desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await expect(link).toBeVisible();
  await page.getByRole('button', { name: 'End call' }).click();
  await expect(link).toBeVisible();
});
