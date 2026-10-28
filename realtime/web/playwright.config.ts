import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  timeout: 45000,
  workers: 1,
  use: {
    baseURL: process.env.DEMO_TEST_URL || 'http://127.0.0.1:8080',
    browserName: 'chromium',
    channel: process.env.PLAYWRIGHT_CHANNEL || 'msedge',
    headless: true,
    viewport: { width: 1440, height: 1050 },
    launchOptions: {
      ignoreDefaultArgs: process.env.RUN_BROWSER_TIMING === '1' ? ['--mute-audio'] : [],
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        '--autoplay-policy=no-user-gesture-required',
      ],
    },
    screenshot: 'only-on-failure',
  },
});
