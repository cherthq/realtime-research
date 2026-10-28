import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import nodePath from 'node:path';

test('browser flash/tone sync and interruption through actual WebRTC playback', async ({
  page,
}) => {
  test.skip(
    process.env.RUN_BROWSER_TIMING !== '1',
    'Opt-in audio-graph instrumentation; not a physical-speaker measurement',
  );
  await page.addInitScript(() => {
    const original = AudioContext.prototype.createAnalyser;
    const originalGain = AudioContext.prototype.createGain;
    const originalPeer = window.RTCPeerConnection;
    (window as any).outputAnalysers = [];
    (window as any).outputGains = [];
    (window as any).timingPeers = [];
    window.RTCPeerConnection = class extends originalPeer {
      constructor(config?: RTCConfiguration) {
        super(config);
        (window as any).timingPeers.push(this);
      }
    };
    AudioContext.prototype.createGain = function () {
      const gain = originalGain.call(this);
      (window as any).outputGains.push(gain);
      return gain;
    };
    AudioContext.prototype.createAnalyser = function () {
      const analyser = original.call(this);
      (window as any).outputAnalysers.push({ context: this, analyser });
      return analyser;
    };
  });
  await page.goto('/');
  if (process.env.DEMO_TEST_ACCESS_CODE) {
    await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE);
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
  }
  // This isolates transport timing even when the deployment defaults to a real
  // model and paid live voice. A flash/beep fixture is never model lip-sync.
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByLabel('Model', { exact: true }).selectOption('mock');
  await page.getByLabel('Conversation', { exact: true }).selectOption('diagnostic');
  await page.getByRole('button', { name: 'Close settings' }).click();
  const created = page.waitForResponse(
    (r) => r.url().endsWith('/api/sessions') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Start call' }).click();
  const session = await (await created).json();
  const path = `/api/sessions/${session.id}`,
    headers = { 'X-Session-Key': session.session_key };
  try {
    await expect(page.locator('audio')).toHaveCount(1);
    await expect
      .poll(
        () =>
          page
            .getByLabel('Generated avatar', { exact: true })
            .evaluate((v: HTMLVideoElement) => v.currentTime),
        { timeout: 15000 },
      )
      .toBeGreaterThan(2);
    const environment = await page.evaluate(async () => {
      const video = document.querySelector('[aria-label="Generated avatar"]') as HTMLVideoElement;
      // Observe the real post-gate output graph, without replacing its source or
      // routing. HTML audio elements are deliberately silent under webAudioMix.
      const nodes = (window as any).outputAnalysers;
      if (nodes.length !== 1) throw new Error('Expected exactly one remote output analyser');
      const { context, analyser } = nodes[0] as { context: AudioContext; analyser: AnalyserNode };
      if (context.state !== 'running') throw new Error('Output context is not running');
      const samples = new Float32Array(analyser.fftSize);
      const canvas = document.createElement('canvas');
      canvas.width = 13;
      canvas.height = 13;
      const paint = canvas.getContext('2d', { willReadFrequently: true })!;
      const state = {
        audio: [] as number[],
        rawAudio: [] as number[],
        video: [] as number[],
        loud: false,
        flash: false,
        rms: 0,
        peak: 0,
        active: true,
        quietAt: 0,
        quietFrames: 0,
        reset() {
          this.audio = [];
          this.rawAudio = [];
          this.video = [];
        },
        close() {
          this.active = false;
        },
      };
      (window as any).transportTiming = state;
      function readAudio() {
        if (!state.active) return;
        analyser.getFloatTimeDomainData(samples);
        state.rms = Math.sqrt(
          samples.reduce((sum, value) => sum + value * value, 0) / samples.length,
        );
        state.peak = Math.max(state.peak, state.rms);
        const loud = state.rms > 0.025;
        if (loud && !state.loud) {
          const at = performance.now();
          state.rawAudio.push(at);
          // Group fragments of one 100ms marker. Preserve every raw edge so a
          // network underrun is reported, not miscounted as another test tone.
          if (!state.audio.length || at - state.audio[state.audio.length - 1] > 200)
            state.audio.push(at);
        }
        if (!loud && state.loud) state.quietAt = performance.now();
        state.quietFrames = loud ? 0 : state.quietFrames + 1;
        state.loud = loud;
        requestAnimationFrame(readAudio);
      }
      function readVideo(_now: number, metadata: VideoFrameCallbackMetadata) {
        if (!state.active) return;
        paint.drawImage(video, 605, 19, 13, 13, 0, 0, 13, 13);
        const pixel = paint.getImageData(6, 6, 1, 1).data;
        const flash = pixel[0] > 120 && pixel[1] > 150 && pixel[2] < 180;
        if (flash && !state.flash) state.video.push(metadata.expectedDisplayTime);
        state.flash = flash;
        video.requestVideoFrameCallback(readVideo);
      }
      requestAnimationFrame(readAudio);
      video.requestVideoFrameCallback(readVideo);
      return {
        sample_rate: context.sampleRate,
        analyser_window_samples: analyser.fftSize,
        context_base_latency_seconds: context.baseLatency,
        context_output_latency_seconds: context.outputLatency,
        browser: navigator.userAgent,
      };
    });
    const trials = [];
    for (let iteration = 0; iteration < 5; iteration++) {
      await page.evaluate(() => (window as any).transportTiming.reset());
      expect(
        (
          await page.request.post(path + '/play', { headers, data: { scenario_id: 'sync_marker' } })
        ).ok(),
      ).toBe(true);
      await expect
        .poll(
          () =>
            page.evaluate(() => {
              const t = (window as any).transportTiming;
              return [t.audio.length, t.video.length];
            }),
          { timeout: 7000 },
        )
        .toEqual([2, 2]);
      await expect
        .poll(() =>
          page.evaluate(() => {
            const t = (window as any).transportTiming;
            return !t.loud && !t.flash && t.quietFrames >= 5;
          }),
        )
        .toBe(true);
      const trial = await page.evaluate(() => {
        const t = (window as any).transportTiming;
        return {
          audio_edges_ms: t.audio,
          raw_audio_edges_ms: t.rawAudio,
          video_edges_ms: t.video,
          offsets_ms: t.audio.map((at: number, i: number) => at - t.video[i]),
        };
      });
      trials.push(trial);
    }
    const offsets = trials
      .flatMap((t) => t.offsets_ms as number[])
      .map(Math.abs)
      .sort((a, b) => a - b);
    const interruptions = [];
    for (const trigger of ['local', 'server'] as const) {
      expect(
        (
          await page.request.post(path + '/play', {
            headers,
            data: { scenario_id: 'interrupt_marker' },
          })
        ).ok(),
      ).toBe(true);
      await expect.poll(() => page.evaluate(() => (window as any).transportTiming.loud)).toBe(true);
      const markedAt = await page.evaluate(
        ({ trigger, path, headers }) => {
          const at = performance.now();
          if (trigger === 'local')
            (document.querySelector('[aria-label="Interrupt"]') as HTMLButtonElement).click();
          else void fetch(path + '/interrupt', { method: 'POST', headers });
          return at;
        },
        { trigger, path, headers },
      );
      await expect
        .poll(
          () =>
            page.evaluate(() => {
              const t = (window as any).transportTiming;
              return !t.loud && t.quietFrames >= 5;
            }),
          { timeout: 1500 },
        )
        .toBe(true);
      const quietAt = await page.evaluate(() => (window as any).transportTiming.quietAt as number);
      const stopMs = quietAt - markedAt;
      interruptions.push({ trigger, audio_graph_stop_ms: stopMs });
    }
    const report = {
      scope:
        'Browser audio-graph vs video-compositor timing; not physical output or model lip-sync',
      fixture: 'sync_marker: frame-aligned tones at 0.52s and 2.0s; flash uses 40ms frame RMS',
      environment,
      trials,
      interruptions,
      measurement_note:
        'RAF sampling adds up to one foreground display interval; analyser window is 128 samples. Device latency is reported, not measured acoustically.',
      marker_count: offsets.length,
      p95_absolute_offset_ms: offsets[Math.ceil(offsets.length * 0.95) - 1],
    };
    const directory = nodePath.resolve(
      process.env.TIMING_OUTPUT || '../../runs/realtime/browser-tests',
    );
    await mkdir(directory, { recursive: true });
    await writeFile(
      nodePath.join(directory, 'transport-timing.json'),
      JSON.stringify(report, null, 2),
    );
    for (const result of interruptions) {
      expect(result.audio_graph_stop_ms).toBeGreaterThanOrEqual(0);
      expect(result.audio_graph_stop_ms).toBeLessThan(300);
    }
    expect(report.marker_count).toBe(10);
    expect(report.p95_absolute_offset_ms).toBeLessThanOrEqual(80);
  } catch (error) {
    const diagnostic = await page.evaluate(async () => {
      const stats: unknown[] = [];
      for (const pc of (window as any).timingPeers as RTCPeerConnection[]) {
        (await pc.getStats()).forEach((value) => {
          if (value.type === 'inbound-rtp') stats.push(value);
        });
      }
      return {
        timing: (window as any).transportTiming,
        contexts: (window as any).outputAnalysers.map(({ context }: any) => ({
          state: context.state,
          time: context.currentTime,
        })),
        gains: (window as any).outputGains.map((gain: GainNode) => gain.gain.value),
        elements: Array.from(document.querySelectorAll('audio')).map((el) => ({
          paused: el.paused,
          gate: el.dataset.playbackOpen,
        })),
        stats,
      };
    });
    await test.info().attach('timing-diagnostic.json', {
      body: JSON.stringify(diagnostic, null, 2),
      contentType: 'application/json',
    });
    await writeFile(
      test.info().outputPath('timing-diagnostic.json'),
      JSON.stringify(diagnostic, null, 2),
    );
    throw error;
  } finally {
    await page.evaluate(() => (window as any).transportTiming?.close());
    if (await page.getByRole('button', { name: 'End call' }).isVisible())
      await page.getByRole('button', { name: 'End call' }).click();
    await page.request.delete(path, { headers });
  }
});
