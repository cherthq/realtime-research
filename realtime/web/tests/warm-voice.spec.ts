import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/** Explicit paid benchmark. Synthetic input only; never captures a person's mic. */
test('warm spoken turns through voice, GPU and browser playback', async ({ page }) => {
  test.skip(process.env.RUN_WARM_VOICE_BASELINE !== '1', 'Requires explicit GPU/provider opt-in');
  test.setTimeout(900000);
  if (!process.env.WARM_VOICE_INPUT_WAV)
    throw new Error('Supply one frozen synthetic spoken prompt');
  const turns = Number(process.env.WARM_VOICE_TURNS || 31);
  const sessionSeconds = Number(process.env.WARM_VOICE_SESSION_SECONDS || 0);
  if (![0, 600].includes(sessionSeconds))
    throw new Error('Session duration must be omitted or 600 seconds');
  if (sessionSeconds && turns !== 31)
    throw new Error('The ten-minute protocol uses 31 spaced turns');
  const model = process.env.WARM_VOICE_MODEL || 'avatarforever';
  if (!['avatarforever', 'mock'].includes(model)) throw new Error('Unsupported benchmark model');
  if (!Number.isInteger(turns) || turns < 2 || turns > 31)
    throw new Error('Use 2–31 total turns; first is excluded');
  const input = await readFile(process.env.WARM_VOICE_INPUT_WAV);
  const output = path.resolve(process.env.WARM_VOICE_OUTPUT || '../../runs/realtime/warm-voice');
  await mkdir(path.dirname(output), { recursive: true });
  await mkdir(output); // Preserve earlier successful and failed measurements.
  const report: any = {
    status: 'running',
    model,
    qualifies_avatar_generation: model === 'avatarforever',
    input_sha256: createHash('sha256').update(input).digest('hex'),
    turns: [],
    session_seconds: sessionSeconds,
    continuity_samples: [],
    scope:
      'Synthetic WebAudio input completion to output audio-graph onset and video-compositor presentation timestamp with its playback gate open. One browser clock; no physical speaker or model lip-sync measurement.',
    uncertainty:
      'Input completion callback and audio RAF sampling add scheduling uncertainty. Speech boundaries use 20ms RMS windows; video timing is the compositor estimate, not an acoustic measurement.',
  };
  await page.addInitScript(() => {
    const state: any = {
      input: null,
      destination: null,
      buffer: null,
      analysers: [],
      peers: [],
      epoch: 0,
      trial: null,
      active: true,
    };
    (window as any).warmVoice = state;
    const originalPeer = window.RTCPeerConnection;
    window.RTCPeerConnection = class extends originalPeer {
      constructor(config?: RTCConfiguration) {
        super(config);
        state.peers.push(this);
      }
    };
    const analyser = AudioContext.prototype.createAnalyser;
    AudioContext.prototype.createAnalyser = function () {
      const node = analyser.call(this);
      state.analysers.push(node);
      return node;
    };
    const getMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (constraints?.audio && !constraints.video) {
        state.input ||= new AudioContext({ sampleRate: 48000 });
        state.destination ||= state.input.createMediaStreamDestination();
        await state.input.resume();
        return state.destination.stream;
      }
      return getMedia(constraints);
    };
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name !== 'chert-playback-start') continue;
        const detail = (entry as PerformanceMark).detail,
          epoch = Math.floor(detail.frameId / 65536);
        state.epoch = Math.max(state.epoch, epoch);
        const trial = state.trial;
        if (trial?.endAt && epoch > trial.previousEpoch && !trial.videoAt)
          trial.videoAt = detail.expectedDisplayTime;
      }
    }).observe({ type: 'mark' });
    function sample() {
      if (!state.active) return;
      const outputs = state.analysers.filter(
        (node: AnalyserNode) => node.fftSize === 128 && node.context !== state.input,
      );
      if (outputs.length === 1) {
        const meter = outputs[0] as AnalyserNode,
          values = new Float32Array(meter.fftSize);
        meter.getFloatTimeDomainData(values);
        state.rms = Math.sqrt(
          values.reduce((sum, value) => sum + value * value, 0) / values.length,
        );
        if (state.trial && !state.trial.endAt && state.rms > 0.01)
          state.trial.prematureOutput = true;
        if (state.trial?.endAt && state.rms > 0.01 && !state.trial.audioAt)
          state.trial.audioAt = performance.now();
      }
      requestAnimationFrame(sample);
    }
    requestAnimationFrame(sample);
  });
  let session: any,
    monitoring = false,
    monitor: Promise<void> | undefined,
    monitorError: unknown;
  const save = () => writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  try {
    await page.goto('/');
    if (process.env.DEMO_TEST_ACCESS_CODE) {
      await page.getByLabel('Access code', { exact: true }).fill(process.env.DEMO_TEST_ACCESS_CODE);
      await page.getByRole('button', { name: 'Continue', exact: true }).click();
    }
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByLabel('Model', { exact: true }).selectOption(model);
    await page.getByLabel('Conversation', { exact: true }).selectOption('voice_diagnostic');
    await page.getByRole('button', { name: 'Close settings' }).click();
    const created = page.waitForResponse(
      (r) => r.url().endsWith('/api/sessions') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Start call' }).click();
    session = await (await created).json();
    if (!session.id) throw new Error('Voice session did not start');
    await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeVisible({
      timeout: 45000,
    });
    await expect
      .poll(() =>
        page.evaluate(() => {
          const h = (window as any).warmVoice;
          return h.analysers.filter(
            (node: AnalyserNode) => node.fftSize === 128 && node.context !== h.input,
          ).length;
        }),
      )
      .toBe(1);
    report.input = await page.evaluate(async (encoded) => {
      const h = (window as any).warmVoice;
      const decoded = await h.input.decodeAudioData(
        Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0)).buffer,
      );
      if (decoded.numberOfChannels !== 1) throw new Error('Use a mono prompt');
      const samples = decoded.getChannelData(0);
      const windowSamples = Math.round(decoded.sampleRate * 0.02),
        active: number[] = [];
      for (let i = 0; i < samples.length; i += windowSamples) {
        const segment = samples.subarray(i, i + windowSamples);
        if (
          Math.sqrt(segment.reduce((sum: number, v: number) => sum + v * v, 0) / segment.length) >
          0.001
        )
          active.push(i);
      }
      if (!active.length) throw new Error('No speech energy in supplied prompt');
      const start = active[0],
        end = Math.min(samples.length, active[active.length - 1] + windowSamples);
      h.buffer = h.input.createBuffer(1, end - start, decoded.sampleRate);
      h.buffer.copyToChannel(samples.slice(start, end), 0);
      return {
        sample_rate: decoded.sampleRate,
        source_samples: decoded.length,
        trim_start: start,
        trim_end: end,
        spoken_seconds: h.buffer.duration,
        trim_rms_threshold: 0.001,
        trim_window_ms: 20,
      };
    }, input.toString('base64'));
    const headers = { 'X-Session-Key': session.session_key };
    const read = async () =>
      (await page.request.get(`/api/sessions/${session.id}`, { headers })).json();
    report.voice = (await read()).metrics.effective_voice;
    await expect
      .poll(() =>
        page
          .getByLabel('Generated avatar', { exact: true })
          .evaluate((v: HTMLVideoElement) => [v.videoWidth, v.videoHeight]),
      )
      .toEqual([640, 480]);
    const turnScheduleStart = Date.now();
    if (sessionSeconds) {
      report.continuity_scope =
        'One ten-minute browser call with 31 spaced synthetic turns and idle media; separate from uninterrupted ten-minute model generation.';
      await page
        .getByLabel('Generated avatar', { exact: true })
        .evaluate((video: HTMLVideoElement) => {
          const h = (window as any).warmVoice;
          h.continuity = { frames: 0, lastAt: performance.now(), maxGapMs: 0, wrongGeometry: 0 };
          const next = (at: number) => {
            if (!h.active) return;
            const s = h.continuity;
            s.frames++;
            s.maxGapMs = Math.max(s.maxGapMs, at - s.lastAt);
            s.lastAt = at;
            if (video.videoWidth !== 640 || video.videoHeight !== 480) s.wrongGeometry++;
            video.requestVideoFrameCallback(next);
          };
          video.requestVideoFrameCallback(next);
        });
      monitoring = true;
      monitor = (async () => {
        while (monitoring) {
          const s = await read(),
            m = s.metrics;
          report.continuity_samples.push({
            elapsed_seconds: m.elapsed_seconds,
            ended: s.ended,
            end_reason: s.end_reason,
            generated_frames: m.generated_frames,
            distinct_generated_frames: m.distinct_generated_frames,
            pending_audio_seconds: m.pending_audio_seconds,
            decoded_batch_queue: m.decoded_batch_queue,
            render_stalls: m.render_stalls,
            gpu_allocated_bytes: m.gpu_memory?.gpu_allocated_bytes,
            host_peak_rss_bytes: m.gpu_memory?.host_peak_rss_bytes,
            error: m.error,
          });
          if (m.error) throw new Error(m.error);
          if (s.ended) {
            report.automatic_end = { reason: s.end_reason, elapsed_seconds: m.elapsed_seconds };
            if (s.end_reason !== 'session_limit')
              throw new Error(`Unexpected end: ${s.end_reason}`);
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      })().catch((error) => {
        monitorError = error;
      });
    }
    for (let index = 0; index < turns; index++) {
      if (sessionSeconds) {
        // Leave ninety seconds for startup, draining and observing the call cap.
        const due = turnScheduleStart + (index * (sessionSeconds - 90) * 1000) / (turns - 1);
        while (Date.now() < due) {
          if (monitorError) throw monitorError;
          if (report.automatic_end) throw new Error('Call ended before all scheduled turns');
          await new Promise((resolve) => setTimeout(resolve, Math.min(1000, due - Date.now())));
        }
      }
      await expect
        .poll(
          async () => {
            const status = await read();
            if (status.ended || status.metrics.error)
              throw new Error(status.metrics.error || 'Call ended');
            return status.metrics.response_active;
          },
          { timeout: 30000, intervals: [400] },
        )
        .toBe(false);
      await expect
        .poll(() => page.evaluate(() => (window as any).warmVoice.rms < 0.005))
        .toBe(true);
      await page.evaluate(async () => {
        const h = (window as any).warmVoice;
        const inbound: unknown[] = [];
        for (const peer of h.peers as RTCPeerConnection[]) {
          (await peer.getStats()).forEach((value) => {
            if (value.type === 'inbound-rtp') inbound.push(value);
          });
        }
        h.trial = {
          previousEpoch: h.epoch,
          inboundBefore: inbound,
          startAt: performance.now(),
          endAt: null,
          audioAt: null,
          videoAt: null,
          prematureOutput: false,
        };
        const source = h.input.createBufferSource();
        source.buffer = h.buffer;
        source.connect(h.destination);
        source.onended = () => {
          h.trial.endAt = performance.now();
          source.disconnect();
        };
        source.start();
      });
      await expect
        .poll(
          () =>
            page.evaluate(() => {
              const t = (window as any).warmVoice.trial;
              return Boolean(t.endAt && t.audioAt && t.videoAt);
            }),
          { timeout: 30000, intervals: [100] },
        )
        .toBe(true);
      const measured = await page.evaluate(() => (window as any).warmVoice.trial);
      expect(
        measured.prematureOutput,
        'Output began before the synthetic prompt ended; latency is not a valid complete-turn sample',
      ).toBe(false);
      const callbackSlip = measured.endAt - measured.startAt - report.input.spoken_seconds * 1000;
      expect(
        Math.abs(callbackSlip),
        'Input completion callback was delayed; do not understate response latency',
      ).toBeLessThan(100);
      const status = await read();
      expect(status.metrics.effective_voice).toBe(report.voice);
      expect(status.metrics.error).toBeFalsy();
      report.turns.push({
        index,
        warm: index > 0,
        audio_ms: measured.audioAt - measured.endAt,
        input_end_callback_slip_ms: callbackSlip,
        video_compositor_ms: measured.videoAt - measured.endAt,
        synchronized_observed_ms: Math.max(measured.audioAt, measured.videoAt) - measured.endAt,
        gpu_execution: status.metrics.gpu_info?.execution,
        generated_frames: status.metrics.generated_frames,
        gpu_allocated_bytes: status.metrics.gpu_memory?.gpu_allocated_bytes,
        voice_timing: status.metrics.voice_timing,
        render_stalls: status.metrics.render_stalls,
      });
      await save();
    }
    const values = report.turns
      .filter((t: any) => t.warm)
      .map((t: any) => t.synchronized_observed_ms)
      .sort((a: number, b: number) => a - b);
    report.warm_turn_count = values.length;
    report.p50_ms = values[Math.ceil(values.length * 0.5) - 1];
    report.p95_ms = values[Math.ceil(values.length * 0.95) - 1];
    report.meets_3000ms_target = report.p95_ms <= 3000;
    if (sessionSeconds) {
      await expect
        .poll(
          () => {
            if (monitorError) throw monitorError;
            return Boolean(report.automatic_end);
          },
          { timeout: 120000, intervals: [1000] },
        )
        .toBe(true);
      monitoring = false;
      await monitor;
      report.browser_continuity = await page.evaluate(() => (window as any).warmVoice.continuity);
      const samples = report.continuity_samples;
      const allocated = samples
        .filter((s: any) => s.gpu_allocated_bytes)
        .map((s: any) => s.gpu_allocated_bytes);
      report.continuity_summary = {
        samples: samples.length,
        max_pending_audio_seconds: Math.max(...samples.map((s: any) => s.pending_audio_seconds)),
        max_decoded_queue: Math.max(...samples.map((s: any) => s.decoded_batch_queue)),
        max_gpu_allocated_bytes: Math.max(...allocated),
        min_gpu_allocated_bytes: Math.min(...allocated),
      };
      await save();
      expect(report.automatic_end.reason).toBe('session_limit');
      expect(report.automatic_end.elapsed_seconds).toBeGreaterThanOrEqual(598);
      expect(report.automatic_end.elapsed_seconds).toBeLessThan(605);
      expect(report.browser_continuity.wrongGeometry).toBe(0);
      expect(report.browser_continuity.maxGapMs).toBeLessThan(2000);
      expect(report.continuity_summary.max_pending_audio_seconds).toBeLessThanOrEqual(20);
      expect(report.continuity_summary.max_decoded_queue).toBeLessThanOrEqual(2);
      expect(samples.every((s: any) => s.render_stalls === 0)).toBe(true);
      expect(
        report.continuity_summary.max_gpu_allocated_bytes -
          report.continuity_summary.min_gpu_allocated_bytes,
      ).toBeLessThan(512 * 1024 * 1024);
      await expect(page.getByRole('button', { name: 'Start call', exact: true })).toBeEnabled({
        timeout: 15000,
      });
      await expect
        .poll(async () => (await (await page.request.get('/api/health')).json()).active_session)
        .toBe(false);
      report.continuity_passed = true;
    }
    report.status = 'completed';
    report.final_metrics = (await read()).metrics;
    await save();
    expect(report.p95_ms).toBeLessThanOrEqual(3000);
  } catch (error) {
    report.failure = String(error);
    if (report.status !== 'completed') report.status = 'failed';
    report.last_trial = await page
      .evaluate(async () => {
        const h = (window as any).warmVoice,
          stats: unknown[] = [];
        for (const peer of h.peers as RTCPeerConnection[]) {
          (await peer.getStats()).forEach((value) => {
            if (value.type === 'inbound-rtp') stats.push(value);
          });
        }
        return {
          trial: h.trial,
          rms: h.rms,
          epoch: h.epoch,
          stats,
          outputContexts: h.analysers
            .filter((node: AnalyserNode) => node.fftSize === 128 && node.context !== h.input)
            .map((node: AnalyserNode) => ({
              state: node.context.state,
              time: node.context.currentTime,
            })),
          elements: Array.from(document.querySelectorAll('audio')).map((el) => ({
            paused: el.paused,
            gate: el.dataset.playbackOpen,
          })),
        };
      })
      .catch(() => null);
    if (session?.id) {
      const response = await page.request
        .get(`/api/sessions/${session.id}`, { headers: { 'X-Session-Key': session.session_key } })
        .catch(() => null);
      report.final_metrics = response?.ok() ? (await response.json()).metrics : null;
    }
    await save();
    throw error;
  } finally {
    monitoring = false;
    await monitor;
    await page
      .evaluate(() => {
        const h = (window as any).warmVoice;
        h.active = false;
        void h.input?.close();
      })
      .catch(() => {});
    if (session?.id)
      await page.request
        .delete(`/api/sessions/${session.id}`, {
          headers: { 'X-Session-Key': session.session_key },
        })
        .catch(() => {});
  }
});
