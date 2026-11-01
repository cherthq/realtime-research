import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Room, RoomEvent, RemoteAudioTrack, RemoteVideoTrack } from 'livekit-client';
import MetadataWorker from 'livekit-client/frame-metadata-worker?worker';
import { useCamera } from './useCamera';
import { BrowserPlayback } from './playback';
import { Conversation } from './Conversation';
import { LiveCaptions } from './LiveCaptions';
import { DeviceSettings } from './DeviceSettings';
import { useTranscripts } from './useTranscripts';
import { useCallQueue } from './useCallQueue';
import './style.css';

type Avatar = {
  id: string;
  name: string;
  image_url: string;
  image_available: boolean;
  casting_status: string;
  voice_profile_id?: string;
  voice_note?: string;
  note?: string;
};
type Voice = {
  id: string;
  name?: string;
  preview_available?: boolean;
  review: { description: string; presentation: string };
};
type AvatarModel = { id: string; name: string; available: boolean; description: string };
type Scenario = { id: string; title: string; text: string; available: boolean };
type Health = {
  geometry?: { width: number; height: number; fps: number; publish_fps: number };
  livekit: boolean;
  openai_configured: boolean;
  active_session: boolean;
  control_transport?: 'sse' | 'poll';
  ui_asset?: string | null;
  camera_vision?: boolean;
  tts_provider?: string;
  tts_configured?: boolean;
};
type Metrics = {
  state: string;
  elapsed_seconds: number;
  transport_fps: number;
  video_max_fps?: number;
  frames_published: number;
  audio_queue_ms: number;
  first_media_ms: number | null;
  server_cancel_ms: number | null;
  interruptions: number;
  mic_rms: number;
  mic_packets: number;
  error: string | null;
  model_generation_fps?: number | null;
  generated_frames?: number;
  held_frames?: number;
  idle_motion_frames?: number;
  continuous_motion?: boolean;
  video_delayed?: boolean;
  avatar_timing?: {
    first_audio_to_gpu_request_ms: number | null;
    first_chunk_roundtrip_ms: number | null;
    decoded_to_publish_ms: number | null;
  };
  render_stalls?: number;
  distinct_generated_frames?: number;
  exact_duplicate_frames?: number;
  distinct_generation_fps?: number | null;
  pending_audio_seconds?: number;
  decoded_batch_queue?: number;
  gpu_memory?: { gpu_allocated_bytes: number; gpu_peak_allocated_bytes: number } | null;
  browser_playback: {
    reports: number;
    cancel_epoch: number;
    cancel_ack_roundtrip_ms: number | null;
    offset_source: string;
  };
};
type Session = {
  id: string;
  session_key: string;
  ended: boolean;
  metrics: Metrics;
  connection: { url: string; token: string };
};

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function api<T>(path: string, options: RequestInit = {}, key?: string): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(key ? { 'X-Session-Key': key } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    if (response.status === 401 && path !== '/access')
      window.dispatchEvent(new Event('chert-access-expired'));
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`,
      response.status,
    );
  }
  return response.json();
}

function Brand() {
  return (
    <span className="wordmark">
      <img src="/chert-logo.png" alt="" width="32" height="32" />
      chert
    </span>
  );
}

function Icon({
  name,
}: {
  name:
    | 'phone'
    | 'mic'
    | 'mute'
    | 'stop'
    | 'play'
    | 'camera'
    | 'cameraOff'
    | 'settings'
    | 'close'
    | 'captions'
    | 'history'
    | 'clock';
}) {
  const paths: Record<string, React.ReactNode> = {
    phone: (
      <path d="M5 3h4l2 5-3 2c1.3 2.7 3.3 4.7 6 6l2-3 5 2v4c0 1-1 2-2 2C9.6 20.4 3.6 14.4 3 5c0-1 1-2 2-2Z" />
    ),
    mic: (
      <>
        <rect x="9" y="2" width="6" height="13" rx="3" />
        <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8" />
      </>
    ),
    mute: (
      <>
        <path d="m3 3 18 18M9 9v3a3 3 0 0 0 5 2M9 5V4a3 3 0 0 1 6 0v7M5 10v2a7 7 0 0 0 12 5M19 10v2M12 19v3" />
      </>
    ),
    stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
    play: <path d="m8 5 12 7-12 7Z" />,
    camera: (
      <>
        <rect x="3" y="6" width="12" height="12" rx="2" />
        <path d="m15 10 6-3v10l-6-3" />
      </>
    ),
    cameraOff: (
      <>
        <path d="m3 3 18 18M7 6h6a2 2 0 0 1 2 2v2l6-3v10M3 8v8a2 2 0 0 0 2 2h8M15 14v2" />
      </>
    ),
    settings: (
      <>
        <path d="M4 7h16M4 17h16" />
        <circle cx="9" cy="7" r="3" fill="currentColor" />
        <circle cx="15" cy="17" r="3" fill="currentColor" />
      </>
    ),
    captions: (
      <>
        <rect x="3" y="5" width="18" height="14" rx="3" />
        <path d="M10 10H7v4h3m7-4h-3v4h3" />
      </>
    ),
    history: (
      <>
        <path d="M8 20 3 22V6a3 3 0 0 1 3-3h12a3 3 0 0 1 3 3v11a3 3 0 0 1-3 3H8Z" />
        <path d="M7 9h10M7 14h6" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    close: <path d="m6 6 12 12M18 6 6 18" />,
  };
  return (
    <svg
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

function AccessGate() {
  const [access, setAccess] = useState<{
    required: boolean;
    authenticated: boolean;
    queue_required?: boolean;
  } | null>(null);
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  useEffect(() => {
    const check = () => {
      void api<{ required: boolean; authenticated: boolean }>('/access')
        .then(setAccess)
        .catch((e) => setError(e.message));
    };
    check();
    window.addEventListener('chert-access-expired', check);
    return () => window.removeEventListener('chert-access-expired', check);
  }, []);
  async function unlock(event: React.FormEvent) {
    event.preventDefault();
    setPending(true);
    setError('');
    try {
      setAccess(await api('/access', { method: 'POST', body: JSON.stringify({ token }) }));
      setToken('');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  async function lock() {
    try {
      setAccess(await api('/access', { method: 'DELETE' }));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  if (access?.authenticated)
    return (
      <App
        queueRequired={access.queue_required === true}
        onLock={access.required ? lock : undefined}
      />
    );
  return (
    <main className="access-page">
      <Brand />
      {access ? (
        <form onSubmit={(event) => void unlock(event)}>
          <h1>Join the demo</h1>
          <label htmlFor="access-code">Access code</label>
          <input
            id="access-code"
            type="password"
            autoComplete="current-password"
            required
            maxLength={256}
            value={token}
            onChange={(event) => setToken(event.target.value)}
          />
          <button className="call-button" disabled={pending}>
            {pending ? 'Opening…' : 'Continue'}
          </button>
        </form>
      ) : (
        <p>Connecting…</p>
      )}
      {error && <p role="alert">{error}</p>}
    </main>
  );
}

function App({
  onLock,
  queueRequired = false,
}: {
  onLock?: () => Promise<void>;
  queueRequired?: boolean;
}) {
  const queue = useCallQueue(queueRequired);
  const [avatars, setAvatars] = useState<Avatar[]>([]);
  const [cases, setCases] = useState<Scenario[]>([]);
  const [avatarId, setAvatarId] = useState('call01');
  const [health, setHealth] = useState<Health | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [pending, setPending] = useState(false);
  const [mic, setMic] = useState(false);
  const [soundBlocked, setSoundBlocked] = useState(false);
  const [hasVideo, setHasVideo] = useState(false);
  const [deliveryDelayed, setDeliveryDelayed] = useState(false);
  const [error, setError] = useState('');
  const [caseId, setCaseId] = useState('s01_greeting');
  const [controlInterrupted, setControlInterrupted] = useState(false);
  const [mediaInterrupted, setMediaInterrupted] = useState(false);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [models, setModels] = useState<AvatarModel[]>([]);
  const [modelId, setModelId] = useState('mock');
  const [voiceId, setVoiceId] = useState('');
  const [mode, setMode] = useState('diagnostic');
  const [casting, setCasting] = useState(false);
  const previewAudio = useRef<HTMLAudioElement | null>(null);
  const room = useRef<Room | null>(null);
  const playback = useRef<BrowserPlayback | null>(null);
  const metadataWorker = useRef<Worker | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const current = useRef<Session | null>(null);
  const updates = useRef<AbortController | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const audio = useRef<HTMLDivElement>(null);
  const stopping = useRef(false);
  const startAttempt = useRef(0);
  const preferences = useRef<HTMLDialogElement>(null);
  const history = useRef<HTMLDialogElement>(null);
  // Choose the initial default only; resizing must preserve the user's toggle.
  const [showCaptions, setShowCaptions] = useState(
    () => !window.matchMedia('(max-width: 760px)').matches,
  );
  const [showTimer, setShowTimer] = useState(true);
  const [cameraLarge, setCameraLarge] = useState(false);
  const [inputDevice, setInputDevice] = useState('');
  const [outputDevice, setOutputDevice] = useState('');
  const [cameraRoom, setCameraRoom] = useState<Room | null>(null);
  const [captionRoom, setCaptionRoom] = useState<Room | null>(null);
  const videoHold = useRef<HTMLCanvasElement>(null);
  const camera = useCamera(cameraRoom);
  const captions = useTranscripts(session, health?.control_transport === 'poll', captionRoom);
  const avatar = avatars.find((a) => a.id === avatarId);
  const defaultVoice =
    voices.find((voice) => voice.id === avatar?.voice_profile_id) ||
    (health?.tts_provider === 'elevenlabs' && !avatar?.voice_profile_id ? voices[0] : undefined);
  const selectedVoice = voiceId ? voices.find((voice) => voice.id === voiceId) : defaultVoice;
  function selectAvatar(id: string) {
    previewAudio.current?.pause();
    setAvatarId(id);
    setVoiceId('');
  }

  useEffect(() => {
    if (!session) {
      setDeliveryDelayed(false);
      return;
    }
    const timer = setInterval(
      () => setDeliveryDelayed(playback.current?.videoDelayed ?? false),
      250,
    );
    return () => clearInterval(timer);
  }, [session?.id]);

  useEffect(() => {
    Promise.all([
      api<Avatar[]>('/avatars'),
      api<Scenario[]>('/scenarios'),
      api<Health>('/health'),
      api<Voice[]>('/voices'),
      api<AvatarModel[]>('/models'),
    ])
      .then(([a, c, h, v, m]) => {
        setAvatars(a);
        setAvatarId((selected) =>
          a.some((avatar) => avatar.id === selected) ? selected : (a[0]?.id ?? ''),
        );
        setCases(c);
        setHealth(h);
        setVoices(v);
        setModels(m);
        if (h.openai_configured && v.length) setMode('voice_diagnostic');
        if (m.find((model) => model.id === 'avatarforever')?.available) setModelId('avatarforever');
      })
      .catch((e) => setError(e.message));
    const interval = setInterval(
      () =>
        api<Health>('/health')
          .then(setHealth)
          .catch(() => setHealth(null)),
      5000,
    );
    const onLeave = () => {
      startAttempt.current++;
      const s = current.current;
      if (s)
        void fetch(`/api/sessions/${s.id}`, {
          method: 'DELETE',
          headers: { 'X-Session-Key': s.session_key },
          keepalive: true,
        });
    };
    window.addEventListener('pagehide', onLeave);
    return () => {
      clearInterval(interval);
      window.removeEventListener('pagehide', onLeave);
      onLeave();
      previewAudio.current?.pause();
      updates.current?.abort();
      playback.current?.close();
      metadataWorker.current?.terminate();
      void room.current?.disconnect();
      void audioContext.current?.close();
    };
  }, []);

  async function observe(s: Session, signal: AbortSignal) {
    const pause = (ms: number) =>
      new Promise<void>((resolve) => {
        const done = () => {
          clearTimeout(timer);
          signal.removeEventListener('abort', done);
          resolve();
        };
        const timer = setTimeout(done, ms);
        signal.addEventListener('abort', done, { once: true });
        if (signal.aborted) done();
      });
    async function apply(state: Session) {
      if (signal.aborted || current.current?.id !== s.id) return;
      setMetrics(state.metrics);
      playback.current?.cancelled(state.metrics.browser_playback.cancel_epoch);
      playback.current?.resume('status');
      setControlInterrupted(false);
      if (state.metrics.error) setError(state.metrics.error);
      if (state.ended) await stop();
    }
    const fatal = (e: unknown) => e instanceof ApiError && [401, 403, 404, 410].includes(e.status);
    async function poll(failedSince: number | null = null) {
      while (!signal.aborted) {
        try {
          const remaining =
            failedSince === null
              ? 3000
              : Math.max(1, Math.ceil(30000 - (performance.now() - failedSince)));
          await apply(
            await api<Session>(
              `/sessions/${s.id}`,
              { signal: AbortSignal.any([signal, AbortSignal.timeout(Math.min(3000, remaining))]) },
              s.session_key,
            ),
          );
          failedSince = null;
        } catch (e) {
          if (signal.aborted) return;
          if (fatal(e)) throw e;
          failedSince ??= performance.now();
          playback.current?.hold('status');
          setControlInterrupted(true);
          if (performance.now() - failedSince >= 30000)
            throw new Error(
              'Call control did not recover within 30 seconds. Start a new call to retry.',
            );
        }
        await pause(failedSince === null ? 100 : 500);
      }
    }
    try {
      // The Windows loopback bridge buffers each HTTP response. Keep SSE for
      // normal hosts and use bounded sequential requests for this local route.
      if (health?.control_transport === 'poll') {
        await poll();
        return;
      }
      // Fall back to bounded polling if SSE drops or stops delivering updates.
      // Preserve this session and voice; expired credentials still end the call.
      const stream = new AbortController();
      const headersDeadline = setTimeout(() => stream.abort(), 3000);
      try {
        const response = await fetch(`/api/sessions/${s.id}/events`, {
          headers: { 'X-Session-Key': s.session_key },
          signal: AbortSignal.any([signal, stream.signal]),
        });
        clearTimeout(headersDeadline);
        if (!response.ok) throw new ApiError('Session status connection failed.', response.status);
        if (!response.body) throw new Error('Session status connection failed.');
        const reader = response.body.getReader(),
          decoder = new TextDecoder();
        let buffer = '';
        while (!signal.aborted) {
          const deadline = setTimeout(() => stream.abort(), 3000);
          let item: ReadableStreamReadResult<Uint8Array>;
          try {
            item = await reader.read();
          } finally {
            clearTimeout(deadline);
          }
          if (item.done) throw new Error('Call control connection ended.');
          buffer += decoder.decode(item.value, { stream: true });
          if (buffer.length > 1024 * 1024) throw new Error('Session status exceeded its limit.');
          let boundary: number;
          while ((boundary = buffer.indexOf('\n\n')) >= 0) {
            const event = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            if (event.startsWith('data: ')) {
              await apply(JSON.parse(event.slice(6)) as Session);
              if (signal.aborted) return;
            }
          }
        }
      } catch (e) {
        if (signal.aborted) return;
        if (fatal(e)) throw e;
        playback.current?.hold('status');
        setControlInterrupted(true);
      } finally {
        clearTimeout(headersDeadline);
        stream.abort();
      }
      if (!signal.aborted) await poll(performance.now());
    } catch (e) {
      if (!signal.aborted) {
        setError((e as Error).message);
        await stop();
      }
    }
  }

  async function start() {
    if (pending || current.current) return;
    const attempt = ++startAttempt.current;
    setError('');
    setPending(true);
    stopping.current = false;
    previewAudio.current?.pause();
    try {
      // Start the output context during the user's click, before network awaits.
      const sound = new AudioContext({ latencyHint: 'interactive' });
      audioContext.current = sound;
      void sound.resume().catch(() => setSoundBlocked(true));
      if (outputDevice)
        await (sound as AudioContext & { setSinkId: (id: string) => Promise<void> }).setSinkId(
          outputDevice,
        );
      let queueKey = '';
      if (queue.enabled) {
        if (queue.place?.state === 'ready' && queue.remaining > 0) {
          queueKey = queue.key();
        } else {
          const joined = await queue.join();
          if (joined.place.state !== 'ready') {
            await sound.close();
            audioContext.current = null;
            return; // Waiting does not request the mic, LiveKit or GPU.
          }
          queueKey = joined.key;
        }
        queue.pause();
      }
      if (attempt !== startAttempt.current) {
        if (queue.enabled) await queue.leave();
        await sound.close().catch(() => {});
        return;
      }
      const s = await api<Session>('/sessions', {
        method: 'POST',
        headers: queueKey ? { 'X-Queue-Key': queueKey } : {},
        body: JSON.stringify({
          avatar_id: avatarId,
          model_id: modelId,
          mode,
          voice_profile_id: voiceId || null,
        }),
      });
      if (queue.enabled) queue.clear();
      if (attempt !== startAttempt.current) {
        await api(
          `/sessions/${s.id}`,
          { method: 'DELETE', signal: AbortSignal.timeout(3000) },
          s.session_key,
        ).catch(() => {});
        return;
      }
      current.current = s;
      setSession(s);
      setMetrics(s.metrics);
      if (mode === 'voice_diagnostic')
        void api<Avatar[]>('/avatars')
          .then(setAvatars)
          .catch(() => {});
      const metadata = new MetadataWorker();
      metadataWorker.current = metadata;
      const control = new BrowserPlayback(
        (body) =>
          api(
            `/sessions/${s.id}/playback`,
            { method: 'POST', body: JSON.stringify(body), signal: AbortSignal.timeout(2000) },
            s.session_key,
          ),
        (visible) => {
          if (current.current?.id === s.id) setHasVideo(visible);
        },
        () => setSoundBlocked(true),
        (message) => {
          setError(message);
          void stop();
        },
        sound,
        s.metrics.continuous_motion === true,
      );
      playback.current = control;
      const r = new Room({
        adaptiveStream: false,
        dynacast: false,
        webAudioMix: { audioContext: sound },
        frameMetadata: { worker: metadata },
        audioCaptureDefaults: {
          ...(inputDevice ? { deviceId: { exact: inputDevice } } : {}),
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      room.current = r;
      r.on(RoomEvent.TrackSubscribed, (track, _publication, participant) => {
        if (participant.identity !== 'avatar') return;
        if (track instanceof RemoteVideoTrack && video.current) {
          track.attach(video.current);
          control.attachVideo(track, video.current, videoHold.current ?? undefined);
        }
        if (track instanceof RemoteAudioTrack && audio.current) {
          const el = control.attachAudio(track);
          if (el) audio.current.appendChild(el);
        }
      });
      r.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((el) => {
          if (el.tagName === 'AUDIO') {
            control.detachAudio(el as HTMLAudioElement);
            el.remove();
          }
        });
      });
      r.on(RoomEvent.DataReceived, (payload, participant, _kind, topic) => {
        if (participant?.identity !== 'avatar' || topic !== 'chert-control' || payload.length > 256)
          return;
        try {
          const event = JSON.parse(new TextDecoder().decode(payload));
          if (
            event.type === 'cancel' &&
            Number.isInteger(event.epoch) &&
            event.epoch >= 0 &&
            event.epoch < 65535
          )
            control.cancelled(event.epoch);
        } catch {
          /* Invalid packets cannot change playback state. */
        }
      });
      r.on(RoomEvent.Disconnected, () => {
        if (!stopping.current && current.current?.id === s.id) void stop();
      });
      r.on(RoomEvent.Reconnecting, () => {
        if (current.current?.id === s.id) {
          control.hold('media');
          setMediaInterrupted(true);
        }
      });
      r.on(RoomEvent.Reconnected, () => {
        if (current.current?.id === s.id) {
          control.resume('media');
          setMediaInterrupted(false);
        }
      });
      await r.connect(s.connection.url, s.connection.token);
      if (attempt !== startAttempt.current || current.current?.id !== s.id) {
        await r.disconnect();
        return;
      }
      await r.startAudio().catch(() => setSoundBlocked(true));
      setCaptionRoom(r);
      if (attempt !== startAttempt.current || current.current?.id !== s.id) {
        await r.disconnect();
        return;
      }
      const controller = new AbortController();
      updates.current = controller;
      void observe(s, controller.signal);
      if (s.metrics.continuous_motion) {
        // Keep setup pending until listening video is actually being presented.
        // Do not enable the microphone against a static reference portrait.
        await control.waitForVideo();
        if (attempt !== startAttempt.current || current.current?.id !== s.id) return;
      }
      if (mode === 'voice_diagnostic') {
        if (health?.camera_vision) setCameraRoom(r);
        try {
          await r.localParticipant.setMicrophoneEnabled(true);
          if (current.current?.id === s.id) setMic(true);
          else await r.localParticipant.setMicrophoneEnabled(false);
        } catch {
          setError(
            'Microphone access is needed to talk. Allow it in your browser, then use the mic button.',
          );
        }
      }
    } catch (e) {
      if (attempt === startAttempt.current) {
        if (queue.enabled) await queue.leave();
        setError((e as Error).message);
        await stop();
      }
    } finally {
      if (attempt === startAttempt.current) setPending(false);
    }
  }

  async function stop() {
    if (stopping.current) return;
    startAttempt.current++;
    stopping.current = true;
    setPending(true);
    setCameraRoom(null);
    setCaptionRoom(null);
    camera.stop();
    playback.current?.close();
    playback.current = null;
    updates.current?.abort();
    updates.current = null;
    const s = current.current;
    current.current = null;
    const r = room.current;
    room.current = null;
    await r?.disconnect().catch((e) => setError(e.message));
    await audioContext.current?.close().catch((e) => setError(e.message));
    audioContext.current = null;
    metadataWorker.current?.terminate();
    metadataWorker.current = null;
    audio.current?.replaceChildren();
    if (video.current) video.current.srcObject = null;
    setSession(null);
    setMic(false);
    setHasVideo(false);
    setSoundBlocked(false);
    setControlInterrupted(false);
    setMediaInterrupted(false);
    if (s)
      await api(
        `/sessions/${s.id}`,
        { method: 'DELETE', signal: AbortSignal.timeout(3000) },
        s.session_key,
      ).catch((e) => setError((previous) => previous || e.message));
    stopping.current = false;
    setPending(false);
  }

  async function toggleMic() {
    try {
      await room.current?.localParticipant.setMicrophoneEnabled(!mic);
      setMic(!mic);
      setError('');
    } catch {
      setError('Microphone unavailable or permission denied. Scripted playback still works.');
    }
  }

  async function play(id: string) {
    const s = current.current;
    if (!s) return;
    try {
      await room.current?.startAudio();
      setSoundBlocked(false);
      await api(
        `/sessions/${s.id}/play`,
        { method: 'POST', body: JSON.stringify({ scenario_id: id }) },
        s.session_key,
      );
      setError('');
      preferences.current?.close();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function interrupt() {
    const s = current.current;
    if (!s) return;
    const control = playback.current,
      sequence = control?.interruptLocally();
    try {
      const result = await api<Session>(
        `/sessions/${s.id}/interrupt`,
        { method: 'POST' },
        s.session_key,
      );
      if (current.current?.id === s.id && sequence !== undefined)
        control?.interruptConfirmed(sequence, result.metrics.browser_playback.cancel_epoch);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function castVoice() {
    setCasting(true);
    setError('');
    try {
      const decision = await api<{ status: string }>('/avatars/' + avatarId + '/cast-voice', {
        method: 'POST',
        body: JSON.stringify({ recast: !!avatar?.voice_profile_id }),
      });
      setAvatars(await api<Avatar[]>('/avatars'));
      if (decision.status !== 'matched')
        setError(
          'Voice matching needs a preference or an explicit rematch. You can select a prepared voice.',
        );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCasting(false);
    }
  }

  function previewVoice() {
    const id = selectedVoice?.id;
    if (!id) return;
    previewAudio.current?.pause();
    const preview = new Audio('/api/voices/' + encodeURIComponent(id) + '/preview');
    previewAudio.current = preview;
    void preview.play().catch(() => setError('Voice preview could not play.'));
  }

  async function selectDevice(kind: MediaDeviceKind, id: string) {
    if (kind === 'videoinput') {
      await camera.selectDevice(id);
    } else if (kind === 'audioinput') {
      if (room.current && !(await room.current.switchActiveDevice(kind, id || 'default'))) {
        throw new Error('Microphone switch failed');
      }
      setInputDevice(id);
    } else {
      // Playback uses Web Audio. Switch that destination, not only the <audio> element.
      const context = audioContext.current as
        | (AudioContext & { setSinkId: (id: string) => Promise<void> })
        | null;
      if (context) await context.setSinkId(id);
      setOutputDevice(id);
    }
  }

  const elapsed = session ? (metrics?.elapsed_seconds ?? 0) : 0;
  const duration = `${Math.floor(elapsed / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(elapsed % 60)
    .toString()
    .padStart(2, '0')}`;
  const state = pending
    ? 'Connecting…'
    : queue.queued
      ? queue.place?.state === 'ready'
        ? 'Your turn'
        : 'Waiting'
      : !session
        ? health?.livekit
          ? 'Ready'
          : 'Offline'
        : controlInterrupted || mediaInterrupted
          ? 'Reconnecting…'
          : metrics?.state === 'responding'
            ? 'Speaking'
            : captions.checking
              ? 'Checking…'
              : metrics?.state === 'buffering'
                ? 'Thinking…'
                : 'Connected';
  const locked = !!session || pending;
  const updateAvailable =
    !import.meta.env.DEV &&
    !!health?.ui_asset &&
    health.ui_asset !== new URL(import.meta.url).pathname;
  const startLabel = pending
    ? 'Connecting…'
    : queue.place?.state === 'ready'
      ? 'Start your call'
      : queue.enabled && health?.active_session
        ? 'Join queue'
        : 'Start call';

  return (
    <main className={`call-page ${queue.queued ? 'has-queue' : ''}`}>
      <h1 className="sr-only">Chert video call</h1>

      {error && (
        <div role="alert" className="error-message">
          <span>{error}</span>
          <button onClick={() => setError('')} aria-label="Dismiss error">
            <Icon name="close" />
          </button>
        </div>
      )}
      {!error && (controlInterrupted || mediaInterrupted) && (
        <div role="status" aria-label="Connection recovery" className="error-message">
          Connection interrupted. Playback is paused while reconnecting…
        </div>
      )}
      {updateAvailable && (
        <div role="status" className="error-message">
          <span>
            {session
              ? 'An update is ready. End this call, then reload.'
              : 'An update is ready. Reload before starting a call.'}
          </span>
          {!session && <button onClick={() => window.location.reload()}>Reload</button>}
        </div>
      )}

      <section
        className={`video-pair ${cameraLarge ? 'camera-large' : 'avatar-large'}`}
        aria-label="Video call"
      >
        <div className={`video-tile local-tile ${cameraLarge ? 'large-tile' : 'small-tile'}`}>
          <video
            ref={camera.video}
            autoPlay
            playsInline
            muted
            width={640}
            height={480}
            className={`local-video ${camera.enabled ? 'visible' : ''}`}
            aria-label="Your camera"
          />
          {!camera.enabled && (
            <div className="camera-placeholder">
              <Icon name="cameraOff" />
              <p>{camera.pending ? 'Allow camera access' : 'Camera off'}</p>
            </div>
          )}
          {!cameraLarge && (
            <button
              className="swap-view"
              onClick={() => setCameraLarge(true)}
              aria-label="Make your camera large"
              title="Make your camera large"
            >
              <span className="swap-hint">Enlarge ↗</span>
            </button>
          )}
          <div className="tile-caption">
            <span>You</span>
            <span className="tile-detail">
              {camera.shared ? 'Shared with AI' : camera.enabled ? 'Local preview' : 'Camera off'}
            </span>
          </div>
        </div>
        <div className={`video-tile avatar-tile ${cameraLarge ? 'small-tile' : 'large-tile'}`}>
          {!hasVideo && !session && !pending && avatar?.image_available && (
            <img
              className="avatar-image"
              src={avatar.image_url}
              alt="Avatar reference"
              width={640}
              height={480}
              fetchPriority="high"
            />
          )}
          <video
            ref={video}
            autoPlay
            playsInline
            muted
            width={640}
            height={480}
            className={hasVideo ? 'visible' : ''}
            aria-label="Generated avatar"
          />
          <canvas
            ref={videoHold}
            className="avatar-hold"
            width={640}
            height={480}
            hidden
            aria-hidden="true"
          />
          {!avatar?.image_available && !hasVideo && (
            <div className="camera-placeholder">
              <p>Avatar unavailable</p>
            </div>
          )}
          {!hasVideo && avatar?.image_available && (session || pending) && (
            <div className="camera-placeholder" role="status" aria-label="Avatar starting">
              <p>Connecting avatar…</p>
            </div>
          )}
          {cameraLarge && (
            <button
              className="swap-view"
              onClick={() => setCameraLarge(false)}
              aria-label="Make the AI video large"
              title="Make the AI video large"
            >
              <span className="swap-hint">Enlarge ↗</span>
            </button>
          )}
          <div className="stage-brand">
            <Brand />
          </div>
          {session && hasVideo && (metrics?.video_delayed || deliveryDelayed) && (
            <span className="video-delay" role="status">
              Video delayed
            </span>
          )}
        </div>
        <div className="stage-overlay">
          <header className="stage-header">
            <div className="call-status">
              <span className={`status-dot ${session ? 'connected' : ''}`} />
              <span role="status">{state}</span>
              {showTimer && session && <time aria-label="Call duration">{duration}</time>}
            </div>
          </header>
          <LiveCaptions
            utterances={captions.utterances}
            reconnecting={captions.reconnecting}
            active={!!session}
            enabled={showCaptions}
          />
        </div>
      </section>
      <div ref={audio} className="audio-container" />

      <footer className="call-footer">
        {queue.queued && !session && !pending && (
          <div className="queue-status" role="status" aria-live="polite">
            <strong>
              {queue.place?.state === 'ready'
                ? 'It’s your turn.'
                : queue.place
                  ? `You’re #${queue.place.position} in line.`
                  : 'Checking your place…'}
            </strong>
            <span>
              {queue.notice ||
                (queue.place?.state === 'ready'
                  ? `Start within ${queue.remaining}s to keep your place.`
                  : 'One call at a time. Keep this page open; your place updates automatically.')}
            </span>
          </div>
        )}
        {!queue.queued && queue.notice && (
          <p className="queue-notice" role="status">
            {queue.notice}
          </p>
        )}
        <div className="call-controls" role="group" aria-label="Call controls">
          <button
            className={`icon-control ${camera.enabled ? 'active' : ''}`}
            onClick={() => void camera.toggle()}
            aria-label={
              camera.pending
                ? 'Cancel camera request'
                : camera.enabled
                  ? 'Turn camera off'
                  : 'Turn camera on'
            }
            title={
              camera.pending
                ? 'Cancel camera request'
                : camera.enabled
                  ? 'Turn camera off'
                  : 'Turn camera on'
            }
            aria-pressed={camera.enabled}
          >
            <Icon name={camera.enabled ? 'camera' : 'cameraOff'} />
          </button>
          <button
            className={`icon-control ${mic ? 'active' : ''}`}
            disabled={!session || pending}
            onClick={() => void toggleMic()}
            aria-label={mic ? 'Mute microphone' : 'Enable microphone'}
            title={mic ? 'Mute microphone' : 'Enable microphone'}
            aria-pressed={mic}
          >
            <Icon name={mic ? 'mic' : 'mute'} />
          </button>
          <button
            className={`icon-control ${showCaptions ? 'active' : ''}`}
            onClick={() => setShowCaptions(!showCaptions)}
            aria-label="Captions"
            title="Captions"
            aria-pressed={showCaptions}
          >
            <Icon name="captions" />
          </button>
          <button
            className="icon-control"
            onClick={() => history.current?.showModal()}
            aria-label="Conversation history"
            title="Conversation history"
            aria-haspopup="dialog"
          >
            <Icon name="history" />
          </button>
          <button
            className="icon-control"
            onClick={() => preferences.current?.showModal()}
            aria-label="Settings"
            title="Settings"
            aria-haspopup="dialog"
          >
            <Icon name="settings" />
          </button>
          {session ? (
            <button
              className="call-button end-call"
              onClick={() => void stop()}
              aria-label="End call"
              title="End call"
            >
              <Icon name="phone" />
            </button>
          ) : queue.queued && queue.place?.state !== 'ready' && !pending ? (
            <button
              className="call-button"
              onClick={() => void queue.leave()}
              aria-label="Leave queue"
              title="Leave queue"
            >
              <Icon name="close" />
              <span className="start-label">Leave queue</span>
            </button>
          ) : (
            <button
              className="call-button"
              disabled={
                pending ||
                !health?.livekit ||
                !avatar?.image_available ||
                updateAvailable ||
                (queue.place?.state === 'ready' && queue.remaining <= 0)
              }
              onClick={() => void start()}
              aria-label={startLabel}
              title={startLabel}
            >
              <Icon name="phone" />
              <span className="start-label">{startLabel}</span>
            </button>
          )}
          {queue.place?.state === 'ready' && !pending && !session && (
            <button
              className="icon-control"
              onClick={() => void queue.leave()}
              aria-label="Leave queue"
              title="Leave queue"
            >
              <Icon name="close" />
            </button>
          )}
        </div>
        {soundBlocked && (
          <button
            className="enable-sound"
            onClick={() => void room.current?.startAudio().then(() => setSoundBlocked(false))}
          >
            Enable sound
          </button>
        )}
        <p className="call-note">
          {modelId === 'mock'
            ? 'Preview · No avatar model connected.'
            : 'AI-generated voice and video.'}
        </p>
        {camera.error && (
          <p className="camera-sharing-error" role="status">
            {camera.error}
          </p>
        )}
      </footer>

      <dialog
        ref={history}
        className="history-dialog"
        aria-labelledby="history-title"
        onClick={(event) => {
          if (event.target === history.current) history.current.close();
        }}
      >
        <div className="dialog-content">
          <header className="dialog-header">
            <h2 id="history-title">Your conversation</h2>
            <button
              className="icon-control"
              aria-label="Close conversation"
              onClick={() => history.current?.close()}
            >
              <Icon name="close" />
            </button>
          </header>
          <Conversation
            utterances={captions.utterances}
            checking={captions.checking}
            active={!!session}
            reconnecting={captions.reconnecting}
          />
        </div>
      </dialog>

      <dialog
        ref={preferences}
        className="settings-dialog"
        aria-labelledby="settings-title"
        onClick={(event) => {
          if (event.target === preferences.current) preferences.current.close();
        }}
      >
        <div className="dialog-content">
          <header className="dialog-header">
            <h2 id="settings-title">Call settings</h2>
            <button
              className="icon-control"
              onClick={() => preferences.current?.close()}
              aria-label="Close settings"
            >
              <Icon name="close" />
            </button>
          </header>
          <section className="display-settings" aria-label="Display">
            <h3>Display</h3>
            <label className="toggle-row">
              <span>
                <Icon name="captions" />
                Live captions
              </span>
              <input
                type="checkbox"
                checked={showCaptions}
                onChange={(event) => setShowCaptions(event.target.checked)}
              />
            </label>
            <label className="toggle-row">
              <span>
                <Icon name="clock" />
                Call timer
              </span>
              <input
                type="checkbox"
                checked={showTimer}
                onChange={(event) => setShowTimer(event.target.checked)}
              />
            </label>
          </section>
          <DeviceSettings
            choices={{
              videoinput: camera.deviceId,
              audioinput: inputDevice,
              audiooutput: outputDevice,
            }}
            refreshKey={`${camera.enabled}:${camera.deviceId}:${session?.id}:${mic}`}
            disabled={pending || camera.pending}
            onSelect={selectDevice}
          />
          {session && (
            <button
              className="interrupt-action"
              onClick={() => {
                preferences.current?.close();
                void interrupt();
              }}
            >
              <Icon name="stop" />
              Interrupt response
            </button>
          )}
          <details className="setup-details" open={!locked}>
            <summary>Avatar & voice</summary>
            {locked && <p className="field-note">End the call to change the avatar or voice.</p>}
            <fieldset disabled={locked}>
              <legend>Avatar</legend>
              <div className="avatar-choices">
                {avatars.map((item) => (
                  <button
                    key={item.id}
                    className={avatarId === item.id ? 'selected' : ''}
                    onClick={() => selectAvatar(item.id)}
                    aria-label={`Select ${item.name}`}
                    aria-pressed={avatarId === item.id}
                  >
                    {item.image_available && (
                      <img src={item.image_url} alt="" width={640} height={480} />
                    )}
                    <span>{item.name}</span>
                  </button>
                ))}
              </div>
              <label htmlFor="model">Model</label>
              <select
                id="model"
                value={modelId}
                onChange={(event) => setModelId(event.target.value)}
              >
                {models.map((item) => (
                  <option key={item.id} value={item.id} disabled={!item.available}>
                    {item.name}
                    {!item.available ? ' · unavailable' : ''}
                  </option>
                ))}
              </select>
              <label htmlFor="mode">Conversation</label>
              <select id="mode" value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="diagnostic">Test audio</option>
                <option value="voice_diagnostic" disabled={!health?.openai_configured}>
                  Live voice
                </option>
              </select>
              {!health?.openai_configured && (
                <p className="field-note">
                  Add your OpenAI key on the server to enable live voice.
                </p>
              )}
              <label htmlFor="voice">Voice</label>
              <select
                id="voice"
                value={voiceId}
                onChange={(event) => {
                  previewAudio.current?.pause();
                  setVoiceId(event.target.value);
                }}
              >
                <option value="">
                  {health?.tts_provider === 'elevenlabs'
                    ? `${avatar?.voice_profile_id ? 'Matched' : 'Default'} voice${defaultVoice?.name ? ` · ${defaultVoice.name}` : ''}`
                    : `Match avatar${avatar?.voice_profile_id ? ` · ${avatar.voice_profile_id}` : ''}`}
                </option>
                {voices.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name || item.id} · {item.review.presentation}
                  </option>
                ))}
              </select>
              {health?.tts_provider === 'elevenlabs' && selectedVoice && (
                <div className="voice-actions">
                  <button onClick={previewVoice} disabled={!selectedVoice.preview_available}>
                    Preview voice
                  </button>
                  <span className="field-note">Stock voice sample</span>
                </div>
              )}
              {avatar?.voice_note && !voiceId && <p className="field-note">{avatar.voice_note}</p>}
              {avatar?.note && <p className="field-note">{avatar.note}</p>}
              {health?.tts_provider === 'elevenlabs' && (
                <p className="field-note">
                  {health.tts_configured
                    ? 'ElevenLabs speech is configured on the server.'
                    : 'Add the ElevenLabs key and voice ID on the server to enable speech.'}
                </p>
              )}
              {health?.tts_provider !== 'elevenlabs' && !voiceId && !avatar?.voice_profile_id && (
                <p className="field-note">
                  The first live call matches a voice to this character and saves it.
                </p>
              )}
              {health?.tts_provider !== 'elevenlabs' && (
                <div className="voice-actions">
                  <button
                    onClick={() => void castVoice()}
                    disabled={casting || !health?.openai_configured || !voices.length}
                  >
                    {casting ? 'Matching…' : avatar?.voice_profile_id ? 'Rematch' : 'Match voice'}
                  </button>
                  <button onClick={previewVoice} disabled={!(voiceId || avatar?.voice_profile_id)}>
                    Preview voice
                  </button>
                </div>
              )}
              {health?.tts_provider !== 'elevenlabs' &&
                avatar?.casting_status === 'needs_recast' && (
                  <p className="field-note">
                    Avatar or voice library changed. Rematch before calling.
                  </p>
                )}
            </fieldset>
            <section className="test-audio" aria-label="Test audio">
              <label htmlFor="test-reply">Test reply</label>
              <div className="test-row">
                <select
                  id="test-reply"
                  value={caseId}
                  onChange={(event) => setCaseId(event.target.value)}
                >
                  {cases.map((item) => (
                    <option value={item.id} key={item.id} disabled={!item.available}>
                      {item.title}
                    </option>
                  ))}
                </select>
                <button
                  disabled={!session || mode !== 'diagnostic'}
                  onClick={() => void play(caseId)}
                  aria-label="Play test reply"
                >
                  <Icon name="play" />
                </button>
              </div>
              <p className="field-note">Start a call in Test audio mode, then play a reply.</p>
            </section>
          </details>
          <details className="technical-details">
            <summary>Connection details</summary>
            <p>{models.find((item) => item.id === modelId)?.description}</p>
            <dl>
              <div>
                <dt>Output</dt>
                <dd>
                  {health?.geometry ? `${health.geometry.width} × ${health.geometry.height}` : '—'}
                </dd>
              </div>
              <div>
                <dt>Video before cap</dt>
                <dd>{metrics?.transport_fps ?? 'n/a'} fps</dd>
              </div>
              <div>
                <dt>Video frame-rate cap</dt>
                <dd>{metrics?.video_max_fps ?? 'n/a'} fps</dd>
              </div>
              <div>
                <dt>Audio queue</dt>
                <dd>{metrics?.audio_queue_ms ?? 'n/a'} ms</dd>
              </div>
              <div>
                <dt>First media ready</dt>
                <dd>{metrics?.first_media_ms ?? 'n/a'} ms</dd>
              </div>
              <div>
                <dt>Mic packets</dt>
                <dd data-testid="mic-packets">{metrics?.mic_packets ?? 0}</dd>
              </div>
            </dl>
            <dl>
              <div>
                <dt>Displayed-frame reports</dt>
                <dd data-testid="playback-reports">{metrics?.browser_playback.reports ?? 0}</dd>
              </div>
              <div>
                <dt>Cancel acknowledgement</dt>
                <dd>{metrics?.browser_playback.cancel_ack_roundtrip_ms?.toFixed(1) ?? 'n/a'} ms</dd>
              </div>
            </dl>
            {modelId !== 'mock' && (
              <dl>
                <div>
                  <dt>Generated video</dt>
                  <dd>{metrics?.model_generation_fps ?? 'n/a'} fps</dd>
                </div>
                <div>
                  <dt>Distinct generated video</dt>
                  <dd>
                    {metrics?.distinct_generation_fps ?? 'n/a'} fps ·{' '}
                    {metrics?.exact_duplicate_frames ?? 0} exact repeats
                  </dd>
                </div>
                <div>
                  <dt>Speech / cached idle / held frames</dt>
                  <dd>
                    {metrics?.generated_frames ?? 0} / {metrics?.idle_motion_frames ?? 0} /{' '}
                    {metrics?.held_frames ?? 0}
                  </dd>
                </div>
                <div>
                  <dt>First audio → render request</dt>
                  <dd>
                    {metrics?.avatar_timing?.first_audio_to_gpu_request_ms?.toFixed(0) ?? 'n/a'} ms
                  </dd>
                </div>
                <div>
                  <dt>First chunk generation + IPC</dt>
                  <dd>
                    {metrics?.avatar_timing?.first_chunk_roundtrip_ms?.toFixed(0) ?? 'n/a'} ms
                  </dd>
                </div>
                <div>
                  <dt>Decoded → publication</dt>
                  <dd>{metrics?.avatar_timing?.decoded_to_publish_ms?.toFixed(0) ?? 'n/a'} ms</dd>
                </div>
                <div>
                  <dt>Pending speech</dt>
                  <dd>{metrics?.pending_audio_seconds ?? 0} s</dd>
                </div>
                <div>
                  <dt>Decoded chunks queued</dt>
                  <dd>{metrics?.decoded_batch_queue ?? 0}</dd>
                </div>
                <div>
                  <dt>Render stalls</dt>
                  <dd>{metrics?.render_stalls ?? 0}</dd>
                </div>
                <div>
                  <dt>GPU allocated / peak</dt>
                  <dd>
                    {metrics?.gpu_memory
                      ? `${(metrics.gpu_memory.gpu_allocated_bytes / 2 ** 30).toFixed(1)} / ${(metrics.gpu_memory.gpu_peak_allocated_bytes / 2 ** 30).toFixed(1)} GiB`
                      : 'n/a'}
                  </dd>
                </div>
              </dl>
            )}
            <p>
              Frame feedback confirms video presentation with audio enabled; speaker latency is not
              measured.{' '}
              {health?.camera_vision
                ? 'During a live call, your enabled camera is shared with the AI. One recent image is used per spoken turn.'
                : 'Your camera stays on this device.'}{' '}
              This app does not save call recordings or transcripts. Conversation providers process
              shared audio, images and text under their own retention policies.
            </p>
          </details>
          {onLock && (
            <div className="voice-actions">
              <button onClick={() => void stop().then(onLock)}>Lock demo</button>
            </div>
          )}
        </div>
      </dialog>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<AccessGate />);
