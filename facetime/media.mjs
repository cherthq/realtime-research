import { AudioSource, AudioStream, LocalAudioTrack, LocalVideoTrack, RoomEvent,
  TrackKind, TrackSource, VideoSource, VideoStream } from '@livekit/rtc-node';

// Publish only the selected participant's tracks. A warm job and another room
// never share streams. Removing a camera also unpublishes its far-end track.
export function forwardTracks(from, to, identity, { video, onError }) {
  const tracks = new Map();
  let closed = false;
  async function pump(track, state) {
    const audio = track.kind === TrackKind.KIND_AUDIO;
    const stream = audio ? new AudioStream(track, { sampleRate: 48000, numChannels: 1, frameSizeMs: 20 }) : new VideoStream(track);
    const reader = stream.getReader();
    state.reader = reader;
    let source, local, publication;
    try {
      while (!state.stopped) {
        const { value, done } = await reader.read();
        if (done || state.stopped) break;
        if (!source) {
          source = audio ? new AudioSource(48000, 1, 80) : new VideoSource(value.frame.width, value.frame.height);
          local = audio ? LocalAudioTrack.createAudioTrack('call-audio', source) : LocalVideoTrack.createVideoTrack('call-video', source);
          publication = await to.localParticipant.publishTrack(local, {
            source: audio ? TrackSource.SOURCE_MICROPHONE : TrackSource.SOURCE_CAMERA,
            simulcast: false, stream: 'facetime',
          });
        }
        if (state.stopped) break;
        if (audio) await source.captureFrame(value);
        else source.captureFrame(value.frame, undefined, value.rotation);
      }
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
      if (publication?.sid) await to.localParticipant?.unpublishTrack(publication.sid, false).catch(() => {});
      await local?.close(false).catch(() => {});
      await source?.close().catch(() => {});
    }
  }
  function subscribe(track, publication, participant) {
    if (closed || participant.identity !== identity || tracks.has(publication.sid) || publication.muted) return;
    if (track.kind !== TrackKind.KIND_AUDIO && !(video && track.kind === TrackKind.KIND_VIDEO)) return;
    const state = { stopped: false };
    tracks.set(publication.sid, state);
    state.task = pump(track, state).catch(() => {
      if (!closed && !state.stopped) onError();
    }).finally(() => { if (tracks.get(publication.sid) === state) tracks.delete(publication.sid); });
  }
  function stop(publication, participant) {
    if (participant.identity !== identity) return;
    const state = tracks.get(publication.sid);
    if (!state) return;
    state.stopped = true;
    void state.reader?.cancel().catch(() => {});
    return state.task;
  }
  const muted = (pub, p) => { void stop(pub, p); };
  const unsubscribed = (_track, pub, p) => { void stop(pub, p); };
  const unmuted = async (pub, p) => {
    const previous = tracks.get(pub.sid);
    if (previous?.stopped) await previous.task;
    if (pub.track && !pub.muted) subscribe(pub.track, pub, p);
  };
  from.on(RoomEvent.TrackSubscribed, subscribe).on(RoomEvent.TrackUnsubscribed, unsubscribed)
    .on(RoomEvent.TrackMuted, muted).on(RoomEvent.TrackUnmuted, unmuted);
  const p = from.remoteParticipants.get(identity);
  for (const pub of p?.trackPublications.values() ?? []) if (pub.track) subscribe(pub.track, pub, p);
  return async () => {
    closed = true;
    from.off(RoomEvent.TrackSubscribed, subscribe).off(RoomEvent.TrackUnsubscribed, unsubscribed)
      .off(RoomEvent.TrackMuted, muted).off(RoomEvent.TrackUnmuted, unmuted);
    await Promise.allSettled([...tracks.values()].map(async state => {
      state.stopped = true;
      await state.reader?.cancel().catch(() => {});
      await state.task;
    }));
  };
}
