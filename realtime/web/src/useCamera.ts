import { useEffect, useRef, useState } from 'react';
import { LocalVideoTrack, Room, Track } from 'livekit-client';

export function useCamera(room: Room | null = null) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const request = useRef(0);
  const [enabled, setEnabled] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [shared, setShared] = useState(false);
  const [deviceId, setDeviceId] = useState('');
  const [streamVersion, setStreamVersion] = useState(0);
  const published = useRef<LocalVideoTrack | null>(null);

  function release() {
    request.current++;
    published.current?.stop();
    setShared(false);
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }

  function stop() {
    release();
    setEnabled(false);
    setPending(false);
  }

  async function toggle() {
    if (stream.current || pending) {
      stop();
      return;
    }
    await capture(deviceId);
  }

  async function selectDevice(value: string) {
    if (!stream.current) {
      setDeviceId(value);
      return;
    }
    await capture(value);
  }

  async function capture(selected: string) {
    const id = ++request.current;
    setPending(true);
    setError('');
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          ...(selected ? { deviceId: { exact: selected } } : {}),
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          aspectRatio: { ideal: 4 / 3 },
        },
      });
      // Permission can resolve after End call or unmount. Never leave that
      // late stream capturing. Publication is owned separately by the call.
      if (id !== request.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      const previous = stream.current;
      stream.current = media;
      media.getVideoTracks().forEach((track) => {
        track.onended = stop;
      });
      if (video.current) video.current.srcObject = media;
      previous?.getTracks().forEach((track) => {
        track.onended = null;
        track.stop();
      });
      setDeviceId(selected);
      setStreamVersion((value) => value + 1);
      setEnabled(true);
    } catch (reason) {
      if (id !== request.current) return;
      const name = (reason as DOMException).name;
      setError(
        name === 'NotAllowedError'
          ? 'Camera access is blocked. Allow it in your browser to try again.'
          : name === 'NotFoundError'
            ? 'No camera found. You can still join with audio.'
            : 'Camera unavailable. Check that another app is not using it.',
      );
    } finally {
      if (id === request.current) setPending(false);
    }
  }

  useEffect(() => {
    if (!room || !enabled || !stream.current) return;
    let disposed = false;
    const track = new LocalVideoTrack(stream.current.getVideoTracks()[0].clone());
    published.current = track;
    void room.localParticipant
      .publishTrack(track, {
        source: Track.Source.Camera,
        simulcast: false,
        videoEncoding: { maxBitrate: 500_000, maxFramerate: 10 },
      })
      .then(() => {
        if (disposed) void room.localParticipant.unpublishTrack(track).catch(() => {});
        else setShared(true);
      })
      .catch(() => {
        if (!disposed) setError('Camera sharing failed. Your camera is only a local preview.');
      });
    return () => {
      disposed = true;
      setShared(false);
      track.stop();
      if (published.current === track) published.current = null;
      void room.localParticipant.unpublishTrack(track).catch(() => {});
    };
  }, [room, enabled, streamVersion]);

  useEffect(() => {
    const leave = () => release();
    window.addEventListener('pagehide', leave);
    return () => {
      window.removeEventListener('pagehide', leave);
      release();
    };
  }, []);
  return { video, enabled, pending, error, shared, deviceId, selectDevice, toggle, stop };
}
