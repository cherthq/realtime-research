import { useEffect, useRef, useState } from 'react';

export function useCamera() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const request = useRef(0);
  const [enabled, setEnabled] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');

  function release() {
    request.current++;
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
    const id = ++request.current;
    setPending(true);
    setError('');
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          aspectRatio: { ideal: 4 / 3 },
        },
      });
      // Permission can resolve after End call or unmount. Never leave that
      // late stream capturing, and never publish this preview to the room.
      if (id !== request.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      media.getVideoTracks().forEach((track) => {
        track.onended = stop;
      });
      if (video.current) video.current.srcObject = media;
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
    const leave = () => release();
    window.addEventListener('pagehide', leave);
    return () => {
      window.removeEventListener('pagehide', leave);
      release();
    };
  }, []);
  return { video, enabled, pending, error, toggle, stop };
}
