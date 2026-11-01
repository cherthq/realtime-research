import { useEffect, useRef, useState } from 'react';

type Place = {
  state: 'waiting' | 'ready' | 'starting';
  position: number;
  waiting: number;
  turn_expires_in: number | null;
  backend_ready: boolean;
};
const storageKey = 'chert-queue-ticket';

function storedTicket() {
  try {
    return sessionStorage.getItem(storageKey) || '';
  } catch {
    return '';
  }
}
function storeTicket(key: string) {
  try {
    if (key) sessionStorage.setItem(storageKey, key);
    else sessionStorage.removeItem(storageKey);
  } catch {
    // Storage can be unavailable in an embedded/private browser; in-memory works.
  }
}

async function request(key: string, method: string, signal?: AbortSignal): Promise<Place> {
  const response = await fetch('/api/queue', {
    method,
    headers: { 'X-Queue-Key': key },
    signal,
  });
  const result = await response.json();
  if (!response.ok)
    throw Object.assign(new Error(result.detail || 'Could not reach the waiting queue.'), {
      status: response.status,
    });
  return result;
}

export function useCallQueue(enabled: boolean) {
  const [key, setKey] = useState(() => (enabled ? storedTicket() : ''));
  const keyRef = useRef(key);
  const [place, setPlace] = useState<Place | null>(null);
  const [paused, setPaused] = useState(false);
  const [notice, setNotice] = useState('');
  const [remaining, setRemaining] = useState(0);
  const deadline = useRef(0);
  const generation = useRef(0);

  function update(value: Place) {
    deadline.current =
      value.turn_expires_in == null ? 0 : Date.now() + value.turn_expires_in * 1000;
    setRemaining(value.turn_expires_in || 0);
    setPlace(value);
    setNotice('');
  }
  function clear() {
    generation.current++;
    keyRef.current = '';
    storeTicket('');
    setKey('');
    setPlace(null);
    setPaused(false);
    deadline.current = 0;
  }
  async function leave() {
    const previous = keyRef.current;
    clear();
    setNotice('');
    if (previous)
      await request(previous, 'DELETE', AbortSignal.timeout(5000)).catch(() => {
        // Lost clients expire server-side after 90 seconds without a heartbeat.
      });
  }
  async function join() {
    const version = generation.current;
    const ticket =
      keyRef.current ||
      Array.from(crypto.getRandomValues(new Uint8Array(32)), (byte) =>
        byte.toString(16).padStart(2, '0'),
      ).join('');
    keyRef.current = ticket;
    storeTicket(ticket);
    const value = await request(ticket, 'POST', AbortSignal.timeout(10000));
    if (version !== generation.current) {
      void request(ticket, 'DELETE', AbortSignal.timeout(5000)).catch(() => {});
      throw new Error('Queue request cancelled.');
    }
    setKey(ticket);
    update(value);
    return { key: ticket, place: value };
  }

  useEffect(() => {
    if (!enabled || !key || paused) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await request(
          key,
          'GET',
          AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]),
        );
        if (!controller.signal.aborted && keyRef.current === key) update(value);
      } catch (error) {
        if (controller.signal.aborted || keyRef.current !== key) return;
        if ((error as { status?: number }).status === 410) {
          clear();
          setNotice('Your place expired. Join again when you’re ready.');
          return;
        }
        deadline.current = 0;
        setPlace((previous) =>
          previous ? { ...previous, state: 'waiting', turn_expires_in: null } : null,
        );
        setNotice('Reconnecting to the queue…');
      }
      if (!controller.signal.aborted) timer = setTimeout(() => void poll(), 3000);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [enabled, key, paused]);

  useEffect(() => {
    if (!key || paused) return;
    const timer = setInterval(
      () => setRemaining(Math.max(0, Math.ceil((deadline.current - Date.now()) / 1000))),
      1000,
    );
    return () => clearInterval(timer);
  }, [key, paused]);

  return {
    enabled,
    place,
    remaining,
    notice,
    queued: !!key,
    key: () => keyRef.current,
    join,
    leave,
    clear,
    pause: () => setPaused(true),
  };
}
