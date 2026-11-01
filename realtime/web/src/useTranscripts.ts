import { useEffect, useRef, useState } from 'react';
import type { Utterance } from './Conversation';
import { RoomEvent, type Room, type RemoteParticipant } from 'livekit-client';

type Source = { id: string; session_key: string };
type Snapshot = {
  revision: number;
  ended: boolean;
  utterances: Utterance[];
  lookup?: { state: string };
};

export function useTranscripts(session: Source | null, polling: boolean, room: Room | null = null) {
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [reconnecting, setReconnecting] = useState(false);
  const [checking, setChecking] = useState(false);
  const previousSession = useRef<string | null>(null);
  useEffect(() => {
    if (!session) {
      setReconnecting(false);
      setChecking(false);
      return;
    }
    if (previousSession.current !== session.id) {
      setUtterances([]);
      setChecking(false);
    }
    previousSession.current = session.id;
    const controller = new AbortController();
    let revision = -1;
    let ended = false;
    const apply = (snapshot: Snapshot, partial = false) => {
      if (controller.signal.aborted) return;
      // Teardown clears server memory. Keep the last visible conversation here.
      ended = snapshot.ended;
      if (ended) setChecking(false);
      if (
        (snapshot.revision > revision || (!partial && snapshot.revision === revision)) &&
        !ended
      ) {
        revision = snapshot.revision;
        setChecking(snapshot.lookup?.state === 'checking');
        setUtterances((previous) => {
          if (!partial) return snapshot.utterances.slice(-200);
          const merged = new Map(previous.map((row) => [row.id, row]));
          for (const row of snapshot.utterances) merged.set(row.id, row);
          return [...merged.values()].sort((a, b) => (a.order ?? 0) - (b.order ?? 0)).slice(-200);
        });
      }
      setReconnecting(false);
    };
    const receive = (
      payload: Uint8Array,
      participant?: RemoteParticipant,
      _kind?: unknown,
      topic?: string,
    ) => {
      if (
        participant?.identity !== 'avatar' ||
        topic !== 'chert-captions' ||
        payload.length > 14000
      )
        return;
      try {
        const value = JSON.parse(new TextDecoder().decode(payload));
        if (
          !Number.isSafeInteger(value.revision) ||
          typeof value.ended !== 'boolean' ||
          !Array.isArray(value.utterances) ||
          value.utterances.length > 4
        )
          return;
        if (
          !value.utterances.every(
            (row: Utterance) =>
              typeof row.id === 'string' &&
              typeof row.text === 'string' &&
              ['user', 'assistant'].includes(row.speaker) &&
              ['partial', 'complete', 'interrupted'].includes(row.state) &&
              Number.isSafeInteger(row.order),
          )
        )
          return;
        apply(value, true);
      } catch {
        /* Ignore malformed room data. */
      }
    };
    room?.on(RoomEvent.DataReceived, receive);
    const pause = () =>
      new Promise<void>((resolve) => {
        const done = () => {
          clearTimeout(timer);
          controller.signal.removeEventListener('abort', done);
          resolve();
        };
        const timer = setTimeout(done, 500);
        controller.signal.addEventListener('abort', done, { once: true });
      });
    const headers = { 'X-Session-Key': session.session_key };
    const path = `/api/sessions/${session.id}/transcripts`;
    async function run() {
      if (!polling) {
        const stream = new AbortController();
        let timeout = setTimeout(() => stream.abort(), 5000);
        try {
          const response = await fetch(path + '/events', {
            headers,
            signal: AbortSignal.any([controller.signal, stream.signal]),
          });
          if (!response.ok || !response.body) throw new Error('Captions unavailable');
          const reader = response.body.getReader(),
            decoder = new TextDecoder();
          let buffer = '';
          while (!controller.signal.aborted && !ended) {
            clearTimeout(timeout);
            timeout = setTimeout(() => stream.abort(), 5000);
            const item = await reader.read();
            if (item.done) break;
            buffer += decoder.decode(item.value, { stream: true });
            if (buffer.length > 2 * 1024 * 1024) throw new Error('Caption size limit');
            let boundary: number;
            while ((boundary = buffer.indexOf('\n\n')) >= 0) {
              const event = buffer.slice(0, boundary);
              buffer = buffer.slice(boundary + 2);
              if (event.startsWith('data: ')) apply(JSON.parse(event.slice(6)));
            }
          }
        } catch {
          /* Recover with snapshots; a caption outage must not stop media. */
        } finally {
          clearTimeout(timeout);
          stream.abort();
        }
      }
      while (!controller.signal.aborted && !ended) {
        try {
          const response = await fetch(path, {
            headers,
            cache: 'no-store',
            signal: AbortSignal.any([controller.signal, AbortSignal.timeout(3000)]),
          });
          if ([401, 403, 404, 410].includes(response.status)) {
            setReconnecting(true);
            return;
          }
          if (!response.ok) throw new Error('Captions unavailable');
          apply(await response.json());
        } catch {
          if (!controller.signal.aborted) setReconnecting(true);
        }
        if (!controller.signal.aborted) await pause();
      }
    }
    void run();
    return () => {
      controller.abort();
      room?.off(RoomEvent.DataReceived, receive);
    };
  }, [session?.id, session?.session_key, polling, room]);
  return { utterances, reconnecting, checking };
}
