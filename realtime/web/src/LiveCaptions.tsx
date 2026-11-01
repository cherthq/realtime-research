import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { Utterance } from './Conversation';
import { SourceLinks } from './SourceLinks';

const READ_TIME_MS = 5000;
const FADE_MS = 280;
type Caption = { row: Utterance; retiring: boolean };

function CaptionBubble({
  row,
  retiring,
  active,
  remove,
}: {
  row: Utterance;
  retiring: boolean;
  active: boolean;
  remove: (id: string) => void;
}) {
  const [phase, setPhase] = useState<'visible' | 'leaving' | 'hidden'>('visible');
  useEffect(() => {
    let hide: ReturnType<typeof setTimeout>;
    const leave = () => {
      setPhase('leaving');
      hide = setTimeout(() => {
        setPhase('hidden');
        if (retiring) remove(row.id);
      }, FADE_MS);
    };
    if (!active || retiring) leave();
    else setPhase('visible');
    // Only actual text/state changes restart reading time, never HTTP snapshots.
    const expiry = active && !retiring ? setTimeout(leave, READ_TIME_MS) : undefined;
    return () => {
      clearTimeout(expiry);
      clearTimeout(hide);
    };
  }, [row.id, row.text, row.state, retiring, active, remove]);
  if (phase === 'hidden') return null;
  return (
    <div className={`live-caption ${row.speaker} ${phase}`} aria-hidden={phase === 'leaving'}>
      <span className="sr-only">{row.speaker === 'user' ? 'You: ' : 'Chert: '}</span>
      <span className="caption-text">{row.text || 'Transcribing…'}</span>
      {row.speaker === 'assistant' && row.text && <SourceLinks sources={row.sources} />}
      {row.state === 'interrupted' && <span className="caption-interrupted">Interrupted</span>}
    </div>
  );
}

// Use the played-speech ledger; expiry only affects the overlay, not history.
export function LiveCaptions({
  utterances,
  reconnecting = false,
  active = true,
  enabled = true,
}: {
  utterances: Utterance[];
  reconnecting?: boolean;
  active?: boolean;
  enabled?: boolean;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const remove = useCallback(
    (id: string) => setCaptions((rows) => rows.filter((entry) => entry.row.id !== id)),
    [],
  );
  useEffect(() => {
    const recent = utterances.filter((row) => row.text || row.speaker === 'user').slice(-2);
    setCaptions((previous) => [
      // Retain outgoing DOM briefly so replacement also fades instead of popping.
      ...previous
        .filter((entry) => !recent.some((row) => row.id === entry.row.id))
        .slice(-2)
        .map((entry) => ({ ...entry, retiring: true })),
      ...recent.map((row) => ({ row, retiring: false })),
    ]);
  }, [utterances]);
  useLayoutEffect(() => {
    if (viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight;
  }, [captions, utterances]);
  return (
    <div
      className="live-captions"
      aria-label="Live captions"
      role="log"
      aria-live="off"
      ref={viewport}
    >
      {captions.map(({ row, retiring }) => (
        <CaptionBubble
          key={row.id}
          row={row}
          retiring={retiring}
          active={active && enabled}
          remove={remove}
        />
      ))}
      {reconnecting && enabled && active && (
        <span className="caption-reconnecting">Reconnecting captions…</span>
      )}
    </div>
  );
}
