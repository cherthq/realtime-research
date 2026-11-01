import { useLayoutEffect, useRef, useState } from 'react';
import { SourceLinks, type WebSource } from './SourceLinks';

export type Utterance = {
  order?: number;
  sources?: WebSource[];
  id: string;
  speaker: 'user' | 'assistant';
  text: string;
  state: 'partial' | 'complete' | 'interrupted';
};

export function Conversation({
  utterances,
  active,
  reconnecting = false,
  checking = false,
}: {
  utterances: Utterance[];
  active: boolean;
  reconnecting?: boolean;
  checking?: boolean;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const [newMessages, setNewMessages] = useState(false);
  const latest = () => {
    const element = viewport.current;
    if (element) element.scrollTop = element.scrollHeight;
    following.current = true;
    setNewMessages(false);
  };
  useLayoutEffect(() => {
    if (following.current || !utterances.length) latest();
    else setNewMessages(true);
  }, [utterances]);
  const visible = utterances.filter((row) => row.text || row.speaker === 'user');
  return (
    <section className="conversation" aria-label="Conversation history">
      <header className="conversation-header">
        <h2>Conversation</h2>
        <span>
          {reconnecting
            ? 'Reconnecting captions…'
            : checking && active
              ? 'Checking the web…'
              : active
                ? 'Live captions'
                : visible.length
                  ? 'Call ended'
                  : 'Voice captions'}
        </span>
      </header>
      <div
        className="conversation-scroll"
        ref={viewport}
        role="log"
        aria-label="Call transcript"
        aria-live="off"
        tabIndex={0}
        onScroll={() => {
          const el = viewport.current!;
          following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
          if (following.current) setNewMessages(false);
        }}
      >
        {!visible.length && (
          <p className="conversation-empty">Your conversation will appear here.</p>
        )}
        {visible.map((row, index) => (
          <div className={`message-row ${row.speaker}`} key={row.id}>
            {visible[index - 1]?.speaker !== row.speaker && (
              <span className="message-author">{row.speaker === 'user' ? 'You' : 'Chert'}</span>
            )}
            <div className={`speech-bubble ${row.state === 'partial' ? 'partial' : ''}`}>
              {row.text || <span className="transcribing">Transcribing…</span>}
              {row.speaker === 'assistant' && row.text && <SourceLinks sources={row.sources} />}
            </div>
            {row.state === 'interrupted' && <span className="message-note">Interrupted</span>}
          </div>
        ))}
      </div>
      {newMessages && (
        <div className="conversation-navigation">
          <button className="latest-messages" onClick={latest}>
            Latest messages ↓
          </button>
        </div>
      )}
    </section>
  );
}
