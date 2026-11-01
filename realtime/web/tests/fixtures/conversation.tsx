// Test harness only. Not included in the production build or served API.
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LiveCaptions } from '../../src/LiveCaptions';
import { Conversation, type Utterance } from '../../src/Conversation';
import { useTranscripts } from '../../src/useTranscripts';
import { useCamera } from '../../src/useCamera';
import { Room, RoomEvent } from 'livekit-client';
import '../../src/style.css';

const fixtureRoom = {
  localParticipant: {
    async publishTrack(track: unknown) {
      (window as any).published = track;
    },
    async unpublishTrack() {
      (window as any).unpublished = true;
    },
  },
} as unknown as Room;
const captionRoom = new Room();
function Fixture() {
  const [session, setSession] = useState<{ id: string; session_key: string } | null>(null);
  const [rows, setRows] = useState<Utterance[]>([]);
  const [sharing, setSharing] = useState(false);
  const [overlayActive, setOverlayActive] = useState(true);
  const camera = useCamera(sharing ? fixtureRoom : null);
  const captions = useTranscripts(session, true, captionRoom);
  (window as any).fixture = {
    rows: setRows,
    overlayActive: setOverlayActive,
    session: setSession,
    share: setSharing,
    cameraDevice: camera.selectDevice,
    caption: (packet: unknown, identity = 'avatar') =>
      captionRoom.emit(
        RoomEvent.DataReceived,
        new TextEncoder().encode(JSON.stringify(packet)),
        { identity } as any,
        undefined,
        'chert-captions',
      ),
  };
  return (
    <main className="call-page">
      <header className="call-header">
        <span className="wordmark">chert</span>
        <span className="call-status">Listening</span>
      </header>
      <section className="video-pair" aria-label="Video call">
        <div className="video-tile local-tile small-tile">
          <video
            ref={camera.video}
            className={camera.enabled ? 'local-video visible' : ''}
            autoPlay
            muted
            playsInline
          />
          {!camera.enabled && (
            <div className="camera-placeholder">
              <p>Your camera is off</p>
            </div>
          )}
          <div className="tile-caption">
            <span>You</span>
            <span>{camera.shared ? 'Shared with AI' : 'Local preview'}</span>
          </div>
        </div>
        <div className="video-tile avatar-tile large-tile">
          <LiveCaptions
            utterances={rows.length ? rows : captions.utterances}
            active={overlayActive}
          />
          <div className="camera-placeholder">
            <p>Avatar video</p>
          </div>
          <div className="tile-caption">
            <span>Avatar</span>
            <span>AI generated</span>
          </div>
        </div>
      </section>
      <Conversation
        utterances={rows.length ? rows : captions.utterances}
        active={!!session}
        reconnecting={captions.reconnecting}
        checking={captions.checking}
      />
      <footer className="call-footer">
        <div className="call-controls" role="group" aria-label="Call controls">
          <button
            className="icon-control"
            onClick={() => void camera.toggle()}
            aria-label="Toggle camera"
          >
            ◉
          </button>
          <button
            className="call-button"
            onClick={() => {
              setSession(null);
              setSharing(false);
            }}
          >
            End call
          </button>
        </div>
      </footer>
    </main>
  );
}
createRoot(document.getElementById('root')!).render(<Fixture />);
