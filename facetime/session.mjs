import { randomBytes } from 'node:crypto';

export const agentName = 'chert-avatar-live';
const request = { avatar_id: 'call07', model_id: 'avatarforever', mode: 'voice_diagnostic' };

// The start key reconciles a response lost after the GPU slot was reserved.
export async function startSession(origin = 'http://127.0.0.1:18837', fetcher = fetch) {
  const key = randomBytes(32).toString('hex');
  const options = {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-start-key': key },
    body: JSON.stringify(request),
  };
  const send = path => fetcher(origin + path, { ...options, signal: AbortSignal.timeout(20000) });
  let response;
  try {
    response = await send('/api/sessions');
  } catch {
    response = await send('/api/sessions/recover');
  }
  if (!response.ok) throw new Error('Avatar session unavailable (HTTP ' + response.status + ')');
  const session = await response.json();
  return {
    connection: session.connection,
    async close() {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const res = await fetcher(origin + '/api/sessions/' + encodeURIComponent(session.id), {
            method: 'DELETE', headers: { 'x-session-key': session.session_key },
            signal: AbortSignal.timeout(10000),
          });
          if (res.ok || res.status === 404 || res.status === 410) return;
        } catch { /* Retry only the same session. */ }
      }
      throw new Error('Avatar session cleanup could not be confirmed');
    },
  };
}
