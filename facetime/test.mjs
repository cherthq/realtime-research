import test from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session.mjs';

const room = { id: 'session-01', session_key: 'example', connection: {url: 'wss://example.invalid', token:'fixture'} };

test('recover an uncertain start without allocating a second avatar session', async () => {
  const calls = [];
  const session = await startSession('http://127.0.0.1:18837', async (url, opt) => {
    calls.push({url, opt});
    if (calls.length === 1) throw new TypeError('lost response');
    if (calls.length === 2) return Response.json(room);
    return new Response(null, {status: 204});
  });
  assert.deepEqual(session.connection, room.connection);
  assert.match(calls[1].url, /\/recover$/);
  assert.match(calls[0].opt.headers['x-start-key'], /^[a-f0-9]{64}$/);
  assert.equal(calls[1].opt.headers['x-start-key'], calls[0].opt.headers['x-start-key']);
  assert.equal(calls[0].opt.body, calls[1].opt.body);
  await session.close();
  assert.equal(calls[2].opt.headers['x-session-key'], 'example');
  assert.equal(calls[2].opt.method, 'DELETE');
});

test('busy GPU declines without reclaiming the current session', async () => {
  let count = 0;
  await assert.rejects(startSession('http://127.0.0.1:18837', async () => {
    count++;
    return new Response(null, {status:409});
  }), /409/);
  assert.equal(count, 1);
});
