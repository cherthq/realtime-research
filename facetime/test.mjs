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

import { EventEmitter } from 'node:events';
import { greetOnBridgeReady } from './greet.mjs';

for (const initiallyReady of [false, true]) {
  test(`greet exactly once after readiness (already ready: ${initiallyReady})`, () => {
    const room = new EventEmitter();
    let greetings = 0;
    room.metadata = initiallyReady ? '{"chert_bridge_ready":true}' : '';
    const close = greetOnBridgeReady(room, () => greetings++);
    assert.equal(greetings, initiallyReady ? 1 : 0);
    for (const metadata of ['broken', 'null', '{"chert_bridge_ready":"true"}', '{"chert_bridge_ready":false}']) {
      room.metadata = metadata;
      room.emit('roomMetadataChanged');
    }
    assert.equal(greetings, initiallyReady ? 1 : 0);
    room.metadata = '{"chert_bridge_ready":true}';
    room.emit('roomMetadataChanged');
    room.emit('roomMetadataChanged');
    assert.equal(greetings, 1);
    assert.equal(room.listenerCount('roomMetadataChanged'), 0);
    close();
  });
}

test('ending a call cancels a pending greeting', () => {
  const room = new EventEmitter();
  const close = greetOnBridgeReady(room, () => assert.fail('call has ended'));
  close();
  room.metadata = '{"chert_bridge_ready":true}';
  room.emit('roomMetadataChanged');
  assert.equal(room.listenerCount('roomMetadataChanged'), 0);
});

test('greeting retries an uncertain response with the same session credentials', async () => {
  const calls = [];
  const session = await startSession('http://127.0.0.1:18837', async (url, opt) => {
    calls.push({url, opt});
    if (calls.length === 1) return Response.json(room);
    if (calls.length === 2) throw new TypeError('lost response');
    return Response.json({greeting_requested: true});
  });
  await session.greet();
  assert.equal(calls.length, 3);
  for (const {url, opt} of calls.slice(1)) {
    assert.equal(url, 'http://127.0.0.1:18837/api/sessions/session-01/greet');
    assert.equal(opt.method, 'POST');
    assert.equal(opt.headers['x-session-key'], 'example');
  }
});
