import { fileURLToPath } from 'node:url';
import { AutoSubscribe, cli, defineAgent, ServerOptions } from '@livekit/agents';
import { Room, RoomEvent } from '@livekit/rtc-node';
import { agentName, startSession } from './session.mjs';
import { forwardTracks } from './media.mjs';
import { greetOnBridgeReady } from './greet.mjs';

export default defineAgent({
  entry: async ctx => {
    const room = ctx.room;
    const inner = new Room();
    let ended = false, active, setup, stopGreeting;
    const forwards = [];
    const stop = () => {
      if (ended) return;
      ended = true;
      stopGreeting?.();
      ctx.shutdown('FaceTime call ended');
    };
    const isBridge = p => p.identity.startsWith('chert-bridge-');
    async function join(p) {
      if (ended || !isBridge(p) || setup) return;
      setup = (async () => {
        // A standby dispatch holds neither a GPU slot nor a billable voice session.
        active = await startSession();
        if (ended) return;
        await inner.connect(active.connection.url, active.connection.token, {autoSubscribe: true, dynacast: false});
        if (ended) return;
        const meta = JSON.parse(ctx.job.metadata || '{}');
        const video = meta.media?.video !== false;
        forwards.push(forwardTracks(inner, room, 'avatar', {video, onError: stop}));
        forwards.push(forwardTracks(room, inner, p.identity, {video, onError: stop}));
        stopGreeting = greetOnBridgeReady(room, () => {
          if (ended) return;
          active.greet().then(() => console.info('FaceTime greeting requested'))
            .catch(() => { console.error('Could not request the FaceTime greeting'); stop(); });
        });
      })().catch(() => { console.error('Could not prepare the FaceTime avatar session'); stop(); });
    }
    room.on(RoomEvent.ParticipantConnected, join);
    room.on(RoomEvent.ParticipantDisconnected, p => { if (isBridge(p)) stop(); });
    inner.on(RoomEvent.Disconnected, stop);
    ctx.addShutdownCallback(async () => {
      ended = true;
      stopGreeting?.();
      await setup;
      await Promise.allSettled(forwards.map(close => close()));
      await inner.disconnect().catch(() => {});
      if (active) await active.close();
    });
    await ctx.connect(undefined, AutoSubscribe.SUBSCRIBE_ALL);
    for (const p of room.remoteParticipants.values()) await join(p);
  },
});

cli.runApp(new ServerOptions({
  agent: fileURLToPath(import.meta.url), agentName,
  numIdleProcesses: 1, host: '127.0.0.1', port: 18839,
}));
