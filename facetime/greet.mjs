// Warm rooms are not calls. Speak only once the FaceTime bridge is ready.
export function greetOnBridgeReady(room, greet) {
  let done = false;
  const stop = () => { done = true; room.off('roomMetadataChanged', check); };
  function check() {
    if (done) return;
    let metadata;
    try { metadata = JSON.parse(room.metadata || '{}'); } catch { return; }
    if (metadata?.chert_bridge_ready !== true) return;
    stop();
    greet();
  }
  room.on('roomMetadataChanged', check);
  check();
  return stop;
}
