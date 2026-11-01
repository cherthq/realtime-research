# Conversation, camera and captions

The demo uses OpenAI for transcription, dialogue and camera understanding, and
ElevenLabs for speech. Keys stay in the backend's private `realtime/.env`.
Replies default to concise English without an artificial output-token ceiling.

Camera sharing is opt-in through the call controls. During a call, the agent can
use a recent camera frame when answering a spoken turn; this is not continuous
screen capture. Turning the camera off stops sharing. The interface shows the
camera-sharing state. Never describe this as an always-on screen reader.

Live captions follow the spoken reply, with partial user transcriptions merged
into a turn. The conversation button opens history; caption bubbles fade away.
Interruption cancels pending speech and generated speaking frames, with a brief
hold of the last generated frame permitted while listening motion is prepared.

See [setup](recreate-demo.md), [listening control](listening-profile.md) and
[web search](web-search.md). Test microphone/speaker behavior on the actual
browser and device; synthetic-input checks do not establish acoustic lip-sync.
