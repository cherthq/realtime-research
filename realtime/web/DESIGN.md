# Chert call interface

## Direction

The November 1 update follows the user's three atmee.ai screenshots: a dark
call surface, a dominant avatar, a small self-camera, overlaid captions and round
controls beneath the video. This replaces the earlier light, equal-tile layout.
Chert retains its own name, logo and existing portraits.

## Layout and motion

- Desktop: a 4:3 avatar with the camera inset inside its top-right corner.
  Chert's logo sits at the avatar's bottom-left, replacing the media-state label.
  Controls center below the frame, outside the media surface.
- Phone: an immersive portrait viewport with a camera inset at the top right.
  This is a presentation crop only. Generation and delivered video remain
  at the configured native resolution; video, reference and interruption-hold
  canvas share the same fit.
- Click or keyboard-activate the small video to swap sizes. The media elements
  and tracks stay mounted; the swap changes only layout classes.
- The latest two utterances overlay the large view, without visible speaker
  headings. They fade out after five seconds without new text, or when the call
  ends. An unchanged recovery snapshot does not revive expired text.
  The conversation button opens
  full scrollable history. Both read the same confirmed playback ledger.
- Settings holds caption/timer toggles and camera/microphone/speaker selectors.
  Avatar and voice configuration and connection details are expandable.
- Graphite surfaces, warm white type and a restrained warm accent. Gradients
  behind video labels provide contrast; they are not decorative backgrounds.
- UI transitions are 160–240ms: control hover/press, menu entry, toggle switches
  and new caption bubbles. Partial text updates retain the existing bubble.
- Never animate media visibility, crossfade generated video or remount tracks
  for a layout/menu change. Interruption holds remain synchronous.
- Respect reduced motion. Keep native dialogs with Escape/focus restoration,
  visible focus and at least 44px call targets, including 320px screens.
- No unimplemented share/bookmark actions, external fonts or animation packages.
  Device enumeration does not request permission. Camera capture remains opt-in.

## Implementation references

- User-supplied screenshots are the visual source, not assets copied into Chert.
- [Anti Slop](https://github.com/miqdadbadjuber/anti-slop), previously selected by
  the user, at commit `3005235d030b4793146ad9afe8d01045efd90e1d`.
- [LiveKit Room API](https://docs.livekit.io/reference/client-sdk-js/classes/Room.html)
  and installed SDK source for enumeration and microphone switching. Speaker
  selection targets the actual Web Audio destination and is disabled when the
  browser does not support it. Camera switching replaces the local preview and
  republishes its separate clone without resetting the avatar.

Browser regressions live in `tests/call-ui.spec.ts`; see [known limits](../QUALIFICATION.md).
