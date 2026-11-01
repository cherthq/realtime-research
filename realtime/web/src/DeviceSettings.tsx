import { useEffect, useState } from 'react';
import { Room } from 'livekit-client';

export type DeviceChoice = Record<MediaDeviceKind, string>;

// Enumeration never requests permission. The call/camera controls own capture.
export function DeviceSettings({
  choices,
  refreshKey,
  disabled,
  onSelect,
}: {
  choices: DeviceChoice;
  refreshKey: string;
  disabled: boolean;
  onSelect: (kind: MediaDeviceKind, id: string) => Promise<void>;
}) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [changing, setChanging] = useState(false);
  const [error, setError] = useState('');
  const canChooseSpeaker =
    typeof AudioContext !== 'undefined' && 'setSinkId' in AudioContext.prototype;
  useEffect(() => {
    let disposed = false;
    const refresh = () =>
      void Room.getLocalDevices(undefined, false)
        .then((items) => {
          if (!disposed) setDevices(items);
        })
        .catch(() => {
          if (!disposed)
            setError('Device names appear after you allow camera or microphone access.');
        });
    refresh();
    navigator.mediaDevices?.addEventListener('devicechange', refresh);
    return () => {
      disposed = true;
      navigator.mediaDevices?.removeEventListener('devicechange', refresh);
    };
  }, [refreshKey]);

  async function choose(kind: MediaDeviceKind, id: string) {
    setChanging(true);
    setError('');
    try {
      await onSelect(kind, id);
    } catch {
      setError('Could not switch devices. Check browser permissions and try again.');
    } finally {
      setChanging(false);
    }
  }
  return (
    <section className="device-settings" aria-label="Devices">
      <h3>Devices</h3>
      {(['videoinput', 'audioinput', 'audiooutput'] as const).map((kind) => {
        const label = { videoinput: 'Camera', audioinput: 'Microphone', audiooutput: 'Speaker' }[
          kind
        ];
        const items = devices.filter(
          (device) => device.kind === kind && device.deviceId && device.deviceId !== 'default',
        );
        return (
          <div key={kind}>
            <label htmlFor={`device-${kind}`}>{label}</label>
            <select
              id={`device-${kind}`}
              value={choices[kind]}
              onChange={(event) => void choose(kind, event.target.value)}
              disabled={disabled || changing || (kind === 'audiooutput' && !canChooseSpeaker)}
            >
              <option value="">System default</option>
              {choices[kind] && !items.some((item) => item.deviceId === choices[kind]) && (
                <option value={choices[kind]}>Selected device (unavailable)</option>
              )}
              {items.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `${label} ${index + 1}`}
                </option>
              ))}
            </select>
          </div>
        );
      })}
      {!canChooseSpeaker && (
        <p className="field-note">Choose your speaker in your device’s sound settings.</p>
      )}
      {error && (
        <p className="device-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
