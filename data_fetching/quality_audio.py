"""Bounded-memory waveform statistics and overlapping speech-window helpers."""

import math
import wave

from .metadata import check, seconds


def speech_segments(windows, duration, config):
    """Hysteresis, minimum silence, short-event filtering and bounded padding."""
    from .metadata import ranges_union

    result, active, quiet = [], None, None
    for row in windows:
        probability, at = row["probability"], row["start"]
        if active is None:
            if probability >= config["vad_threshold"]:
                active = at
            continue
        if probability >= config["vad_release_threshold"]:
            quiet = None
        else:
            quiet = at if quiet is None else quiet
            if row["end"] - quiet >= config["vad_min_silence"]:
                if quiet - active >= config["vad_min_speech"]:
                    result.append(
                        [
                            max(0, active - config["vad_padding"]),
                            min(duration, quiet + config["vad_padding"]),
                        ]
                    )
                active, quiet = None, None
    if active is not None:
        stop = quiet if quiet is not None else duration
        if stop - active >= config["vad_min_speech"]:
            result.append(
                [
                    max(0, active - config["vad_padding"]),
                    min(duration, stop + config["vad_padding"]),
                ]
            )
    return ranges_union(result)


def score_findings(scores, start, speech, threshold, min_bad_seconds, fps=25):
    """Retain short bad speaking runs that a whole-window median would conceal."""
    from .metadata import ranges_union

    bad = []
    for index, score in enumerate(scores):
        a, b = start + index / fps, start + (index + 1) / fps
        if score < threshold:
            bad.extend(
                [[max(a, s), min(b, e)] for s, e in speech if max(a, s) < min(b, e)]
            )
    return [
        span
        for span in ranges_union(bad)
        if span[1] - span[0] + 1e-7 >= min_bad_seconds
    ]


def waveform_check(path):
    import numpy as np

    total, square, peak, clipped, silent, windows = 0, 0.0, 0.0, 0, 0, []
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("Waveform QC requires signed 16-bit PCM")
        rate, channels = source.getframerate(), source.getnchannels()
        while data := source.readframes(rate):
            values = (
                np.frombuffer(data, dtype="<i2").reshape(-1, channels).astype(float)
                / 32768
            )
            rms = float(np.sqrt(np.mean(values**2)))
            block_peak = float(np.max(np.abs(values)))
            windows.append(
                {
                    "start": total / channels / rate,
                    "end": (total + values.size) / channels / rate,
                    "rms": rms,
                    "peak": block_peak,
                    "channel_rms": np.sqrt(np.mean(values**2, axis=0)).tolist(),
                }
            )
            square += float(np.sum(values**2))
            peak, total = max(peak, block_peak), total + values.size
            clipped += int(np.sum(np.abs(values) >= 0.999))
            silent += int(np.sum(np.abs(values) < 0.0001))
    if not total:
        raise ValueError("Empty audio")
    duration = total / channels / rate
    flags = []
    if clipped / total > 0.001:
        flags.append(
            {
                "code": "possible_clipping",
                "start": 0,
                "end": duration,
                "origin": "heuristic",
            }
        )
    if silent / total > 0.95:
        flags.append(
            {
                "code": "mostly_silent",
                "start": 0,
                "end": duration,
                "origin": "heuristic",
            }
        )
    return check(
        "flag" if flags else "pass",
        "pcm_waveform",
        [(0, duration)],
        flags,
        peak=peak,
        rms=math.sqrt(square / total),
        near_full_scale_fraction=clipped / total,
        silence_fraction=silent / total,
        channels=channels,
        sample_rate=rate,
        windows=windows,
    )


def speech_windows(duration, speech, window=3.0, stride=1.0):
    """Cover speech using full-length windows, including an end-aligned final one."""
    if duration < window:
        return []
    starts, at = [], 0.0
    while at + window < duration - 1e-7:
        starts.append(at)
        at += stride
    starts.append(duration - window)
    return [
        (a, a + window)
        for a in sorted(set(starts))
        if any(s < a + window and e > a for s, e in speech)
    ]


def speech_coverage(speech, inspected):
    numerator = seconds(
        [
            (max(a, s), min(b, e))
            for a, b in inspected
            for s, e in speech
            if max(a, s) < min(b, e)
        ]
    )
    denominator = seconds(speech)
    return min(1, numerator / denominator) if denominator else 0
