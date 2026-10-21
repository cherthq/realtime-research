"""FFmpeg operations; no neural dependencies or network access."""

from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave


def run(command, *, binary=False):
    result = subprocess.run([str(arg) for arg in command], capture_output=True,
                            text=not binary, check=False)
    if result.returncode:
        message = result.stderr.decode(errors="replace") if binary else result.stderr
        raise RuntimeError(f"{command[0]} failed: {message[-4000:]}")
    return result.stdout


def require_tools():
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError("Install binaries on PATH: " + ", ".join(missing))


def probe(path):
    metadata = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                               "-of", "json", path]))
    videos = [s for s in metadata["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
    if not videos:
        raise ValueError(f"No video stream: {path}")
    video = videos[0]
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    if rotation % 360:
        raise ValueError("Rotated video is unsupported; normalize orientation before importing")
    if video.get("sample_aspect_ratio", "1:1") not in {"1:1", "N/A", "0:1"}:
        raise ValueError("Non-square pixels unsupported; normalize before importing")
    fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    if fps <= 0:
        fps = float(Fraction(video.get("r_frame_rate", "0/1")))
    origin = float(video.get("start_time", metadata.get("format", {}).get("start_time", 0)))
    duration = float(video.get("duration", 0))
    if duration <= 0:
        # Matroska DURATION tags are end timestamps, not lengths when the first
        # frame starts away from zero. Prefer them over container-level duration.
        tagged_end = video.get("tags", {}).get("DURATION", "")
        if tagged_end:
            hours, minutes, seconds = map(float, tagged_end.split(":"))
            duration = hours * 3600 + minutes * 60 + seconds - origin
    if duration <= 0:
        fmt = metadata.get("format", {})
        if abs(origin) > 0.001:
            raise ValueError("Non-zero video origin without stream duration; normalize/remux before importing")
        duration = float(fmt.get("duration", 0))
    if duration <= 0 or fps <= 0 or not all(math.isfinite(x) for x in (origin, duration, fps)):
        raise ValueError("Cannot establish video duration/frame rate")
    return {"width": int(video["width"]), "height": int(video["height"]),
            "fps": fps, "duration": duration, "video_start": origin,
            "video_index": video["index"], "audio_index": audios[0]["index"] if audios else None,
            "audio_start": float(audios[0].get("start_time", origin)) if audios else None,
            "audio_duration": float(audios[0].get("duration", 0)) if audios else None,
            "raw": metadata}


def frame_png(path, seconds, info):
    # Input seeking uses absolute timestamps, not the container's possibly
    # different origin. Output trimming handles the common media clock separately.
    return run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-seek_timestamp", "1", "-ss", f"{info['video_start'] + seconds:.9f}",
                "-i", path, "-map", f"0:{info['video_index']}", "-frames:v", "1",
                "-f", "image2pipe", "-c:v", "png", "pipe:1"], binary=True)


def render_clip(path, directory, start, end, crop, config, info):
    """Subtract one common timestamp from both streams, including audio delay."""
    if info["audio_index"] is None:
        raise ValueError("missing_audio")
    if not 0 <= start < end <= info["duration"] + 0.05:
        raise ValueError("Invalid render interval")
    x, y, width, height = crop
    if (any(not isinstance(v, int) or v % 2 for v in crop) or x < 0 or y < 0
            or width < config.width or height < config.height
            or x + width > info["width"] or y + height > info["height"]
            or width * config.height != height * config.width):
        raise ValueError("Crop must fit source, preserve aspect ratio, and avoid upscaling")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    absolute_start, absolute_end = info["video_start"] + start, info["video_start"] + end
    duration = end - start
    vf = (f"[0:{info['video_index']}]trim=start={absolute_start:.9f}:end={absolute_end:.9f},"
          f"setpts=PTS-({absolute_start:.9f})/TB,crop={width}:{height}:{x}:{y},"
          f"scale={config.width}:{config.height}:flags=lanczos,setsar=1[v]")
    af = (f"[0:{info['audio_index']}]atrim=start={absolute_start:.9f}:end={absolute_end:.9f},"
          f"asetpts=PTS-({absolute_start:.9f})/TB,aresample=48000:async=1:first_pts=0,"
          f"apad,atrim=duration={duration:.9f}[a]")
    video = directory / "video.mp4"
    temporary = directory / "video.partial.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-copyts",
         "-i", path, "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-fps_mode", "passthrough", "-c:a", "aac", "-ar", "48000", "-ac", "1",
         "-t", f"{duration:.9f}", "-movflags", "+faststart", temporary])
    temporary.replace(video)
    audio = directory / "audio.wav"
    temporary_audio = directory / "audio.partial.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", video,
         "-vn", "-t", f"{duration:.9f}", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1",
         temporary_audio])
    temporary_audio.replace(audio)
    reference = directory / "reference.png"
    # Use a decoded output frame, so reference and output crop are identical.
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", video,
         "-frames:v", "1", reference])
    validate_clip(video, audio, config, duration)
    return {"video": video, "audio": audio, "reference_image": reference}


def validate_clip(video, audio, config, duration):
    info = probe(video)
    tolerance = 1 / info["fps"] + 0.002
    if (info["width"], info["height"]) != (config.width, config.height):
        raise ValueError("Wrong output dimensions")
    if info["audio_index"] is None or abs(info["duration"] - duration) > tolerance:
        raise ValueError("Output missing audio or video duration differs from trim interval")
    if abs(info["video_start"]) > tolerance or abs(info["audio_start"]) > tolerance:
        raise ValueError("Output timestamps are not near the common zero origin")
    if info["audio_duration"] and abs(info["audio_duration"] - duration) > tolerance:
        raise ValueError("Muxed audio duration differs from trim interval")
    with wave.open(str(audio), "rb") as handle:
        wav_duration = handle.getnframes() / handle.getframerate()
        if handle.getnchannels() != 1 or handle.getframerate() != 48000:
            raise ValueError("Expected mono 48 kHz PCM WAV")
    if abs(wav_duration - duration) > tolerance:
        raise ValueError("Extracted audio duration differs from trim interval")
    return info
