"""Lazy optional scene/detection adapters. Never download models implicitly."""

from importlib.metadata import version
import math

from .core import file_hash, read_json, within
from .media import frame_png


def runtime_versions():
    return {name: version(name) for name in
            ("mediapipe", "numpy", "opencv-contrib-python", "scenedetect")}


def scene_intervals(path, start, end, config):
    from scenedetect import ContentDetector, FrameTimecode, SceneManager, open_video

    video = open_video(str(path))
    try:
        # All cuts, including very short scenes, matter: short scenes are rejected
        # later, not merged across cuts by a long minimum-scene-length setting.
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=config.scene_threshold, min_scene_len=1))
        # In PySceneDetect integer numeric inputs mean FRAME COUNTS, not seconds.
        # Explicit float-based timecodes avoid treating a user interval of 60s
        # as 60 frames, including when JSON manifests contain integer timestamps.
        video.seek(FrameTimecode(float(start), fps=video.frame_rate))
        manager.detect_scenes(video=video, end_time=FrameTimecode(float(end), fps=video.frame_rate))
        scenes = manager.get_scene_list(start_in_scene=True)
        if not scenes:
            return [(start, end)]
        return [(max(start, a.get_seconds()), min(end, b.get_seconds())) for a, b in scenes]
    finally:
        # PySceneDetect's OpenCV backend has no public close() in 0.6.x.
        if hasattr(video, "_cap"):
            video._cap.release()


def model_paths(root):
    manifest = read_json(root / "models" / "models.json")
    paths, hashes = {}, {}
    for name in ("face", "pose"):
        record = manifest[name]
        path = within(root, record["path"])
        digest = file_hash(path)
        if digest != record["sha256"]:
            raise ValueError(f"{name} detector checksum mismatch; left untouched")
        paths[name], hashes[name] = path, digest
    return paths, hashes


class Detector:
    def __init__(self, paths, config):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        # IMAGE mode makes each sample independent of process order and resume.
        self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(paths["face"])),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=config.face_confidence))
        try:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(paths["pose"])),
                running_mode=vision.RunningMode.IMAGE, num_poses=2,
                min_pose_detection_confidence=config.shoulder_confidence,
                min_pose_presence_confidence=config.shoulder_confidence))
        except Exception:
            self.face.close()
            raise

    def close(self):
        self.face.close()
        self.pose.close()

    def detect(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        face_result, pose_result = self.face.detect(image), self.pose.detect(image)
        height, width = rgb.shape[:2]
        faces = []
        for face in face_result.detections:
            box = face.bounding_box
            faces.append({"box": [box.origin_x, box.origin_y,
                                  box.origin_x + box.width, box.origin_y + box.height],
                          "score": float(face.categories[0].score),
                          "keypoints": [[p.x * width, p.y * height] for p in face.keypoints or []]})
        poses = [[{"x": float(p.x * width), "y": float(p.y * height),
                   "visibility": float(p.visibility or 0), "presence": float(p.presence or 0)}
                  for p in pose] for pose in pose_result.pose_landmarks]
        return {"faces": faces, "poses": poses}


def decode_png(data):
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Frame extraction failed")
    return frame


def sample_clip(path, start, end, info, config, detector):
    import cv2

    samples = []
    count = max(1, math.ceil((end - start) * config.sample_fps))
    for index in range(count):
        # Start, approximately uniform interior, and last actual video frame.
        time = start + index * max(0, end - start - 1 / info["fps"]) / max(1, count - 1)
        sample = {"time": time}
        try:
            frame = decode_png(frame_png(path, time, info))
            sample.update(detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except (RuntimeError, ValueError) as exc:
            sample.update(error=str(exc), faces=[], poses=[])
        samples.append(sample)
    return samples


def save_previews(path, directory, samples, crop, info):
    import cv2

    outputs = []
    for label, index in (("first", 0), ("middle", len(samples) // 2), ("last", len(samples) - 1)):
        sample = samples[index]
        frame = decode_png(frame_png(path, sample["time"], info))
        for face in sample.get("faces", []):
            l, t, r, b = map(round, face["box"])
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 2)
        for pose in sample.get("poses", []):
            for point in pose:
                if point["visibility"] >= 0.5:
                    cv2.circle(frame, (round(point["x"]), round(point["y"])), 3, (255, 100, 0), -1)
        if crop:
            x, y, w, h = crop
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text = f"{sample['time']:.3f}s | {sample.get('rejection_reason') or 'candidate - REVIEW REQUIRED'}"
        cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        output = directory / f"preview_{label}.jpg"
        # imencode + tofile supports Windows Unicode paths unlike some imwrite builds.
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Preview encoding failed")
        output.write_bytes(encoded.tobytes())
        outputs.append(output)
    return outputs
