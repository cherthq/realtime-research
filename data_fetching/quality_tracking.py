"""Shot-local face/body association and explicit, bounded proposal recovery.

Recovery proposes geometry only: it never invents pixels or certifies visibility.
No biometric embeddings, model downloads, or cross-shot identity assumptions.
"""

from copy import deepcopy
import math

from .core import assess_samples, fit_crop


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    return intersection / max(1, area(a) + area(b) - intersection)


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def expand(region, width, height, fraction):
    """Safety space is optional at source edges; required anatomy is not clipped."""
    l, t, r, b = region
    dx, dy = (r - l) * fraction, (b - t) * fraction
    return [max(0, l - dx), max(0, t - dy), min(width, r + dx), min(height, b + dy)]


def pair_assessment(face, pose, width, height, config, *, depth=None):
    geometry = config.geometry()
    if depth is not None:
        from dataclasses import replace

        geometry = replace(geometry, torso_depth=depth)
    if len(pose) < 13:
        return {"rejection_reason": "incomplete_pose"}
    sample = {"faces": [face], "poses": [pose]}
    result = assess_samples([sample], width, height, geometry)
    # Preserve the underlying cause instead of collapsing it into a generic failure.
    result["rejection_reason"] = (
        sample.get("rejection_reason") or result["rejection_reason"]
    )
    if (
        result["rejection_reason"] == "torso_region_outside_source"
        and geometry.torso_depth >= config.torso_min_depth
    ):
        # Requested lower-body space is a preference, not a measured landmark.
        # Use genuinely visible torso only; never clip required head/shoulders.
        from dataclasses import replace

        a, b = pose[11], pose[12]
        shoulder_width = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        available = (height - max(face["box"][3], a["y"], b["y"]) - 0.01) / max(
            1, shoulder_width
        )
        if config.torso_min_depth <= available < geometry.torso_depth:
            limited = {"faces": [face], "poses": [pose]}
            recovered = assess_samples(
                [limited], width, height, replace(geometry, torso_depth=available)
            )
            if not recovered["rejection_reason"]:
                result = {
                    **recovered,
                    "source_limited_torso": True,
                    "visible_torso_depth": available,
                }
    if not result["rejection_reason"] and face.get("keypoints"):
        nose = pose[0]
        l, t, r, b = face["box"]
        dx, dy = (r - l) * 0.35, (b - t) * 0.35
        if (
            min(nose.get("presence", 0), nose.get("visibility", 0))
            >= config.shoulder_confidence
        ):
            if not l - dx <= nose["x"] <= r + dx or not t - dy <= nose["y"] <= b + dy:
                result["rejection_reason"] = "face_pose_landmark_mismatch"
    return result


class SubjectTracker:
    def __init__(self, config):
        self.config = config
        self.track = 0
        self.previous_face = None
        self.last_observed = None

    def reset(self):
        self.track += 1
        self.previous_face = None
        self.last_observed = None

    def assess(self, detection, width, height, stamp):
        faces, poses = detection.get("faces", []), detection.get("poses", [])
        row = {
            "detections": deepcopy(detection),
            "face": None,
            "region": None,
            "wide_region": None,
            "core_region": None,
            "association": None,
            "flags": [],
            "uncertainties": [],
            "track_state": "unresolved",
            "track_id": str(self.track),
        }
        pairs, failures = [], []
        for fi, face in enumerate(faces):
            for pi, pose in enumerate(poses):
                # Association is independent of whether a requested torso can fit.
                assessed = pair_assessment(
                    face, pose, width, height, self.config, depth=0
                )
                if assessed["rejection_reason"]:
                    failures.append(assessed["rejection_reason"])
                else:
                    pairs.append((fi, pi))
        if len(pairs) != 1:
            if len(pairs) > 1:
                row["flags"] = ["multiple_or_ambiguous_person_associations"]
                self.reset()
            else:
                reason = (
                    "missing_face"
                    if not faces
                    else "missing_pose"
                    if not poses
                    else failures[0]
                    if failures
                    else "no_supported_face_body_pair"
                )
                row["uncertainties"] = [reason]
            row["track_id"] = str(self.track)
            return row
        fi, pi = pairs[0]
        face, pose = faces[fi], poses[pi]
        primary = pair_assessment(face, pose, width, height, self.config)
        row.update(
            face=face["box"],
            association={
                "face_index": fi,
                "pose_index": pi,
                "method": "face_shoulders_nose_geometry_and_temporal_iou",
            },
        )
        if self.previous_face is not None:
            if (
                stamp - self.last_observed > self.config.tracking_gap + 0.1
                or overlap(face["box"], self.previous_face) < self.config.track_iou
            ):
                self.reset()
                row["flags"].append("track_discontinuity")
        self.previous_face, self.last_observed = face["box"], stamp
        row["track_id"] = str(self.track)
        secondary = [i for i in range(len(faces)) if i != fi]
        row["association"]["secondary_face_indices"] = secondary
        if secondary:
            row["uncertainties"].append("unassociated_secondary_face")
            if any(
                faces[i]["score"] >= self.config.secondary_face_confidence
                for i in secondary
            ):
                row["uncertainties"].append("strong_secondary_face_review_required")
        if primary["rejection_reason"]:
            row["flags"].append(primary["rejection_reason"])
            return row
        if primary.get("source_limited_torso"):
            row["uncertainties"].append("torso_extent_limited_by_source")
            row["framing_evidence"] = {
                "visible_torso_depth": primary["visible_torso_depth"],
                "requested_depth": self.config.geometry().torso_depth,
            }
        region = primary["required_region"]
        required = expand(region, width, height, self.config.crop_safety)
        try:
            fit_crop(required, width, height, self.config.geometry())
        except ValueError as exc:
            row["flags"].append(str(exc))
            return row
        core = pair_assessment(face, pose, width, height, self.config, depth=0)
        wide = pair_assessment(
            face, pose, width, height, self.config, depth=self.config.torso_depth
        )
        row.update(
            region=required,
            core_region=core.get("required_region"),
            wide_region=wide.get("required_region")
            if not wide["rejection_reason"]
            else None,
            track_state="observed",
        )
        return row


def recover_rows(rows, max_gap, min_iou):
    """Keep brief misses reviewable without splitting; never bridge cuts/hard flags.

    Rows between observed anchors retain their raw missing detections and explicit
    uncertainties. The union envelope is a crop proposal, NOT measured visibility.
    The caller must insert a boundary before any shot transition.
    """
    previous, pending = None, []
    for row in rows:
        if row.get("boundary") or row["flags"]:
            yield from pending
            pending, previous = [], None
            yield row
            continue
        if row.get("region"):
            if pending:
                supported = (
                    previous is not None
                    and previous["track_id"] == row["track_id"]
                    and row["time"] - pending[0]["time"] <= max_gap + 1e-7
                    and overlap(previous["face"], row["face"]) >= min_iou
                )
                for missing in pending:
                    if supported:
                        for name in ("region", "core_region", "wide_region"):
                            a, b = previous.get(name), row.get(name)
                            missing[name] = union(a, b) if a and b else None
                        missing["track_state"] = "recovered_proposal"
                        missing["uncertainties"].append("detector_gap_review_required")
                        missing["recovery"] = {
                            "method": "bounded_anchor_union",
                            "anchors": [previous["time"], row["time"]],
                        }
                    yield missing
                pending = []
            previous = row
            yield row
        else:
            pending.append(row)
            if previous is None or row["end"] - pending[0]["time"] > max_gap + 1e-7:
                yield from pending
                pending, previous = [], None
    yield from pending


def mapped_region(region, crop, width, height):
    if region is None:
        return None
    x, y, w, h = crop
    return [
        (region[0] - x) * width / w,
        (region[1] - y) * height / h,
        (region[2] - x) * width / w,
        (region[3] - y) * height / h,
    ]


def region_inside(region, width, height, tolerance=0):
    return bool(
        region
        and all(math.isfinite(x) for x in region)
        and -tolerance <= region[0] < region[2] <= width + tolerance
        and -tolerance <= region[1] < region[3] <= height + tolerance
    )
