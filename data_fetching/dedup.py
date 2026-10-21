"""Perceptual duplicate proposals, never automatic identity or rights approval.

Visual temporal hashes flag reencodes and likely overlapping reuploads. Crops,
brief overlaps and visually similar backgrounds can defeat or confuse this
heuristic. Unresolved sources never enter confirmed unique-hour totals.
"""

from collections import Counter
import json

from .core import read_json, write_json


def dhash(frame):
    import cv2
    gray = cv2.cvtColor(cv2.resize(frame, (9, 8)), cv2.COLOR_BGR2GRAY)
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


def signatures(path, duration, step=5):
    import cv2
    cap = cv2.VideoCapture(str(path))
    result = []
    try:
        cursor = 0
        while cursor < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, cursor * 1000)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                # Include a center crop to propose common letterboxed variants.
                cropped = frame[h // 10:h - h // 10, w // 10:w - w // 10]
                result.append({"time": cursor, "full": str(dhash(frame)), "center": str(dhash(cropped))})
            cursor += step
    finally:
        cap.release()
    return result


def matches(first, second, minimum=6):
    votes, matched, distances = Counter(), {}, Counter()
    for a in first:
        for b in second:
            # Do not match a full frame to a different crop geometry.
            distance = min((int(a[k]) ^ int(b[k])).bit_count() for k in ("full", "center"))
            if distance <= 5:
                offset = round((a["time"] - b["time"]) / 5) * 5
                votes[offset] += 1
                distances[offset] += distance
                matched.setdefault(offset, set()).add(b["time"])
    if not votes:
        return None
    offset = max(votes, key=lambda x: (len(matched[x]), -distances[x] / votes[x]))
    times = matched[offset]
    if len(times) < minimum or max(times) - min(times) < 25:
        return None
    return {"suggested_offset_seconds": offset, "matched_samples": len(times),
            "method": "temporal_dhash_proposal", "status": "needs_manual_alignment_review"}


def screen(queue, source):
    directory = queue.root / "fingerprints"
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / (source["sha256"] + ".json")
    if not cache.exists():
        write_json(cache, signatures(source["original"], json.loads(source["media"])["duration"]))
    current = read_json(cache)
    proposals = []
    for other in queue.sources():
        if not other["sha256"] or other["id"] == source["id"] or other["sha256"] == source["sha256"]:
            continue
        prior = directory / (other["sha256"] + ".json")
        if prior.exists():
            proposal = matches(read_json(prior), current)
            if proposal:
                proposals.append({"other_source_id": other["id"], **proposal})
    write_json(directory / (source["id"] + "_proposals.json"), proposals)
    if proposals and source["duplicate_status"] == "unverified":
        with queue.transaction():
            queue.db.execute("UPDATE sources SET duplicate_status='suspected_duplicate' WHERE id=?", (source["id"],))
            queue.event("duplicate_proposals", {"id": source["id"], "proposals": proposals})
    return proposals
