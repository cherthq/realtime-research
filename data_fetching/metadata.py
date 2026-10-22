"""Version-2 quality metadata, half-open interval accounting and review gates."""

from datetime import datetime, timezone
import math

from .core import fingerprint

CHECK_STATES = {"pass", "flag", "fail", "unverified", "error", "not_applicable"}
REQUIRED_CHECKS = ("continuity", "framing", "visual_quality", "text_overlay", "layout",
                   "appearance", "audio_quality", "speech", "active_speaker", "lip_sync",
                   "render_validation")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def ranges_union(ranges):
    merged = []
    for a, b in sorted(ranges):
        if not finite(a) or not finite(b) or not 0 <= a < b:
            raise ValueError("Invalid half-open time interval")
        if merged and a <= merged[-1][1] + 1e-7:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def seconds(ranges):
    return sum(b - a for a, b in ranges_union(ranges))


def subtract_ranges(start, end, blocked):
    cursor = start
    clipped = [(max(start, a), min(end, b)) for a, b in blocked if a < end and b > start]
    for a, b in ranges_union(clipped):
        if a > cursor:
            yield cursor, a
        cursor = max(cursor, b)
    if cursor < end:
        yield cursor, end


def interval_id(source_hash, start_pts, end_pts, time_base):
    return fingerprint([source_hash, start_pts, end_pts, str(time_base)])[:32]


def check(status, method, ranges=(), findings=(), **extra):
    value = {"status": status, "method": method, "analyzed_ranges": ranges_union(ranges),
             "findings": list(findings), **extra}
    validate_check(value)
    return value


def validate_check(value):
    if not isinstance(value, dict) or value.get("status") not in CHECK_STATES:
        raise ValueError("Invalid check status")
    if not isinstance(value.get("method"), str) or not value["method"]:
        raise ValueError("Check method required")
    ranges_union(value.get("analyzed_ranges", []))
    if value["status"] == "pass" and not value.get("analyzed_ranges"):
        raise ValueError("A pass requires actual inspected ranges")
    if value["status"] == "not_applicable" and not value.get("reason"):
        raise ValueError("not_applicable requires a reason")
    if not isinstance(value.get("findings", []), list):
        raise ValueError("Check findings must be a list")
    for finding in value.get("findings", []):
        if not isinstance(finding, dict) or not isinstance(finding.get("code"), str):
            raise ValueError("Finding requires a reason code")
        if not finite(finding.get("start")) or not finite(finding.get("end")) or not 0 <= finding["start"] < finding["end"]:
            raise ValueError("Finding requires finite time bounds")
    # JSON serialization additionally rejects NaN anywhere in nested structures.
    fingerprint(value)


def disposition(checks):
    for name in REQUIRED_CHECKS:
        validate_check(checks[name])
    if any(c["status"] == "fail" for c in checks.values()):
        return "rejected"
    return "eligible" if all(checks[k]["status"] == "pass" for k in REQUIRED_CHECKS) else "needs_review"


def review_binding(record):
    return fingerprint({"interval_id": record["interval_id"], "processing_id": record["processing_id"],
                        "checks": record["checks"], "artifacts": record["artifacts"],
                        "crop": record["crop"]})


def validate_v2(record):
    if record.get("schema_version") != 2:
        raise ValueError("Expected metadata schema_version 2")
    if not finite(record.get("start")) or not finite(record.get("end")) or not 0 <= record["start"] < record["end"]:
        raise ValueError("Invalid candidate bounds")
    if record["end"] > record["source"]["media"]["duration"] + 0.05:
        raise ValueError("Interval exceeds source")
    for key in ("interval_id", "processing_id", "parent_span_id", "track_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"Missing {key}")
    if disposition(record["checks"]) != record.get("qc_disposition"):
        raise ValueError("QC disposition does not match checks")
    fingerprint(record)


def validate_review(record, review):
    validate_v2(record)
    if review.get("review_binding") != review_binding(record):
        raise ValueError("Stale or missing v2 review binding; regenerate template")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or not review.get("reviewed_at"):
        raise ValueError("Reviewer and UTC reviewed_at required")
    try:
        stamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("Not UTC")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("reviewed_at must be an ISO UTC timestamp") from exc
    decisions = review.get("check_decisions", {})
    if not isinstance(decisions, dict):
        raise ValueError("check_decisions must be an object")
    for name in REQUIRED_CHECKS:
        result = record["checks"][name]
        if result["status"] != "pass":
            override = decisions.get(name, {})
            if not (isinstance(override, dict) and override.get("status") == "approved"
                    and isinstance(override.get("reason"), str) and override["reason"].strip()
                    and isinstance(override.get("evidence"), list) and override["evidence"]
                    and all(isinstance(item, str) and item.strip() for item in override["evidence"])):
                raise ValueError(f"Unresolved check: {name}; documented human decision required")
        if name == "render_validation" and result["status"] != "pass":
            raise ValueError("Corrupt/unvalidated renders cannot be overridden")
    if record["duration_bucket"] != "long" and review.get("include_short") is not True:
        raise ValueError("Short footage excluded by default; explicitly set include_short")
    if record.get("speech_bucket") == "listening" and review.get("include_listening") is not True:
        raise ValueError("Listening footage excluded by default")


def unique_hours(records):
    groups = {}
    for record in records:
        groups.setdefault(record["source"]["sha256"], []).append((record["start"], record["end"]))
    return sum(seconds(spans) for spans in groups.values()) / 3600
