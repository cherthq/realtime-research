"""Duration-weighted evaluation against manually labelled source intervals.

This measures proposal selection, not learned-model certification. Labels must
come from full interval review, and calibration/test groups must be disjoint.
"""

from .core import read_jsonl
from .metadata import finite, ranges_union, seconds


def evaluate(records, annotations, stage="candidates", approved_ids=()):
    if stage not in {"candidates", "eligible", "approved"}:
        raise ValueError("Unknown benchmark stage")
    approved = set(approved_ids)
    selected, known_hashes, labels, partitions = {}, {}, {}, {}
    for record in records:
        source = record["source_id"]
        known_hashes.setdefault(source, set()).add(record["source"]["sha256"])
        include = (
            stage == "candidates"
            or stage == "eligible"
            and record["qc_disposition"] == "eligible"
            or stage == "approved"
            and record["id"] in approved
        )
        if include:
            selected.setdefault(source, []).append((record["start"], record["end"]))
    for label in annotations:
        a, b, source = label.get("start"), label.get("end"), label.get("source_id")
        if (
            not source
            or not finite(a)
            or not finite(b)
            or not 0 <= a < b
            or label.get("label") not in {"usable", "unusable", "uncertain"}
        ):
            raise ValueError(
                "Labels need source_id, finite bounds and usable/unusable/uncertain label"
            )
        if (
            not label.get("reviewer")
            or not label.get("evidence")
            or not label.get("source_sha256")
        ):
            raise ValueError("Labels need reviewer, evidence and original SHA-256")
        if source in known_hashes and known_hashes[source] != {label["source_sha256"]}:
            raise ValueError("Benchmark source hash mismatch")
        partition = label.get("partition")
        group = label.get("group_id")
        if partition not in {"calibration", "test"} or not group:
            raise ValueError(
                "Labels need calibration/test partition and source/speaker group_id"
            )
        for key in (
            ("group", group),
            ("source", source),
            ("original", label["source_sha256"]),
        ):
            if key in partitions and partitions[key] != partition:
                raise ValueError("Calibration/test group/source leakage")
            partitions[key] = partition
        previous = labels.setdefault(source, [])
        if any(max(a, row["start"]) < min(b, row["end"]) for row in previous):
            raise ValueError("Overlapping benchmark labels")
        previous.append(label)
    results = {}
    for partition in ("calibration", "test"):
        totals = {
            "usable_seconds": 0,
            "usable_retained_seconds": 0,
            "unusable_seconds": 0,
            "unusable_retained_seconds": 0,
            "uncertain_label_seconds": 0,
        }
        for source, rows in labels.items():
            spans = ranges_union(selected.get(source, []))
            for row in rows:
                if row["partition"] != partition:
                    continue
                length = row["end"] - row["start"]
                if row["label"] == "uncertain":
                    totals["uncertain_label_seconds"] += length
                    continue
                retained = seconds(
                    [
                        (max(a, row["start"]), min(b, row["end"]))
                        for a, b in spans
                        if max(a, row["start"]) < min(b, row["end"])
                    ]
                )
                totals[row["label"] + "_seconds"] += length
                totals[row["label"] + "_retained_seconds"] += retained
        retained = (
            totals["usable_retained_seconds"] + totals["unusable_retained_seconds"]
        )
        totals["retained_precision"] = (
            totals["usable_retained_seconds"] / retained if retained else None
        )
        totals["usable_recall"] = (
            totals["usable_retained_seconds"] / totals["usable_seconds"]
            if totals["usable_seconds"]
            else None
        )
        results[partition] = totals
    return {
        "stage": stage,
        "partitions": results,
        "note": "Candidate retention is not training approval. Unlabelled footage and uncertain labels do not establish precision.",
    }


def benchmark(root, labels, stage="candidates", reviews=None):
    from .pipeline import build_export, candidate_records

    if stage == "approved" and reviews is None:
        raise ValueError("Approved benchmark requires bound reviews")
    from .core import read_json

    approved = [r["id"] for r in build_export(root, reviews)] if reviews else []
    annotations = read_jsonl(labels)
    for annotation in annotations:
        source = annotation.get("source_id", "")
        from .core import SAFE_ID

        if not isinstance(source, str) or not SAFE_ID.fullmatch(source):
            raise ValueError("Invalid benchmark source ID")
        original = read_json(root / "sources" / (source + ".json"))
        if original["sha256"] != annotation.get("source_sha256"):
            raise ValueError("Benchmark label does not match acquired original")
        if (
            not finite(annotation.get("end"))
            or annotation["end"] > original["media"]["duration"]
        ):
            raise ValueError("Benchmark label escapes source duration")
    return evaluate(candidate_records(root), annotations, stage, approved)
