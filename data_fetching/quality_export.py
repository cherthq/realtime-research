"""Non-destructive discovery/legacy import, review reports and local Hub packaging."""

from pathlib import Path
import shutil

from .core import (Source, fingerprint, read_json, read_jsonl,
                   within, write_json, write_jsonl)
from .metadata import finite
from .pipeline import build_export, candidate_records


def import_discovery(path):
    """Return sources and an audit envelope; geometry-only proposals are never approvals."""
    sources, envelopes, seen = {}, [], set()
    for upstream in read_jsonl(path):
        if upstream.get("schema_version") == 2 and "source" in upstream and isinstance(upstream["source"], dict):
            definition = upstream["source"]["definition"]
        elif "source" in upstream and isinstance(upstream["source"], dict) and "definition" in upstream["source"]:
            definition = upstream["source"]["definition"]
        else:
            definition = upstream
        url = definition.get("url") or definition.get("source_url")
        local = definition.get("path")
        if not url and not local:
            raise ValueError("Discovery row requires explicit url/source_url or local path")
        key = fingerprint(url or local)[:24]
        source = Source.parse({"id": "import_" + key, **({"url": url} if url else {"path": local}),
                               "license": definition.get("license") or "unknown",
                               "license_evidence": definition.get("license_evidence") if isinstance(definition.get("license_evidence"), str) else "",
                               "source": "imported discovery; permissions and quality pending"}, Path(path).resolve().parent)
        proposals = []
        for proposal in upstream.get("clean_intervals", []):
            a, b = proposal.get("start"), proposal.get("end")
            if not finite(a) or not finite(b) or not 0 <= a < b:
                raise ValueError("Invalid proposed clean interval")
            logical = fingerprint([source.url or source.path, a, b])
            if logical not in seen:
                proposals.append({"start": a, "end": b, "status": "unverified"})
                seen.add(logical)
        canonical = source.url or source.path
        source = Source.parse({**source.to_dict(), "id": "import_" + fingerprint(canonical)[:24]}, Path(path).resolve().parent)
        sources[source.id] = source.to_dict()
        envelopes.append({"source_id": source.id, "upstream": upstream,
                          "proposed_intervals": proposals, "quality_status": "unverified",
                          "source_approved": False, "legacy_reviews_require_revalidation": True})
    return list(sources.values()), envelopes


def package_hub(root, reviews, destination):
    """Prepare explicit approved-only VideoFolder package. Never publishes anything."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Hub package destination already exists; choose a new directory")
    rows = build_export(root, reviews)
    if not rows:
        raise ValueError("No approved samples to package")
    if any(row.get("schema_version") != 2 or not row.get("redistribution_approved") for row in rows):
        raise ValueError("Hub packaging requires v2 reviews and separate redistribution_approved")
    # Validate all inputs before creating a package; interrupted packages have no COMPLETE marker.
    grouped = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)
    destination.mkdir(parents=True)
    for split, samples in grouped.items():
        directory = destination / split
        directory.mkdir()
        metadata = []
        for row in samples:
            value = dict(row)
            for key, extension in (("video", ".mp4"), ("audio", ".wav"), ("reference_image", ".png"), ("qc_metadata", ".json")):
                # Only MP4 is a VideoFolder media row; companion files are metadata paths.
                name = row["id"] + extension
                target_dir = directory if key == "video" else destination / "companions" / split
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(within(root, row[key]), target_dir / name)
                value[key] = (target_dir / name).relative_to(destination).as_posix()
            value["file_name"] = row["id"] + ".mp4"
            metadata.append(value)
        write_jsonl(directory / "metadata.jsonl", metadata)
    (destination / "README.md").write_text("# Reviewed FaceTime-style intervals\n\nApproved rendered intervals only.\n"
        "Per-row license/evidence apply; no blanket license is granted.\n"
        "QC sidecars retain provenance; companion evidence remains in the private processing dataset.\n", encoding="utf-8")
    write_json(destination / "COMPLETE.json", {"rows": len(rows), "splits": {k: len(v) for k, v in grouped.items()}})
    return len(rows)


def duplicate_proposals(root, max_pairs=100000):
    """Temporal dHash candidates only; never assigns identity/content groups automatically."""
    records = [r for r in candidate_records(root) if r.get("schema_version") == 2]
    signatures = {}
    for row in records:
        path = within(root, row["artifacts"]["evidence_output-tracks.json"]["path"])
        tracks = read_json(path)
        if tracks:
            signatures[row["id"]] = [int(tracks[round(i * (len(tracks) - 1) / 7)]["metrics"]["dhash"], 16) for i in range(8)]
    result, compared = [], 0
    for index, a in enumerate(records):
        for b in records[index + 1:]:
            if compared >= max_pairs:
                return {"proposals": result, "truncated": True, "pairs_compared": compared}
            if a["source"]["sha256"] == b["source"]["sha256"] or a["id"] not in signatures or b["id"] not in signatures:
                continue
            compared += 1
            distances = [(x ^ y).bit_count() / 256 for x, y in zip(signatures[a["id"]], signatures[b["id"]])]
            if sum(d < 0.12 for d in distances) >= 6:
                result.append({"a": a["id"], "b": b["id"], "status": "review_required",
                               "method": "aligned_temporal_dhash", "distances": distances,
                               "limitation": "Different trims/crops may be missed; similar backgrounds may false-match"})
    return {"proposals": result, "truncated": False, "pairs_compared": compared}
