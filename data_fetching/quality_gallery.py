"""Local-only evidence gallery; media stays under the ignored dataset root."""

import html
import os
from pathlib import Path

from .core import read_json, within, write_json
from .pipeline import candidate_records, validate_candidate


def build_gallery(root, destination):
    destination = Path(destination).resolve()
    # A public docs directory must not accidentally receive copyrighted test media.
    destination.relative_to(Path(root).resolve())
    if (destination / "index.html").exists():
        raise ValueError("Gallery exists; choose a new result directory")
    records = [r for r in candidate_records(root) if r.get("schema_version") == 2]
    latest = {}
    for record in sorted(records, key=lambda r: r["created_at"]):
        latest[record["interval_id"]] = record
    records = list(latest.values())
    for record in records:
        validate_candidate(root, record)
    destination.mkdir(parents=True, exist_ok=True)
    esc = lambda value: html.escape(str(value), quote=True)
    relative = lambda path: Path(
        os.path.relpath(within(root, path), destination)
    ).as_posix()
    cards, report = [], []
    for row in sorted(
        records, key=lambda r: (r["source_id"], r["start"], r["processing_id"])
    ):
        findings = "".join(
            f"<tr><td>{esc(name)}</td><td>{esc(result['status'])}</td><td>{esc(result.get('reason') or ', '.join(sorted({f['code'] for f in result.get('findings', [])})))}</td></tr>"
            for name, result in row["checks"].items()
        )
        video = relative(row["artifacts"]["video"]["path"])
        poster = relative(row["artifacts"]["reference_image"]["path"])
        metadata = relative(
            str(Path(row["artifacts"]["video"]["path"]).parent / "candidate.json")
        )
        cards.append(
            f'<section><h2>{esc(row["source_id"])} · {row["start"]:.3f}–{row["end"]:.3f}s</h2><p>{esc(row.get("requested_framing_profile", row["framing"]))} · {esc(row["duration_bucket"])} · {esc(row["qc_disposition"])}</p><video controls preload="metadata" poster="{esc(poster)}" src="{esc(video)}"></video><details><summary>Checks and evidence</summary><table>{findings}</table><a href="{esc(metadata)}">Full metadata</a></details></section>'
        )
        report.append(
            {
                "id": row["id"],
                "start": row["start"],
                "end": row["end"],
                "duration": row["end"] - row["start"],
                "framing": row["framing"],
                "qc_disposition": row["qc_disposition"],
                "checks": {k: v["status"] for k, v in row["checks"].items()},
                "tracking": row.get("tracking_summary", {}),
                "training_approved": False,
            }
        )
    processing = read_json(root / "reports/quality-process.json")
    write_json(
        destination / "report.json",
        {
            "candidates": report,
            "processing": processing,
            "note": "Diagnostic gallery is not training or source approval. Counts include processing revisions, not unique approved hours.",
        },
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FaceTime pipeline evidence</title><style>body{background:#0b1421;color:#d9e5ee;font:16px/1.5 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}section{background:#172638;border:1px solid #34506a;border-radius:12px;padding:18px}h2{font-size:18px}video{width:100%;background:#000}a,summary{color:#91cfff}summary{cursor:pointer}table{font-size:12px;width:100%;overflow-wrap:anywhere}td{padding:5px;border-bottom:1px solid #34506a}.warning{color:#ffd166}</style><main><h1>Actual pipeline outputs</h1><p class="warning">Candidates only—not approved training data. Missing checks remain unverified. Expand each clip to inspect its checks.</p><p><a href="report.json">Scan reports, boundaries and candidate summaries</a></p><div class="grid">"""
    from .core import atomic_text

    atomic_text(
        destination / "index.html",
        page
        + (
            "".join(cards)
            or "<p>No candidates; inspect the scan report for exclusion reasons.</p>"
        )
        + "</div></main></html>",
    )
    return destination / "index.html"
