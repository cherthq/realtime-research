"""Explicit, allowlisted uploads to a Hugging Face dataset.

Public redistribution approval is independent of quality and training approval.
Tokens come only from the Hub's normal local credential mechanisms.
"""

import json
from pathlib import Path
import time

from .core import file_hash, write_json


def publication_manifest(queue):
    records, files = [], []
    for source in queue.sources():
        if not source["original"] or source["redistribution_status"] != "approved":
            continue
        if not source["evidence"] or not source["reviewer"]:
            raise ValueError("Redistribution decision lacks evidence/reviewer")
        original = Path(source["original"])
        if file_hash(original) != source["sha256"]:
            raise ValueError("Original changed; refusing publication")
        metadata = json.loads(source["metadata"])
        rejections = [json.loads(r[0])["reason"] for r in queue.db.execute(
          "SELECT payload FROM events WHERE kind='source_quality_rejection' ORDER BY id")
          if json.loads(r[0])["id"] == source["id"]]
        remote = f"originals/{source['sha256']}{original.suffix}"
        files.append((original, remote, source["sha256"]))
        intervals = []
        for r in queue.db.execute("SELECT * FROM intervals WHERE source_id=? ORDER BY start", (source["id"],)):
            intervals.append({k: r[k] for k in ("id", "revision", "start", "end", "framing", "method", "quality", "inspection", "speaker_id", "conversation_id", "split")}
                            | {"issues": json.loads(r["issues"])})
        records.append({"id": source["id"], "source_url": metadata.get("source_page", source["url"]),
          "original_uploader": metadata.get("uploader"), "title": metadata.get("title"),
          "creator_credit": metadata.get("creator_credit", metadata.get("uploader")),
          "license": metadata.get("license", "unknown"), "permission_evidence": source["evidence"],
          "license_url": metadata.get("license_url"),
          "license_evidence_checked": metadata.get("license_evidence_checked"),
          "modifications": ["video/audio streams merged without re-encoding when downloaded separately"],
          "redistribution_status": source["redistribution_status"],
          "training_permission_status": source["permission_status"],
          "curation_state": source["state"], "source_screening_rejections": rejections,
          "sampled_quality_observations": metadata.get("sampled_quality_observations"),
          "duplicate_status": source["duplicate_status"], "canonical_recording": source["canonical"],
          "timeline_offset_seconds": source["offset"], "original_sha256": source["sha256"],
          "original_path": remote, "duration_seconds": json.loads(source["media"])["duration"],
          "acquired_at": source["acquired"], "intervals": intervals})
    return records, list({remote: (local, remote, digest) for local, remote, digest in files}.values())


def publish(queue, repo_id, dry_run=False):
    records, files = publication_manifest(queue)
    if not records:
        return {"state": "no_redistribution_approved_sources", "files": 0}
    if dry_run:
        return {"repo_id": repo_id, "sources": len(records), "files": [p for _, p, _ in files],
                "bytes": sum(p.stat().st_size for p, _, _ in files)}
    from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_url, get_hf_file_metadata

    if not get_token():
        raise ValueError("Hugging Face authentication missing; use hf auth login or HF_TOKEN locally")
    api = HfApi()
    info = api.dataset_info(repo_id)
    repo_id = info.id  # Follow an owner-authorized repository transfer/redirect.
    pending = []
    for local, remote, digest in files:
        uploaded = queue.db.execute("SELECT sha256 FROM uploads WHERE path=?", (repo_id + "/" + remote,)).fetchone()
        if not uploaded or uploaded[0] != digest:
            pending.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
    manifest = ("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n").encode()
    pending.append(CommitOperationAdd(path_in_repo="discovery/sources.jsonl", path_or_fileobj=manifest))
    # Keep an existing owner's dataset card intact; bootstrap only an empty card.
    existing = {s.rfilename for s in info.siblings}
    if "README.md" not in existing:
        card = '''---
pretty_name: FaceTime Video Curation Queue
tags:
- video
- human-review-required
---
# FaceTime video curation queue

Original real-person recordings and continuous-interval curation records.
This is a discovery/pilot collection. Inclusion does not establish training
approval. Inspect each source's license and permission status, quality review,
duplicate status and interval timestamps in `discovery/sources.jsonl`.

Only sources with recorded redistribution evidence are stored here. Individual
recordings retain their stated licenses and attribution; no blanket dataset
license is asserted. Sampled geometry checks do not verify overlays, natural
speech, speaker match or cleanliness throughout an interval. Pending intervals
must not be counted as approved usable hours. Keep full originals and never
concatenate shots or duplicate footage to reach a duration target.
'''
        pending.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    commit = api.create_commit(repo_id=repo_id, repo_type="dataset", operations=pending,
                               commit_message="Store attributable source originals and pending curation records",
                               parent_commit=info.sha)
    # Verify content-addressed video objects on the committed revision. ETags for
    # LFS/Xet large files represent the source SHA-256; fail closed otherwise.
    for local, remote, digest in files:
        metadata = get_hf_file_metadata(hf_hub_url(repo_id, remote, repo_type="dataset", revision=commit.oid))
        if metadata.size != local.stat().st_size or metadata.etag != digest:
            raise ValueError(f"Uploaded object hash/size could not be verified: {remote}; local media retained")
        with queue.transaction():
            queue.db.execute("INSERT OR REPLACE INTO uploads VALUES(?,?,?,?)",
                             (repo_id + "/" + remote, digest, commit.oid, time.time()))
    result = {"state": "uploaded_and_verified", "repo_id": repo_id, "commit": commit.oid,
              "sources": len(records), "video_files": len(files)}
    write_json(queue.root / "upload-report.json", result)
    return result
