"""Result writers: dwell records, event stream and human-review queue."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .merger import DwellRecord

RECORD_FIELDS = [
    "merged_id",
    "track_ids",
    "first_seen_s",
    "last_seen_s",
    "dwell_span_s",
    "dwell_visible_s",
    "occlusion_s",
    "n_frames",
    "max_gap_s",
    "exit_type",
    "merge_confidence",
    "needs_review",
    "mean_height_px",
]


def _row(record: DwellRecord) -> dict:
    """Flatten a dwell record into a CSV row."""
    return {
        "merged_id": record.merged_id,
        "track_ids": "|".join(str(t) for t in record.track_ids),
        "first_seen_s": round(record.first_seen, 3),
        "last_seen_s": round(record.last_seen, 3),
        "dwell_span_s": round(record.dwell_span_s, 3),
        "dwell_visible_s": round(record.dwell_visible_s, 3),
        "occlusion_s": round(record.occlusion_s, 3),
        "n_frames": record.n_frames,
        "max_gap_s": round(record.max_gap_s, 3),
        "exit_type": record.exit_type,
        "merge_confidence": round(record.merge_confidence, 4),
        "needs_review": int(record.needs_review),
        "mean_height_px": round(record.mean_height, 1),
    }


def write_records_csv(records: list[DwellRecord], path: str | Path) -> Path:
    """Write the dwell records as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_row(record))
    return path


def write_records_json(records: list[DwellRecord], path: str | Path, extra: dict | None = None) -> Path:
    """Write the dwell records plus run metadata as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"records": [_row(r) for r in records]}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def write_events_jsonl(events: list[dict], path: str | Path) -> Path:
    """Write the per-event stream (track lifecycle) as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def write_review_csv(records: list[DwellRecord], path: str | Path) -> Path | None:
    """Write the low-confidence merges that still need a human decision."""
    pending = [r for r in records if r.needs_review]
    if not pending:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in pending:
            writer.writerow(_row(record))
    return path
