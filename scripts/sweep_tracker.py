"""Sweep tracker configurations and score them on the test clips.

Ground truth (number of distinct people visible) for the clips used as benchmarks:

  supermarket_browse  1
  lobby_static        1
  fashion_mall        3

For those, the number of raw tracks should be as close to the truth as possible: too
many means the tracker fragments one person into several ids, too few means people are
being merged together.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peopleflow.config import load_config  # noqa: E402
from peopleflow.pipeline import PeopleFlowPipeline  # noqa: E402

BASE = yaml.safe_load((ROOT / "configs" / "trackers" / "tracktrack_reid.yaml").read_text(encoding="utf-8"))

CLIPS = [
    ("supermarket_browse", 1),
    ("lobby_static", 1),
    ("fashion_mall", 3),
    ("coffee_shop", None),
    ("mall_busy_timelapse", None),
]

VARIANTS: dict[str, dict] = {
    "current": {},
    "low_thresh": {"track_high_thresh": 0.25, "track_low_thresh": 0.10, "new_track_thresh": 0.40},
    "stock_thresh": {"track_high_thresh": 0.60, "track_low_thresh": 0.25, "new_track_thresh": 0.70},
    "lenient_match": {"match_thresh": 0.85},
    "strict_match": {"match_thresh": 0.55},
    "no_lost_rebind": {"lost_match_thr": 0.0},
    "reid_heavy": {"reid_weight": 0.75, "iou_weight": 0.25},
    "iou_heavy": {"reid_weight": 0.25, "iou_weight": 0.75},
    "long_buffer": {"track_buffer": 300},
    "short_buffer": {"track_buffer": 45},
    "tai_tight": {"tai_thr": 0.35},
    "minlen1": {"min_track_len": 1},
}


def run(variant: dict, clip: str) -> dict:
    """Run the pipeline for one configuration on one clip and return summary metrics."""
    data = copy.deepcopy(BASE)
    data.update(variant)

    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    handle.write(yaml.safe_dump(data, sort_keys=False))
    handle.close()

    cfg = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    cfg.source.path = str(ROOT / "assets" / "videos" / f"{clip}.mp4")
    cfg.tracker.config = handle.name
    cfg.output.overlay = False
    cfg.output.write_json = False
    cfg.output.write_events = False
    cfg.output.write_review = False

    result = PeopleFlowPipeline(cfg).run(out_dir=ROOT / "out" / "_sweep", progress=False)
    Path(handle.name).unlink(missing_ok=True)

    records = result.records
    span = sum(r.dwell_span_s for r in records)
    visible = sum(r.dwell_visible_s for r in records)
    return {
        "raw_tracks": result.stats["raw_tracks"],
        "reported": len(records),
        "merged_pairs": result.stats["merged_pairs"],
        "visible_ratio": round(visible / span, 3) if span else 0.0,
        "max_span": round(max((r.dwell_span_s for r in records), default=0.0), 1),
    }


def main() -> None:
    print(f"{'variant':16} {'clip':22} {'truth':>5} {'raw':>4} {'rep':>4} {'mrg':>4} {'vis%':>6} {'maxsp':>6}")
    print("-" * 78)
    table: dict[str, dict[str, dict]] = {}
    for name, variant in VARIANTS.items():
        table[name] = {}
        for clip, truth in CLIPS:
            metrics = run(variant, clip)
            table[name][clip] = metrics
            flag = ""
            if truth is not None:
                delta = abs(metrics["raw_tracks"] - truth)
                flag = "  ok" if delta <= 1 else f"  off by {delta:+d}"
            print(
                f"{name:16} {clip:22} {str(truth or '-'):>5} {metrics['raw_tracks']:>4} "
                f"{metrics['reported']:>4} {metrics['merged_pairs']:>4} "
                f"{metrics['visible_ratio']:>6.2f} {metrics['max_span']:>6.1f}{flag}"
            )
        print("-" * 78)

    (ROOT / "out" / "_sweep_results.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
