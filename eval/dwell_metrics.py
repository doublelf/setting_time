"""Evaluate dwell-time accuracy against a hand-made ground truth.

The ground truth is a CSV with one row per person:

    person_id,start_s,end_s

``scripts/make_occlusion_test.py`` produces such a reference automatically for the
occlusion clips; for a real installation the file is filled in by hand.

Metrics follow the convention used in the retail-analytics literature: the mean
absolute error is reported next to the median absolute error, because dwell times are
right-skewed and the mean alone hides how many visits are badly wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from peopleflow.config import load_config  # noqa: E402
from peopleflow.pipeline import PeopleFlowPipeline  # noqa: E402


def load_ground_truth(path: Path) -> list[tuple[str, float, float]]:
    """Read a ground-truth CSV into (person_id, start_s, end_s) tuples."""
    entries = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            person = row.get("person_id") or row.get("id") or ""
            start = row.get("start_s") or row.get("start") or ""
            end = row.get("end_s") or row.get("end") or ""
            if person and start and end:
                entries.append((person, float(start), float(end)))
    return entries


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Length of the overlap between two time intervals."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match(predictions, truth) -> list[tuple[object, object, float]]:
    """Greedily pair predictions with ground truth by temporal overlap."""
    pairs = []
    used = set()
    for prediction in predictions:
        best, best_overlap = None, 0.0
        for index, entry in enumerate(truth):
            if index in used:
                continue
            amount = overlap((prediction.first_seen, prediction.last_seen), (entry[1], entry[2]))
            if amount > best_overlap:
                best, best_overlap = index, amount
        if best is not None:
            used.add(best)
            pairs.append((prediction, truth[best], best_overlap))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dwell times against a ground truth.")
    parser.add_argument("--source", required=True, help="video the ground truth refers to")
    parser.add_argument("--ground-truth", required=True, help="CSV with person_id,start_s,end_s")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", help="output directory")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path

    cfg = load_config(args.config, root=ROOT)
    cfg.source.path = str(source)
    cfg.output.overlay = False
    out_dir = Path(args.out) if args.out else ROOT / "out" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = PeopleFlowPipeline(cfg).run(out_dir=out_dir, progress=False)
    truth = load_ground_truth(gt_path)
    pairs = match(result.records, truth)

    errors = []
    print(f"\n=== dwell time vs ground truth ({len(truth)} people) ===")
    print(f"{'person':>10} {'gt(s)':>9} {'pred(s)':>9} {'error(s)':>9} {'overlap(s)':>10}")
    for prediction, entry, amount in pairs:
        gt_dwell = entry[2] - entry[1]
        error = prediction.dwell_span_s - gt_dwell
        errors.append(abs(error))
        print(f"{entry[0]:>10} {gt_dwell:>9.1f} {prediction.dwell_span_s:>9.1f} {error:>9.1f} {amount:>10.1f}")

    unmatched_truth = len(truth) - len(pairs)
    unmatched_pred = len(result.records) - len(pairs)

    print("\n=== metrics ===")
    if errors:
        arr = np.array(errors)
        print(f"  dwell MAE         {arr.mean():.2f} s")
        print(f"  dwell median AE   {np.median(arr):.2f} s")
        print(f"  dwell max AE      {arr.max():.2f} s")
    else:
        print("  no prediction could be matched to the ground truth")
    print(f"  people predicted  {len(result.records)} (ground truth {len(truth)})")
    print(f"  count MAE         {abs(len(result.records) - len(truth)) / max(1, len(truth)):.2f}")
    print(f"  unmatched truth   {unmatched_truth}")
    print(f"  unmatched pred    {unmatched_pred}")

    summary = {
        "source": str(source),
        "ground_truth": str(gt_path),
        "people_ground_truth": len(truth),
        "people_predicted": len(result.records),
        "dwell_mae_s": float(np.mean(errors)) if errors else None,
        "dwell_median_ae_s": float(np.median(errors)) if errors else None,
        "dwell_max_ae_s": float(np.max(errors)) if errors else None,
        "unmatched_truth": unmatched_truth,
        "unmatched_pred": unmatched_pred,
        "run_stats": result.stats,
    }
    (out_dir / "dwell_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'dwell_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
