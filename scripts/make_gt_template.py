"""Create an empty ground-truth template for manual annotation.

Fill in one row per person: when they became visible and when they left.  The
evaluator matches predictions to these rows by temporal overlap.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER = ["person_id", "start_s", "end_s"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a ground-truth template.")
    parser.add_argument("--out", default=str(ROOT / "templates" / "ground_truth.csv"))
    parser.add_argument("--rows", type=int, default=10, help="number of empty rows")
    args = parser.parse_args()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for index in range(1, args.rows + 1):
            writer.writerow([index, "", ""])

    print(f"wrote {path}")
    print("fill in start_s / end_s for every person, then run:")
    print(f"  python eval/dwell_metrics.py --source <video> --ground-truth {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
