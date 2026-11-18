"""Measure how well the appearance embedding separates people.

The tracker's own ids are used as ground truth: embeddings of the same track should
have a much higher cosine similarity than embeddings of different tracks.  This is the
signal the identity merger relies on, and it is the number to look at when tuning
``min_cosine`` / ``merge_threshold``.

    python scripts/bench/measure_reid_quality.py --video assets/videos/coffee_shop.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from peopleflow.config import load_config  # noqa: E402
from peopleflow.tracker import PeopleTracker  # noqa: E402


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="assets/videos/coffee_shop.mp4")
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video

    cfg = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    tracker = PeopleTracker(cfg)

    cap = cv2.VideoCapture(str(video))
    per_track: dict[int, list[np.ndarray]] = defaultdict(list)
    frames = 0
    while frames < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        for obs in tracker.update(frame):
            if obs.embedding is not None:
                per_track[obs.track_id].append(obs.embedding)
        frames += 1
    cap.release()

    tracks = {tid: embs for tid, embs in per_track.items() if len(embs) >= 2}
    same: list[float] = []
    for embs in tracks.values():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                same.append(cos(embs[i], embs[j]))

    ids = sorted(tracks)
    different: list[float] = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            for ea in tracks[ids[a]]:
                for eb in tracks[ids[b]]:
                    different.append(cos(ea, eb))

    print(f"\nvideo={video.name}  frames={frames}  tracks={len(tracks)}")
    if not same or not different:
        print("not enough data")
        return 1

    same_arr, diff_arr = np.array(same), np.array(different)
    print(
        f"  same track : mean={same_arr.mean():.3f} p05={np.percentile(same_arr, 5):.3f} "
        f"p50={np.percentile(same_arr, 50):.3f}"
    )
    print(
        f"  different  : mean={diff_arr.mean():.3f} p95={np.percentile(diff_arr, 95):.3f} "
        f"p99={np.percentile(diff_arr, 99):.3f} max={diff_arr.max():.3f}"
    )

    thresholds = np.arange(0.0, 1.0, 0.01)
    best = max(
        thresholds,
        key=lambda t: ((same_arr >= t).mean() - (diff_arr >= t).mean(), (same_arr >= t).mean()),
    )
    print(
        f"  best cosine threshold = {best:.2f}  "
        f"recall(same)={(same_arr >= best).mean():.3f}  "
        f"false-accept(diff)={(diff_arr >= best).mean():.3f}"
    )
    print("\n  pick min_cosine just above the different-track p95 and merge_threshold near the best threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
