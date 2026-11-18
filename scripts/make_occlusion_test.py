"""Build a video with a controlled occlusion, together with its ground truth.

The tracker is run on the source clip to find the person that stays longest on screen.
That person's box is then blacked out for a configurable window, which reproduces what
a shelf, a pillar or a crowd does to a real installation.

The ground truth is unambiguous: the person is still the same person before and after
the blackout, so the pipeline must report a single dwell record that spans the whole
clip instead of two fragments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peopleflow.config import load_config  # noqa: E402
from peopleflow.tracker import PeopleTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an occlusion test clip.")
    parser.add_argument("--source", default="supermarket_browse")
    parser.add_argument("--out", default=str(ROOT / "assets" / "videos" / "occlusion_test.mp4"))
    parser.add_argument("--hide-start", type=float, default=8.0, help="seconds")
    parser.add_argument("--hide-end", type=float, default=13.0, help="seconds")
    parser.add_argument("--margin", type=int, default=24, help="pixels to expand the box")
    args = parser.parse_args()

    source = ROOT / "assets" / "videos" / f"{args.source}.mp4"
    if not source.exists():
        source = Path(args.source)

    cfg = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    tracker = PeopleTracker(cfg)

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    boxes: list[list[int]] = []
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        observations = tracker.update(frame)
        if observations:
            best = max(observations, key=lambda o: o.conf)
            x1, y1, x2, y2 = [int(v) for v in best.xyxy[:4]]
            boxes.append([x1, y1, x2, y2])
        else:
            boxes.append([0, 0, 0, 0])
    cap.release()

    if not boxes or not any(b[2] > 0 for b in boxes):
        print("no person tracked, cannot build the test")
        return 1

    # The person must be visible before the blackout starts, so pick the first frame
    # inside the hidden window and use the box from just before it.
    start_index = int(args.hide_start * fps)
    end_index = int(args.hide_end * fps)
    reference = next((b for b in boxes[:start_index] if b[2] > 0), None)
    if reference is None:
        print("person not visible before the blackout window")
        return 1

    x1 = max(0, reference[0] - args.margin)
    y1 = max(0, reference[1] - args.margin)
    x2 = min(width, reference[2] + args.margin)
    y2 = min(height, reference[3] + args.margin)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    hidden_frames = 0
    for index, frame in enumerate(frames):
        if start_index <= index < end_index:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            hidden_frames += 1
        writer.write(frame)
    writer.release()

    ground_truth = {
        "source": str(source),
        "occluded_video": str(out_path),
        "hidden_from_s": round(start_index / fps, 3),
        "hidden_to_s": round(end_index / fps, 3),
        "hidden_frames": hidden_frames,
        "hidden_box": [x1, y1, x2, y2],
        "fps": fps,
        "total_frames": len(frames),
        "duration_s": round(len(frames) / fps, 3),
        "expected_people": 1,
        "expected_dwell_s": round(len(frames) / fps, 3),
    }
    gt_path = out_path.with_suffix(".json")
    gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    # The person stays in the scene for the whole clip; the blackout only hides them.
    csv_path = out_path.with_suffix(".csv")
    csv_path.write_text(
        "person_id,start_s,end_s\n" f"1,0.0,{ground_truth['duration_s']}\n",
        encoding="utf-8",
    )

    print(f"wrote {out_path}")
    print(f"  hid frames {start_index}..{end_index} ({hidden_frames} frames, {hidden_frames / fps:.1f}s)")
    print(f"  box {x1, y1, x2, y2}")
    print(f"  ground truth -> {gt_path}")
    print(f"  ground truth -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
