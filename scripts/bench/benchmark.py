"""End-to-end throughput benchmark of the pipeline.

    python scripts/bench/benchmark.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from peopleflow.config import load_config  # noqa: E402
from peopleflow.pipeline import PeopleFlowPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    videos = args.videos or [
        "assets/videos/supermarket_browse.mp4",
        "assets/videos/lobby_static.mp4",
        "assets/videos/coffee_shop.mp4",
        "assets/videos/mall_busy_timelapse.mp4",
    ]

    cfg = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    cfg.source.frame_stride = args.stride
    cfg.output.overlay = False
    cfg.output.write_json = False
    cfg.output.write_events = False
    cfg.output.write_review = False

    print(f"{'video':30} {'fps':>7} {'ms/frame':>9} {'frames':>7} {'people':>7}")
    print("-" * 64)
    for video in videos:
        path = ROOT / video
        if not path.exists():
            continue
        cfg.source.path = str(path)
        result = PeopleFlowPipeline(cfg).run(out_dir=ROOT / "out" / "_bench", progress=False)
        stats = result.stats
        print(
            f"{path.name:30} {stats['processing_fps']:>7.1f} "
            f"{stats['elapsed_s'] / max(1, stats['frames_processed']) * 1000:>9.1f} "
            f"{stats['frames_processed']:>7} {stats['reported_people']:>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
