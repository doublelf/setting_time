"""Command line entry point: analyse a video file or a live stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peopleflow.config import load_config  # noqa: E402
from peopleflow.pipeline import PeopleFlowPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect people in a video and report each person's dwell time.",
    )
    parser.add_argument("--source", help="video file, RTSP url or camera index")
    parser.add_argument(
        "--kind",
        choices=["file", "rtsp", "webcam"],
        default=None,
        help="how to interpret --source (inferred from the source when omitted)",
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML config file")
    parser.add_argument("--out", help="output directory (default: from config)")
    parser.add_argument("--model", help="detector checkpoint override")
    parser.add_argument("--tracker", help="tracker YAML override")
    parser.add_argument("--frame-stride", type=int, help="process every N-th frame")
    parser.add_argument("--max-duration", type=float, help="stop after N seconds of source time")
    parser.add_argument("--conf", type=float, help="detection confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, help="detector input size")
    parser.add_argument("--device", help="torch device, e.g. cpu or 0")
    parser.add_argument("--merge-threshold", type=float, help="acceptance score of a cross-track merge")
    parser.add_argument("--merge-window", type=float, help="seconds a person may be invisible and still match")
    parser.add_argument("--min-cosine", type=float, help="hard floor on appearance similarity")
    parser.add_argument("--no-overlay", action="store_true", help="skip rendering the overlay video")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--top", type=int, default=10, help="how many rows of the summary to print")
    return parser


def apply_overrides(cfg, args: argparse.Namespace) -> None:
    """Apply CLI overrides on top of the YAML configuration."""
    if args.source:
        cfg.source.path = args.source
        if args.kind is None:
            # Infer the source type: an rtsp:// / http:// url is a stream, a bare
            # integer is a camera index, anything else is a file.
            text = str(args.source)
            if text.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                cfg.source.kind = "rtsp"
            elif text.isdigit():
                cfg.source.kind = "webcam"
            else:
                cfg.source.kind = "file"
    if args.kind:
        cfg.source.kind = args.kind
    if args.frame_stride:
        cfg.source.frame_stride = args.frame_stride
    if args.max_duration:
        cfg.source.max_duration_s = args.max_duration
    if args.model:
        cfg.detector.model = args.model
    if args.tracker:
        cfg.tracker.config = str(Path(args.tracker).resolve())
    if args.conf is not None:
        cfg.detector.conf = args.conf
    if args.iou is not None:
        cfg.detector.iou = args.iou
    if args.imgsz:
        cfg.detector.imgsz = args.imgsz
    if args.device:
        cfg.detector.device = args.device
    if args.merge_threshold is not None:
        cfg.dwell.merge_threshold = args.merge_threshold
    if args.merge_window is not None:
        cfg.dwell.merge_window_s = args.merge_window
    if args.min_cosine is not None:
        cfg.dwell.min_cosine = args.min_cosine
    if args.no_overlay:
        cfg.output.overlay = False
    if args.out:
        cfg.output.dir = str(Path(args.out).resolve())


def print_summary(result, top: int) -> None:
    """Print the run statistics and the longest dwells."""
    stats = result.stats
    print("\n=== run statistics ===")
    for key in (
        "frames_processed",
        "source_fps",
        "frame_stride",
        "effective_fps",
        "elapsed_s",
        "processing_fps",
        "raw_tracks",
        "reported_people",
        "merged_fragments",
        "merged_pairs",
    ):
        print(f"  {key:18} {stats.get(key)}")

    print("\n=== dwell records (longest first) ===")
    header = f"  {'id':>5} {'tracks':>16} {'span(s)':>9} {'visible(s)':>10} {'occl(s)':>8} {'frames':>7} {'conf':>6} exit"
    print(header)
    for record in sorted(result.records, key=lambda r: -r.dwell_span_s)[:top]:
        tracks = ",".join(str(t) for t in record.track_ids)
        flag = " *" if record.needs_review else ""
        print(
            f"  {record.merged_id:>5} {tracks:>16} {record.dwell_span_s:>9.1f} {record.dwell_visible_s:>10.1f} "
            f"{record.occlusion_s:>8.1f} {record.n_frames:>7} {record.merge_confidence:>6.2f} {record.exit_type}{flag}"
        )
    print("\n  (* = merge below the review threshold, listed in dwell_review.csv)")

    print("\n=== outputs ===")
    for key, value in result.output_paths.items():
        print(f"  {key:14} {value}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, root=ROOT)
    apply_overrides(cfg, args)

    if not cfg.source.path:
        print("error: --source is required", file=sys.stderr)
        return 2

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source : {cfg.source.path}  (stride={cfg.source.frame_stride}, kind={cfg.source.kind})")
    print(f"model  : {cfg.detector.model}  device={cfg.detector.device}  conf={cfg.detector.conf}")
    print(f"tracker: {cfg.tracker.config}")
    print(f"out    : {out_dir}")

    pipeline = PeopleFlowPipeline(cfg)
    result = pipeline.run(out_dir=out_dir, progress=not args.quiet)
    print_summary(result, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
