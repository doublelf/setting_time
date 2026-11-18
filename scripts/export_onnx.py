"""Export the detector to ONNX so it can run on ONNX Runtime / OpenVINO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the person detector to ONNX.")
    parser.add_argument("--model", default="yolo26n.pt", help="checkpoint to export")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=str(ROOT / "assets" / "weights"))
    args = parser.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"exporting {args.model} at imgsz={args.imgsz} ...")
    produced = model.export(format="onnx", imgsz=args.imgsz, simplify=True, verbose=True)
    print(f"exported -> {produced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
