"""Ultralytics YOLO detector wrapper (person class only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import DetectorConfig


class Detector:
    """Thin wrapper around a YOLO checkpoint restricted to the person class."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        model_path = cfg.model
        if not Path(model_path).exists():
            # Let Ultralytics resolve/auto-download the official asset name.
            model_path = cfg.model
        self.model = YOLO(model_path)
        self.model.to(cfg.device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run detection and return an (N, 6) array of [x1, y1, x2, y2, conf, cls]."""
        results = self.model.predict(
            image,
            classes=self.cfg.classes or None,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            max_det=self.cfg.max_det,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)[:, None]
        cls = boxes.cls.cpu().numpy().astype(np.float32)[:, None]
        return np.concatenate([xyxy, conf, cls], axis=1)
