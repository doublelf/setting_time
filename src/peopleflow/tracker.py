"""Multi-object tracking adapter around Ultralytics TrackTrack, with ReID embeddings.

The tracker is used in two ways by this project:

* it produces stable track ids over time, and
* it maintains a per-track appearance embedding (``smooth_feat``) that the identity
  merger reuses, so no extra ReID pass is needed on our side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from .config import Config, resolve_asset
from .reid import FastReIDEncoder
from .runtime import configure_process_threads, tune_yolo_model


@dataclass
class Observation:
    """A tracked person in a single frame."""

    track_id: int
    xyxy: np.ndarray
    conf: float
    embedding: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


class PeopleTracker:
    """Run detection + tracking over consecutive frames of a single stream."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        intra = cfg.runtime.intra_op_threads
        inter = cfg.runtime.inter_op_threads
        configure_process_threads(intra)

        model_path = cfg.detector.model
        if not Path(model_path).exists():
            model_path = resolve_asset(Path(model_path).name, cfg.root)
        self.model = YOLO(str(model_path))
        self._detector_path = str(model_path)
        if str(model_path).endswith(".pt"):
            self.model.to(cfg.detector.device)

        self.tracker_config = self._resolve_tracker_config(cfg)
        self._tracker_yaml = yaml.safe_load(Path(self.tracker_config).read_text(encoding="utf-8")) or {}

        # Bound the detector's ONNX thread pool before the first inference.
        if str(model_path).endswith(".onnx"):
            tune_yolo_model(self.model, model_path, intra, inter)

        self._prepare_tracker(intra, inter)
    # -- setup ----------------------------------------------------------------------
    def _resolve_tracker_config(self, cfg: Config) -> str:
        """Write a copy of the tracker YAML with the ReID model path made absolute.

        The tracker YAML is handed to Ultralytics verbatim, and its ``model`` field is
        resolved against the current working directory.  Writing a resolved copy keeps
        the repository config portable while guaranteeing the intended encoder is used.
        """
        source = Path(cfg.tracker.config).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        name = data.get("model")
        if name and name != "auto":
            resolved = resolve_asset(str(name), cfg.root)
            if resolved != str(name) and Path(resolved).is_file():
                data["model"] = str(Path(resolved).resolve())
        target = Path(tempfile.gettempdir()) / f"peopleflow_tracker_{source.stem}.yaml"
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return str(target)

    def _prepare_tracker(self, intra: int, inter: int) -> None:
        """Create the tracker and its ReID encoder, then cap their thread pools."""
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.track(
            source=blank,
            persist=self.cfg.tracker.persist,
            tracker=self.tracker_config,
            classes=self.cfg.detector.classes or None,
            conf=self.cfg.detector.conf,
            verbose=False,
            **self._device_kwarg(),
        )
        self._sync_device()

        # The detector's ONNX session only materialises once the predictor was built.
        tune_yolo_model(self.model, self._detector_path, intra, inter)

        tracker = self._tracker()
        if tracker is None:
            return

        name = self._tracker_yaml.get("model", "auto")
        if not name or name == "auto" or not self._tracker_yaml.get("with_reid", False):
            return

        path = resolve_asset(str(name), self.cfg.root)
        if not Path(path).is_file():
            return

        # Replace the framework encoder: same contract, far cheaper on CPU.
        self.encoder = FastReIDEncoder(
            model_path=str(Path(path).resolve()),
            intra_op_threads=intra,
            inter_op_threads=inter,
            imgsz=self.cfg.reid.imgsz,
            refresh_interval=max(1, self.cfg.reid.refresh_interval),
            cache_iou=self.cfg.reid.cache_iou,
        )
        tracker.encoder = self.encoder

    def _encoder(self):
        """Return the ReID encoder owned by the tracker, if any."""
        tracker = self._tracker()
        return getattr(tracker, "encoder", None) if tracker is not None else None

    # -- internals ------------------------------------------------------------------
    def _tracker(self):
        """Return the live tracker instance owned by the predictor, if available."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return None
        return trackers[0]

    def _collect_embeddings(self) -> dict[int, np.ndarray]:
        """Snapshot the smoothed appearance embedding of every track the tracker keeps."""
        tracker = self._tracker()
        if tracker is None:
            return {}
        embeddings: dict[int, np.ndarray] = {}
        for bucket in ("tracked_stracks", "lost_stracks"):
            for track in getattr(tracker, bucket, []) or []:
                feat = getattr(track, "smooth_feat", None)
                if feat is None:
                    feat = getattr(track, "curr_feat", None)
                if feat is None:
                    continue
                arr = np.asarray(feat, dtype=np.float32).reshape(-1)
                if arr.size and np.linalg.norm(arr) > 1e-12:
                    embeddings[int(track.track_id)] = arr
        return embeddings

    def alive_track_ids(self) -> set[int] | None:
        """Ids the tracker still keeps in memory (tracked + lost buffers)."""
        tracker = self._tracker()
        if tracker is None:
            return None
        ids: set[int] = set()
        for bucket in ("tracked_stracks", "lost_stracks"):
            for track in getattr(tracker, bucket, []) or []:
                ids.add(int(track.track_id))
        return ids

    def _device_kwarg(self) -> dict:
        """Return the ``device`` argument to send with every inference call."""
        device = self.cfg.detector.device
        return {"device": device} if device else {}

    def _sync_device(self) -> None:
        """Keep ``predictor.args.device`` in exactly the form we send it.

        Ultralytics rebuilds the predictor (and with it the tracker and every ONNX
        Runtime session) whenever the requested device differs from the one already
        stored in the predictor's args.  The stored value is a ``torch.device`` while
        the request is a plain string, and for ONNX models it stays ``None``, so the
        comparison would always report a change.  Writing back the value we send makes
        the comparison stable and keeps the predictor alive across frames.
        """
        device = self.cfg.detector.device
        predictor = getattr(self.model, "predictor", None)
        args = getattr(predictor, "args", None)
        if device and args is not None:
            args.device = device

    # -- public API -----------------------------------------------------------------
    def update(self, image: np.ndarray) -> list[Observation]:
        """Advance the tracker by one frame and return the tracked people."""
        results = self.model.track(
            source=image,
            persist=self.cfg.tracker.persist,
            tracker=self.tracker_config,
            classes=self.cfg.detector.classes or None,
            conf=self.cfg.detector.conf,
            iou=self.cfg.detector.iou,
            imgsz=self.cfg.detector.imgsz,
            max_det=self.cfg.detector.max_det,
            verbose=False,
            **self._device_kwarg(),
        )
        self._sync_device()

        result = results[0]
        embeddings = self._collect_embeddings()
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        if not getattr(boxes, "is_track", False) or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        ids = boxes.id.cpu().numpy().astype(np.int64)

        observations = []
        for i, track_id in enumerate(ids):
            observations.append(
                Observation(
                    track_id=int(track_id),
                    xyxy=xyxy[i],
                    conf=float(conf[i]),
                    embedding=embeddings.get(int(track_id)),
                )
            )
        return observations
