"""A fast ReID encoder for CPU inference.

The encoder shipped with Ultralytics builds its crop batch with PyTorch and resizes it
with ``F.interpolate``.  On CPU that costs more than the inference itself, because the
crops are taken from full-resolution frames.

This implementation keeps the exact same semantics (crop the detection box, resize to a
square without letterboxing, convert BGR->RGB and scale to [0, 1]) but does the image
work with OpenCV and talks to ONNX Runtime directly with numpy, so no PyTorch tensor is
involved at all.

An optional cache re-uses the embedding of a detection whose box barely moved, which
saves a full forward pass for people who stand still.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert an (N, 4) xywh array to (N, 4) xyxy."""
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FastReIDEncoder:
    """Callable replacement for ``ultralytics.trackers.utils.reid.ReID``."""

    model_path: str
    intra_op_threads: int = 4
    inter_op_threads: int = 1
    cache_iou: float = 0.70
    cache_max_age: int = 6
    refresh_interval: int = 3
    imgsz: int | None = None  # None -> resolve from the model metadata
    _session: object = field(default=None, init=False, repr=False)
    _input_name: str = field(default="", init=False)
    _cache: list = field(default_factory=list, init=False, repr=False)
    encode_calls: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    refresh_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Create the ONNX Runtime session with a bounded thread pool."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, self.intra_op_threads)
        options.inter_op_num_threads = max(1, self.inter_op_threads)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self._session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.imgsz = self._resolve_imgsz()

    def _resolve_imgsz(self) -> int:
        """Determine the crop size the network was trained for.

        The published ReID encoders use a dynamic spatial input, so the session shape
        does not reveal the trained resolution - it has to come from the model metadata
        (``imgsz: '[448, 448]'``).  Falling back to a smaller size than the one used
        for training silently degrades the embedding quality, so the metadata wins
        unless the caller asked for a specific size.
        """
        if self.imgsz:
            return int(self.imgsz)

        shape = self._session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            return int(shape[2])

        metadata = dict(self._session.get_modelmeta().custom_metadata_map)
        raw = metadata.get("imgsz")
        if raw:
            import ast

            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)) and parsed:
                return int(parsed[0])
            if isinstance(parsed, int):
                return parsed
        return 448

    # -- inference -------------------------------------------------------------------
    def _crop(self, image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Crop a detection out of the frame and resize it to the model input size."""
        h, w = image.shape[:2]
        x1 = int(max(0, min(w - 1, round(box[0]))))
        y1 = int(max(0, min(h - 1, round(box[1]))))
        x2 = int(max(x1 + 1, min(w, round(box[2]))))
        y2 = int(max(y1 + 1, min(h, round(box[3]))))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

    def _encode(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Run the ReID network over a batch of xyxy boxes."""
        crops = []
        for box in boxes:
            crop = self._crop(image, box)
            if crop is None:
                crop = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            crops.append(crop)
        batch = np.stack(crops).astype(np.float32)
        batch = batch[:, :, :, ::-1].transpose(0, 3, 1, 2) / 255.0
        outputs = self._session.run(None, {self._input_name: np.ascontiguousarray(batch)})
        return np.asarray(outputs[0], dtype=np.float32)

    def __call__(self, image: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Return one embedding per detection, reusing cached ones when possible."""
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets[None, :]
        if dets.shape[0] == 0:
            return []
        boxes = _xywh_to_xyxy(dets[:, :4])

        for entry in self._cache:
            entry["age"] += 1

        embeddings: list[np.ndarray | None] = [None] * len(boxes)
        pending: list[int] = []
        for i, box in enumerate(boxes):
            hit = None
            for entry in self._cache:
                if entry["age"] > self.cache_max_age:
                    continue
                if _iou(box, entry["box"]) >= self.cache_iou:
                    hit = entry
                    break
            if hit is not None:
                hit["age"] += 1
                hit["stale"] += 1
                if hit["stale"] >= self.refresh_interval:
                    pending.append(i)
                    hit["stale"] = 0
                    self.refresh_calls += 1
                else:
                    embeddings[i] = hit["embedding"]
                    self.cache_hits += 1
            else:
                pending.append(i)

        if pending:
            self.encode_calls += 1
            feats = self._encode(image, boxes[pending])
            for k, i in enumerate(pending):
                embeddings[i] = feats[k]
                existing = next((e for e in self._cache if e["age"] <= self.cache_max_age and _iou(boxes[i], e["box"]) >= self.cache_iou), None)
                if existing is not None:
                    existing["box"] = boxes[i].copy()
                    existing["embedding"] = feats[k]
                    existing["age"] = 0
                    existing["stale"] = 0
                else:
                    self._cache.append({"box": boxes[i].copy(), "embedding": feats[k], "age": 0, "stale": 0})

        self._cache = [entry for entry in self._cache if entry["age"] <= self.cache_max_age]
        return [e for e in embeddings]  # type: ignore[misc]
