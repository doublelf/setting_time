"""Layered configuration for the people-flow / dwell-time pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class SourceConfig:
    kind: str = "file"  # file | rtsp | webcam
    path: str = ""
    frame_stride: int = 2
    max_duration_s: float | None = None
    read_timeout_s: float = 30.0


@dataclass
class DetectorConfig:
    model: str = "yolo26n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cpu"
    classes: list[int] = field(default_factory=lambda: [0])  # COCO person
    half: bool = False
    max_det: int = 100


@dataclass
class TrackerConfig:
    config: str = "configs/trackers/tracktrack_reid.yaml"
    track_buffer: int = 120
    persist: bool = True


@dataclass
class ReIDConfig:
    enabled: bool = True
    model: str = "yolo26n-reid.onnx"
    device: str = "cpu"
    imgsz: int = 320  # crop size fed to the encoder
    refresh_interval: int = 3  # re-encode a tracked person every N frames
    cache_iou: float = 0.70


@dataclass
class RuntimeConfig:
    """CPU thread budget shared by PyTorch and every ONNX Runtime session."""

    intra_op_threads: int = 8
    inter_op_threads: int = 1


@dataclass
class DwellConfig:
    """Tuning knobs for the dwell timer and the cross-track identity merger."""

    merge_window_s: float = 60.0
    merge_threshold: float = 0.80
    review_threshold: float = 0.70
    min_cosine: float = 0.70
    w_appearance: float = 0.90
    w_spatial: float = 0.06
    w_size: float = 0.04
    spatial_tolerance_px: float = 260.0
    max_speed_px_s: float = 120.0
    size_tolerance_ratio: float = 0.45
    border_margin_px: float = 10.0
    exit_speed_px: float = 1.5
    interval_gap_s: float = 0.25
    min_track_frames: int = 3
    min_dwell_s: float = 1.0      # shorter sightings are treated as detection noise


@dataclass
class OutputConfig:
    dir: str = "out"
    overlay: bool = True
    overlay_fps: float | None = None
    write_json: bool = True
    write_events: bool = True
    write_review: bool = True


@dataclass
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dwell: DwellConfig = field(default_factory=DwellConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    root: Path = field(default_factory=Path.cwd)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a plain dict, ignoring unknown keys."""
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in known:
            continue
        target = hints.get(key)
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _build(target, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve(root: Path, value: str) -> str:
    """Resolve a path relative to the project root when it exists there."""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    candidate = root / p
    return str(candidate) if candidate.exists() else value


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Load a YAML config, applying the project-root-relative path resolution."""
    project_root = Path(root) if root else Path(__file__).resolve().parents[2]
    raw: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = _build(Config, raw)
    cfg.root = project_root
    cfg.detector.model = _resolve(project_root, cfg.detector.model)
    cfg.tracker.config = _resolve(project_root, cfg.tracker.config)
    cfg.reid.model = _resolve(project_root, cfg.reid.model)
    cfg.output.dir = _resolve(project_root, cfg.output.dir)
    return cfg


def resolve_asset(name: str, root: Path | None = None) -> str:
    """Resolve a bare Ultralytics asset name (or project-relative path) to a real file."""
    from ultralytics.utils import SETTINGS

    candidate = Path(name)
    if candidate.is_file():
        return str(candidate)
    if root is not None:
        in_project = Path(root) / candidate
        if in_project.is_file():
            return str(in_project)
    in_weights = Path(SETTINGS["weights_dir"]) / candidate.name
    if in_weights.is_file():
        return str(in_weights)
    return name
