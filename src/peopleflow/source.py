"""Video input: files, RTSP streams and webcams, with frame sampling and real timestamps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from .config import SourceConfig


@dataclass
class Frame:
    """A single sampled frame with its timeline position."""

    index: int  # index in the source stream (not in the sampled sequence)
    seq: int  # index in the sampled sequence
    timestamp: float  # seconds on the source timeline
    image: "cv2.MatLike"


def _open_capture(cfg: SourceConfig) -> cv2.VideoCapture:
    """Open a capture handle for the configured source."""
    if cfg.kind == "webcam":
        source = int(cfg.path) if str(cfg.path).isdigit() else 0
        cap = cv2.VideoCapture(source)
    else:
        if not Path(cfg.path).exists():
            raise FileNotFoundError(f"source not found: {cfg.path}")
        cap = cv2.VideoCapture(cfg.path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {cfg.path}")
    return cap


class VideoSource:
    """Iterate over sampled frames of a video file or live stream.

    Timestamps always come from the source timeline (video PTS for files, wall clock
    for live sources) so that dwell durations stay correct even when frames are skipped.
    """

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.cap = _open_capture(cfg)
        self.is_live = cfg.kind in ("rtsp", "webcam")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not self.is_live and (self.fps <= 0 or self.fps > 240):
            self.fps = 25.0  # unreliable container metadata fallback
        self.stride = max(1, int(cfg.frame_stride))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._live_origin = time.perf_counter()
        self._frame_index = 0
        self._seq = 0

    def __len__(self) -> int:
        if self.is_live or self.total_frames <= 0:
            return 0
        return self.total_frames // self.stride + 1

    @property
    def effective_fps(self) -> float:
        """Frame rate of the *sampled* sequence (what the tracker actually sees)."""
        return self.fps / self.stride if self.fps > 0 else 0.0

    def _timestamp(self, index: int) -> float:
        """Resolve the timeline position of a source frame."""
        if self.is_live:
            return time.perf_counter() - self._live_origin
        if self.fps > 0:
            return index / self.fps
        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return pos_msec / 1000.0 if pos_msec > 0 else index / 25.0

    def __iter__(self) -> Iterator[Frame]:
        """Yield sampled frames until the source is exhausted."""
        deadline = None
        if self.cfg.max_duration_s:
            deadline = time.perf_counter() + self.cfg.max_duration_s

        while True:
            if deadline is not None and time.perf_counter() > deadline:
                break

            index = self._frame_index
            if index % self.stride == 0:
                ok, image = self.cap.read()
                if not ok:
                    break
                if self.frame_width <= 0:
                    self.frame_height, self.frame_width = image.shape[:2]
                timestamp = self._timestamp(index)
                frame = Frame(index=index, seq=self._seq, timestamp=timestamp, image=image)
                self._seq += 1
                yield frame
            else:
                if not self.cap.grab():
                    break
            self._frame_index += 1

    def release(self) -> None:
        """Release the underlying capture handle."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
