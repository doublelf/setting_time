"""Per-track dwell timing.

The timer keeps a life-cycle record for every tracker id: when it first appeared, when
it was last seen, the intervals in which it was actually visible, and whether it left the
scene or merely disappeared behind an occluder.

The exit/occlusion distinction is the foundation of the identity merger: a person who
walks out of frame and comes back is *two* business visits and must never be merged,
while a person hidden behind a shelf for a few seconds is *one* visit and must be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DwellConfig
from .geometry import bbox_center, bbox_height, bbox_width
from .tracker import Observation


@dataclass
class TrackState:
    """Life-cycle record of a single tracker id."""

    track_id: int
    first_seen: float
    last_seen: float
    intervals: list[list[float]] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    n_obs: int = 0
    embedding: np.ndarray | None = None
    exited: bool = False
    exit_reason: str = ""
    closed: bool = False
    merged_into: int | None = None

    @property
    def visible_s(self) -> float:
        """Total time the person was actually detected."""
        return float(sum(e - s for s, e in self.intervals if e > s))

    @property
    def span_s(self) -> float:
        """Total time from first to last observation, gaps included."""
        return float(max(0.0, self.last_seen - self.first_seen))

    @property
    def last_box(self) -> list[float] | None:
        return self.boxes[-1] if self.boxes else None

    @property
    def last_center(self) -> tuple[float, float] | None:
        return bbox_center(self.boxes[-1]) if self.boxes else None

    @property
    def first_center(self) -> tuple[float, float] | None:
        return bbox_center(self.boxes[0]) if self.boxes else None

    @property
    def max_gap_s(self) -> float:
        """Longest invisible stretch inside the track's life span."""
        if len(self.intervals) < 2:
            return 0.0
        gaps = [self.intervals[i + 1][0] - self.intervals[i][1] for i in range(len(self.intervals) - 1)]
        return float(max(gaps)) if gaps else 0.0


class DwellTimer:
    """Accumulate dwell intervals for every tracked person."""

    def __init__(self, cfg: DwellConfig, frame_interval: float = 1 / 15):
        self.cfg = cfg
        self.tracks: dict[int, TrackState] = {}
        self.frame_interval = frame_interval
        self.frame_shape: tuple[int, int] | None = None
        self.events: list[dict] = []

    # -- helpers --------------------------------------------------------------------
    def _log(self, kind: str, timestamp: float, track_id: int, **payload) -> None:
        self.events.append({"event": kind, "t": round(timestamp, 3), "track_id": track_id, **payload})

    def log_event(self, kind: str, timestamp: float, track_id: int, **payload) -> None:
        """Record an externally generated event (e.g. a merge decision)."""
        self._log(kind, timestamp, track_id, **payload)

    def _touches_border(self, box: list[float]) -> tuple[bool, str | None]:
        """Whether a box touches an image border, and which one."""
        if self.frame_shape is None:
            return False, None
        h, w = self.frame_shape
        m = self.cfg.border_margin_px
        x1, y1, x2, y2 = box[:4]
        if x1 <= m:
            return True, "left"
        if x2 >= w - m:
            return True, "right"
        if y1 <= m:
            return True, "top"
        if y2 >= h - m:
            return True, "bottom"
        return False, None

    def _judge_exit(self, state: TrackState) -> None:
        """Decide whether a closed track left the scene or was merely occluded."""
        if state.boxes:
            touching, side = self._touches_border(state.boxes[-1])
            if touching:
                if len(state.boxes) >= 2:
                    prev = np.array(bbox_center(state.boxes[-2]))
                    last = np.array(bbox_center(state.boxes[-1]))
                    vel = last - prev
                    outward = {
                        "left": vel[0] < -self.cfg.exit_speed_px,
                        "right": vel[0] > self.cfg.exit_speed_px,
                        "top": vel[1] < -self.cfg.exit_speed_px,
                        "bottom": vel[1] > self.cfg.exit_speed_px,
                    }[side]
                    if outward:
                        state.exited = True
                        state.exit_reason = f"exit_{side}"
                        return
                    state.exit_reason = f"touch_{side}"
                    return
                state.exited = True
                state.exit_reason = f"exit_{side}"
                return
        state.exit_reason = "occluded" if state.n_obs else "empty"

    # -- public API -----------------------------------------------------------------
    def update(
        self,
        observations: list[Observation],
        timestamp: float,
        frame_shape: tuple[int, int],
        alive_ids: set[int] | None = None,
    ) -> None:
        """Fold one frame of observations into the timer state."""
        self.frame_shape = frame_shape
        seen: set[int] = set()

        for obs in observations:
            seen.add(obs.track_id)
            state = self.tracks.get(obs.track_id)
            if state is None:
                state = TrackState(track_id=obs.track_id, first_seen=timestamp, last_seen=timestamp)
                self.tracks[obs.track_id] = state
                self._log("track_created", timestamp, obs.track_id, conf=round(obs.conf, 3))

            state.last_seen = timestamp
            state.n_obs += 1
            state.boxes.append([float(v) for v in obs.xyxy[:4]])
            state.heights.append(bbox_height(obs.xyxy))
            state.widths.append(bbox_width(obs.xyxy))
            if obs.embedding is not None:
                state.embedding = obs.embedding

            if state.intervals and timestamp - state.intervals[-1][1] <= self.cfg.interval_gap_s:
                state.intervals[-1][1] = timestamp
            else:
                state.intervals.append([timestamp, timestamp])

        # Close tracks the tracker has dropped (or that timed out).
        close_after = max(self.frame_interval * 3.0, self.cfg.interval_gap_s * 2.0)
        for track_id, state in self.tracks.items():
            if state.closed or track_id in seen:
                continue
            if alive_ids is not None:
                if track_id in alive_ids:
                    continue
            elif timestamp - state.last_seen > close_after:
                continue
            state.closed = True
            self._judge_exit(state)
            self._log("track_closed", timestamp, track_id, reason=state.exit_reason)

    def finalize(self, end_time: float) -> None:
        """Close every still-open track at the end of the stream."""
        for track_id, state in self.tracks.items():
            if state.closed:
                continue
            state.closed = True
            self._judge_exit(state)
            self._log("track_closed", end_time, track_id, reason=state.exit_reason)

    def visible_states(self) -> list[TrackState]:
        """Tracks that carry enough evidence to be reported."""
        return [s for s in self.tracks.values() if s.n_obs >= self.cfg.min_track_frames]
