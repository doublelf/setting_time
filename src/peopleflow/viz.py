"""Overlay rendering for the analysed video."""

from __future__ import annotations

import numpy as np

from .dwell import TrackState
from .tracker import Observation

COLORS = [
    (66, 133, 244),
    (234, 67, 53),
    (52, 168, 83),
    (251, 188, 5),
    (171, 71, 188),
    (0, 172, 193),
    (255, 112, 67),
    (118, 183, 87),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def draw_overlay(
    image: np.ndarray,
    observations: list[Observation],
    states: dict[int, TrackState],
    merged_ids: dict[int, int],
    fps: float = 0.0,
) -> np.ndarray:
    """Draw boxes, trails and live dwell timers onto a copy of the frame."""
    canvas = image.copy()

    for obs in observations:
        state = states.get(obs.track_id)
        if state is None:
            continue
        color = _color(obs.track_id)
        x1, y1, x2, y2 = [int(v) for v in obs.xyxy[:4]]
        cv2_rectangle(canvas, (x1, y1), (x2, y2), color)

        trail = state.boxes[-40:]
        if len(trail) > 1:
            points = np.array(
                [[int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)] for b in trail], dtype=np.int32
            )
            cv2_polylines(canvas, points, color)

        label_id = merged_ids.get(obs.track_id, obs.track_id)
        dwell = state.span_s
        text = f"#{label_id} {dwell:.1f}s"
        cv2_text(canvas, text, (x1, max(0, y1 - 6)), color)

    hud = f"{len(observations)} tracked" + (f"  |  {fps:.1f} fps" if fps else "")
    cv2_text(canvas, hud, (10, 24), (255, 255, 255))
    return canvas


def cv2_rectangle(canvas, p1, p2, color) -> None:
    """Draw a rectangle (thin wrapper kept local to avoid a global cv2 import here)."""
    import cv2

    cv2.rectangle(canvas, p1, p2, color, 2)


def cv2_polylines(canvas, points, color) -> None:
    """Draw a polyline (thin wrapper kept local to avoid a global cv2 import here)."""
    import cv2

    cv2.polylines(canvas, [points], isClosed=False, color=color, thickness=2)


def cv2_text(canvas, text, org, color) -> None:
    """Draw filled text with a dark outline (thin wrapper kept local)."""
    import cv2

    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
