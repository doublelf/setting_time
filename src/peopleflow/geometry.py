"""Small geometry / numeric helpers shared by the dwell timer and identity merger."""

from __future__ import annotations

import numpy as np

Interval = tuple[float, float]


def bbox_center(box: np.ndarray | list[float]) -> tuple[float, float]:
    """Return the (x, y) centre of an xyxy box."""
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_height(box: np.ndarray | list[float]) -> float:
    """Return the height of an xyxy box."""
    return float(box[3] - box[1])


def bbox_width(box: np.ndarray | list[float]) -> float:
    """Return the width of an xyxy box."""
    return float(box[2] - box[0])


def l2_normalize(vec: np.ndarray | None) -> np.ndarray | None:
    """L2-normalise a vector, returning None for missing / zero-norm input."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return arr / norm


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine similarity of two embeddings; None when either side is missing."""
    na, nb = l2_normalize(a), l2_normalize(b)
    if na is None or nb is None:
        return None
    return float(np.dot(na, nb))


def merge_intervals(intervals: list[Interval], gap_tol: float = 0.0) -> list[Interval]:
    """Union a list of (start, end) intervals, optionally bridging gaps <= gap_tol."""
    if not intervals:
        return []
    ordered = sorted((float(s), float(e)) for s, e in intervals if e >= s)
    merged: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + gap_tol:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def intervals_total(intervals: list[Interval]) -> float:
    """Total covered duration of a list of intervals (assumed disjoint or merged)."""
    return float(sum(e - s for s, e in intervals if e > s))


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def scale_score(value: float, tolerance: float) -> float:
    """Map a magnitude to a [0, 1] score that decays linearly to zero at `tolerance`."""
    if tolerance <= 0:
        return 0.0
    return float(max(0.0, 1.0 - value / tolerance))
