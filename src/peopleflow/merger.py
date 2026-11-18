"""Cross-track identity merging.

The tracker keeps an id alive for `track_buffer` frames and can rebind through a relaxed
second association pass, but a longer occlusion always breaks a track in two.  This
module repairs those breaks: it scores every plausible (old track -> new track) pairing
and greedily accepts the strongest ones, so that the dwell time of one person is
reported once instead of in fragments.

Scoring combines three independent signals:

* appearance  - cosine similarity of the tracker's smoothed ReID embeddings,
* spatial     - how plausible the new position is given where the old track vanished,
* size        - consistency of the body size across the break.

Tracks that clearly left the scene (their last box touched a frame border while moving
outwards) are never merged: walking out and coming back is a new visit, not a
continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DwellConfig
from .dwell import TrackState
from .geometry import (
    bbox_center,
    cosine_similarity,
    intervals_total,
    merge_intervals,
    point_distance,
    scale_score,
)


@dataclass
class MergeDecision:
    """An accepted (or rejected) merge between two tracker ids."""

    old_id: int
    new_id: int
    score: float
    cosine: float
    spatial: float
    size: float
    gap_s: float
    accepted: bool = True


@dataclass
class DwellRecord:
    """The final per-person dwell summary."""

    merged_id: int
    track_ids: list[int]
    first_seen: float
    last_seen: float
    dwell_span_s: float
    dwell_visible_s: float
    occlusion_s: float
    n_frames: int
    max_gap_s: float
    exit_type: str
    merge_confidence: float
    mean_height: float
    needs_review: bool = False
    components: dict = field(default_factory=dict)


def _appearance_score(cosine: float, min_cosine: float) -> float:
    """Map a cosine similarity to a [0, 1] score that is zero at the rejection floor.

    A plain rescale of the cosine would let a moderately similar pair (say 0.65, which
    is still within the range two different people produce) reach a high score once
    the spatial and size terms are added.  Rescaling from the rejection floor upwards
    keeps the appearance evidence dominant: only a clearly matching appearance can
    carry a merge on its own.
    """
    span = max(1e-6, 1.0 - min_cosine)
    return float(max(0.0, min(1.0, (cosine - min_cosine) / span)))


class _UnionFind:
    """Minimal union-find used to group merged tracks."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class IdentityMerger:
    """Merge track fragments that belong to the same person."""

    def __init__(self, cfg: DwellConfig):
        self.cfg = cfg
        self.decisions: list[MergeDecision] = []

    # -- scoring --------------------------------------------------------------------
    def _median_height(self, state: TrackState) -> float:
        return float(np.median(state.heights)) if state.heights else 0.0

    def score_pair(self, old: TrackState, new: TrackState) -> MergeDecision | None:
        """Score a candidate pairing, or return None when it is not admissible.

        The appearance evidence carries the decision: on real footage the embedding of
        one person stays far above the range two different people produce, so it is
        the primary term.  Geometry acts as a plausibility gate on top of it - a
        matching-looking person who is on the other side of the scene is not a match.
        """
        gap = new.first_seen - old.last_seen
        if gap < -1e-6 or gap > self.cfg.merge_window_s:
            return None
        if old.exited:
            return None

        cosine = cosine_similarity(old.embedding, new.embedding)
        if cosine is None or cosine < self.cfg.min_cosine:
            return None

        old_center = old.last_center
        new_center = new.first_center
        if old_center is None or new_center is None:
            return None
        distance = point_distance(old_center, new_center)

        # A person covers ground while they are out of sight, so the plausible
        # distance grows with the length of the gap.
        tolerance = max(
            self.cfg.spatial_tolerance_px,
            gap * self.cfg.max_speed_px_s,
        )
        if distance > tolerance:
            return None
        spatial = scale_score(distance, tolerance)

        h_old, h_new = self._median_height(old), self._median_height(new)
        if h_old <= 0 or h_new <= 0:
            size = 0.0
        else:
            ratio = abs(h_old - h_new) / max(h_old, h_new)
            size = scale_score(ratio, self.cfg.size_tolerance_ratio)

        score = (
            self.cfg.w_appearance * cosine
            + self.cfg.w_spatial * spatial
            + self.cfg.w_size * size
        )
        return MergeDecision(
            old_id=old.track_id,
            new_id=new.track_id,
            score=float(score),
            cosine=float(cosine),
            spatial=float(spatial),
            size=float(size),
            gap_s=float(gap),
        )

    # -- public API -----------------------------------------------------------------
    def merge(self, states: list[TrackState]) -> tuple[list[DwellRecord], list[MergeDecision]]:
        """Group track fragments into per-person records."""
        ordered = sorted(states, key=lambda s: (s.first_seen, s.track_id))
        candidates: list[MergeDecision] = []
        for i, old in enumerate(ordered):
            for new in ordered[i + 1 :]:
                decision = self.score_pair(old, new)
                if decision is not None:
                    candidates.append(decision)

        candidates.sort(key=lambda d: d.score, reverse=True)

        used_as_old: set[int] = set()
        used_as_new: set[int] = set()
        accepted: list[MergeDecision] = []
        for decision in candidates:
            if decision.old_id in used_as_old or decision.new_id in used_as_new:
                decision.accepted = False
                continue
            if decision.score < self.cfg.merge_threshold:
                decision.accepted = False
                break
            accepted.append(decision)
            used_as_old.add(decision.old_id)
            used_as_new.add(decision.new_id)

        self.decisions = candidates

        union = _UnionFind()
        for state in ordered:
            union.find(state.track_id)
        for decision in accepted:
            union.union(decision.old_id, decision.new_id)

        groups: dict[int, list[TrackState]] = {}
        for state in ordered:
            groups.setdefault(union.find(state.track_id), []).append(state)

        accepted_by_new = {d.new_id: d for d in accepted}
        records = [self._build_record(members, accepted_by_new) for members in groups.values()]
        records.sort(key=lambda r: r.first_seen)
        return records, accepted

    def _build_record(self, members: list[TrackState], accepted_by_new: dict[int, MergeDecision]) -> DwellRecord:
        """Collapse a group of track fragments into one dwell record."""
        members = sorted(members, key=lambda s: s.first_seen)
        intervals: list[tuple[float, float]] = []
        for member in members:
            intervals.extend((s, e) for s, e in member.intervals)
        merged = merge_intervals(intervals)

        first_seen = min(m.first_seen for m in members)
        last_seen = max(m.last_seen for m in members)
        span = max(0.0, last_seen - first_seen)
        visible = intervals_total(merged)

        confidences = [accepted_by_new[m.track_id].score for m in members if m.track_id in accepted_by_new]
        confidence = min(confidences) if confidences else 1.0

        heights = [h for m in members for h in m.heights]
        mean_height = float(np.median(heights)) if heights else 0.0

        max_gap = 0.0
        for i in range(len(merged) - 1):
            max_gap = max(max_gap, merged[i + 1][0] - merged[i][1])

        exit_type = members[-1].exit_reason or "unknown"
        components = {}
        if len(members) > 1:
            decision = accepted_by_new.get(members[-1].track_id)
            if decision:
                components = {
                    "cosine": round(decision.cosine, 4),
                    "spatial": round(decision.spatial, 4),
                    "size": round(decision.size, 4),
                    "gap_s": round(decision.gap_s, 3),
                }

        return DwellRecord(
            merged_id=members[0].track_id,
            track_ids=[m.track_id for m in members],
            first_seen=first_seen,
            last_seen=last_seen,
            dwell_span_s=span,
            dwell_visible_s=visible,
            occlusion_s=max(0.0, span - visible),
            n_frames=sum(m.n_obs for m in members),
            max_gap_s=max_gap,
            exit_type=exit_type,
            merge_confidence=float(confidence),
            mean_height=mean_height,
            needs_review=len(members) > 1 and confidence < self.cfg.review_threshold,
            components=components,
        )
