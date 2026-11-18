"""Unit tests for the parts that do not need a model: timing, geometry and merging."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peopleflow.config import DwellConfig  # noqa: E402
from peopleflow.dwell import DwellTimer  # noqa: E402
from peopleflow.geometry import cosine_similarity, merge_intervals, scale_score  # noqa: E402
from peopleflow.merger import IdentityMerger  # noqa: E402
from peopleflow.tracker import Observation  # noqa: E402


# -- geometry ---------------------------------------------------------------------
def test_merge_intervals_unions_overlapping_ranges():
    merged = merge_intervals([(0.0, 2.0), (1.0, 3.0), (5.0, 6.0)])
    assert merged == [(0.0, 3.0), (5.0, 6.0)]


def test_merge_intervals_bridges_small_gaps():
    merged = merge_intervals([(0.0, 1.0), (1.2, 2.0)], gap_tol=0.3)
    assert merged == [(0.0, 2.0)]


def test_cosine_similarity_is_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a * 7.0) == pytest.approx(1.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_cosine_similarity_handles_missing_embeddings():
    assert cosine_similarity(None, np.ones(4)) is None
    assert cosine_similarity(np.zeros(4), np.ones(4)) is None


def test_scale_score_decays_to_zero_at_tolerance():
    assert scale_score(0.0, 100.0) == 1.0
    assert scale_score(100.0, 100.0) == 0.0
    assert scale_score(150.0, 100.0) == 0.0


# -- dwell timer ------------------------------------------------------------------
def _observation(track_id: int, box, embedding=None) -> Observation:
    return Observation(
        track_id=track_id,
        xyxy=np.array(box, dtype=np.float32),
        conf=0.9,
        embedding=embedding,
    )


def test_timer_accumulates_visible_time():
    timer = DwellTimer(DwellConfig())
    for step in range(5):
        timer.update([_observation(1, (0, 0, 10, 10))], timestamp=step * 0.1, frame_shape=(100, 100))
    state = timer.tracks[1]
    assert state.n_obs == 5
    assert state.visible_s == pytest.approx(0.4)
    assert state.span_s == pytest.approx(0.4)


def test_timer_marks_exit_when_leaving_through_the_border():
    timer = DwellTimer(DwellConfig())
    timer.update([_observation(1, (90, 40, 120, 90))], timestamp=0.0, frame_shape=(100, 100))
    timer.update([_observation(1, (95, 40, 130, 90))], timestamp=0.1, frame_shape=(100, 100))
    timer.update([], timestamp=0.2, frame_shape=(100, 100), alive_ids=set())
    state = timer.tracks[1]
    assert state.exited is True
    assert state.exit_reason.startswith("exit_")


def test_timer_does_not_mark_exit_for_a_person_standing_at_the_border():
    timer = DwellTimer(DwellConfig())
    timer.update([_observation(1, (0, 40, 20, 90))], timestamp=0.0, frame_shape=(100, 100))
    timer.update([_observation(1, (0, 40, 20, 90))], timestamp=0.1, frame_shape=(100, 100))
    timer.update([], timestamp=0.2, frame_shape=(100, 100), alive_ids=set())
    state = timer.tracks[1]
    assert state.exited is False


# -- identity merger ---------------------------------------------------------------
def _track(track_id, first, last, embedding, boxes=None, exited=False):
    from peopleflow.dwell import TrackState

    state = TrackState(track_id=track_id, first_seen=first, last_seen=last)
    state.intervals = [[first, last]]
    state.n_obs = 10
    state.heights = [100.0]
    state.widths = [50.0]
    state.embedding = np.asarray(embedding, dtype=np.float32)
    state.boxes = boxes or [[0.0, 0.0, 50.0, 100.0]]
    state.exited = exited
    return state


def test_merger_joins_two_fragments_of_the_same_person():
    cfg = DwellConfig()
    merger = IdentityMerger(cfg)
    a = _track(1, 0.0, 5.0, [1.0, 0.0, 0.0], boxes=[[0.0, 0.0, 50.0, 100.0]])
    b = _track(2, 12.0, 20.0, [0.98, 0.1, 0.0], boxes=[[120.0, 0.0, 170.0, 100.0]])

    records, accepted = merger.merge([a, b])
    assert len(accepted) == 1
    assert len(records) == 1
    assert records[0].track_ids == [1, 2]
    assert records[0].dwell_span_s == pytest.approx(20.0)


def test_merger_keeps_different_people_apart():
    cfg = DwellConfig()
    merger = IdentityMerger(cfg)
    a = _track(1, 0.0, 5.0, [1.0, 0.0, 0.0], boxes=[[0.0, 0.0, 50.0, 100.0]])
    b = _track(2, 12.0, 20.0, [0.0, 1.0, 0.0], boxes=[[120.0, 0.0, 170.0, 100.0]])

    records, accepted = merger.merge([a, b])
    assert accepted == []
    assert len(records) == 2


def test_merger_never_joins_a_person_who_left_the_scene():
    cfg = DwellConfig()
    merger = IdentityMerger(cfg)
    a = _track(1, 0.0, 5.0, [1.0, 0.0, 0.0], boxes=[[0.0, 0.0, 50.0, 100.0]], exited=True)
    b = _track(2, 12.0, 20.0, [0.99, 0.1, 0.0], boxes=[[120.0, 0.0, 170.0, 100.0]])

    records, accepted = merger.merge([a, b])
    assert accepted == []
    assert len(records) == 2


def test_merger_rejects_pairs_outside_the_time_window():
    cfg = DwellConfig(merge_window_s=5.0)
    merger = IdentityMerger(cfg)
    a = _track(1, 0.0, 5.0, [1.0, 0.0, 0.0])
    b = _track(2, 30.0, 40.0, [0.99, 0.1, 0.0])

    _, accepted = merger.merge([a, b])
    assert accepted == []


def test_merger_uses_one_record_per_track():
    cfg = DwellConfig()
    merger = IdentityMerger(cfg)
    a = _track(1, 0.0, 5.0, [1.0, 0.0, 0.0])
    records, accepted = merger.merge([a])
    assert len(records) == 1
    assert records[0].track_ids == [1]
    assert accepted == []
