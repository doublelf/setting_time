"""End-to-end orchestration: source -> detection -> tracking -> dwell -> merge -> output."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .dwell import DwellTimer, TrackState
from .merger import DwellRecord, IdentityMerger, MergeDecision
from .sinks import write_events_jsonl, write_records_csv, write_records_json, write_review_csv
from .source import VideoSource
from .tracker import PeopleTracker
from .viz import draw_overlay


@dataclass
class RunResult:
    """Everything a run produced."""

    records: list[DwellRecord] = field(default_factory=list)
    decisions: list[MergeDecision] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)


class PeopleFlowPipeline:
    """Analyse one video stream and report the dwell time of every person."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracker = PeopleTracker(cfg)
        self.merger = IdentityMerger(cfg.dwell)
        self.timer = DwellTimer(cfg.dwell)
        self.merged_ids: dict[int, int] = {}
        self.retired: list[TrackState] = []

    # -- online merging -------------------------------------------------------------
    def _try_online_merge(self, state: TrackState) -> MergeDecision | None:
        """Bind a freshly born track to a recently interrupted one, if plausible."""
        best: MergeDecision | None = None
        for old in self.retired:
            if old.exited or old.merged_into is not None:
                continue
            decision = self.merger.score_pair(old, state)
            if decision is None:
                continue
            if best is None or decision.score > best.score:
                best = decision
        if best is not None and best.score >= self.cfg.dwell.merge_threshold:
            root = self.merged_ids.get(best.old_id, best.old_id)
            self.merged_ids[best.old_id] = root
            self.merged_ids[state.track_id] = root
            state.merged_into = root
            best.old_id = root
            return best
        self.merged_ids[state.track_id] = state.track_id
        return None

    def _retire(self) -> None:
        """Move closed tracks into the candidate pool for future merges."""
        for track_id, state in self.timer.tracks.items():
            if state.closed and state.track_id not in {s.track_id for s in self.retired}:
                self.retired.append(state)

    # -- main loop ------------------------------------------------------------------
    def run(self, out_dir: str | Path | None = None, progress: bool = True) -> RunResult:
        """Process the configured source and write all outputs."""
        out_dir = Path(out_dir or self.cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        source = VideoSource(self.cfg.source)
        overlay_writer = None
        overlay_path = out_dir / "overlay.mp4"
        frame_interval = 1.0 / source.effective_fps if source.effective_fps > 0 else 1 / 15
        self.timer.frame_interval = frame_interval

        processed = 0
        total = len(source) or 0
        start = time.perf_counter()
        last_log = start
        last_timestamp = 0.0

        try:
            for frame in source:
                observations = self.tracker.update(frame.image)
                alive = self.tracker.alive_track_ids()

                known = set(self.timer.tracks)
                self.timer.update(observations, frame.timestamp, frame.image.shape[:2], alive)

                for obs in observations:
                    if obs.track_id in known:
                        continue
                    state = self.timer.tracks.get(obs.track_id)
                    if state is None:
                        continue
                    decision = self._try_online_merge(state)
                    if decision is not None:
                        self.timer.log_event(
                            "track_merged",
                            frame.timestamp,
                            obs.track_id,
                            into=decision.new_id,
                            score=round(decision.score, 4),
                        )
                self._retire()
                last_timestamp = frame.timestamp

                if self.cfg.output.overlay:
                    overlay = draw_overlay(
                        frame.image,
                        observations,
                        self.timer.tracks,
                        self.merged_ids,
                        fps=0.0,
                    )
                    if overlay_writer is None:
                        fps = self.cfg.output.overlay_fps or source.effective_fps or 25.0
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        overlay_writer = cv2.VideoWriter(
                            str(overlay_path), fourcc, float(fps), (frame.image.shape[1], frame.image.shape[0])
                        )
                    overlay_writer.write(overlay)

                processed += 1
                if progress and time.perf_counter() - last_log > 2.0:
                    last_log = time.perf_counter()
                    print(f"  ... {processed} frames, {len(self.timer.tracks)} raw tracks", flush=True)
        finally:
            source.release()
            if overlay_writer is not None:
                overlay_writer.release()

        elapsed = time.perf_counter() - start
        self.timer.finalize(last_timestamp)
        self._retire()

        states = self.timer.visible_states()
        records, accepted = self.merger.merge(states)

        # Record the merges that the global pass performed, so the event log tells the
        # whole story of how a visit was assembled.
        for decision in accepted:
            old_state = self.timer.tracks.get(decision.old_id)
            self.timer.log_event(
                "track_merged",
                old_state.last_seen if old_state else decision.old_id,
                decision.new_id,
                into=decision.old_id,
                score=round(decision.score, 4),
                cosine=round(decision.cosine, 4),
                gap_s=round(decision.gap_s, 3),
            )

        # A sighting that lasts only a fraction of a second is detection noise, not a
        # visit.  Those are dropped from the report but kept in the event log.
        min_dwell = self.cfg.dwell.min_dwell_s
        if min_dwell > 0:
            noise = [r for r in records if r.dwell_span_s < min_dwell]
            records = [r for r in records if r.dwell_span_s >= min_dwell]
            for record in noise:
                self.timer.log_event(
                    "dropped_short_sighting",
                    record.last_seen,
                    record.merged_id,
                    dwell_s=round(record.dwell_span_s, 3),
                )

        result = RunResult(records=records, decisions=self.merger.decisions, events=self.timer.events)
        result.stats = {
            "source": str(self.cfg.source.path),
            "frames_processed": processed,
            "source_fps": round(source.fps, 2),
            "frame_stride": source.stride,
            "effective_fps": round(source.effective_fps, 2),
            "elapsed_s": round(elapsed, 2),
            "processing_fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
            "raw_tracks": len(self.timer.tracks),
            "reported_people": len(records),
            "dropped_short": len(self.timer.tracks) - len(records),
            "merged_fragments": sum(len(r.track_ids) - 1 for r in records),
            "merged_pairs": len(accepted),
        }

        paths: dict[str, str] = {}
        paths["records_csv"] = str(write_records_csv(records, out_dir / "dwell_records.csv"))
        if self.cfg.output.write_json:
            paths["records_json"] = str(write_records_json(records, out_dir / "dwell_records.json", result.stats))
        if self.cfg.output.write_events:
            paths["events_jsonl"] = str(write_events_jsonl(self.timer.events, out_dir / "events.jsonl"))
        if self.cfg.output.write_review:
            review = write_review_csv(records, out_dir / "dwell_review.csv")
            if review:
                paths["review_csv"] = str(review)
        if overlay_writer is not None:
            paths["overlay"] = str(overlay_path)
        result.output_paths = paths
        return result
