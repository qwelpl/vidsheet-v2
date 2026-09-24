"""Temporal note tracking and sub-frame timing (§9, §10, §11, §12, §53, §54).

Runs from :mod:`detect` are linked across frames into persistent falling-bar
tracks. Each track's leading-edge trajectory is fit to a line so the exact
strike time is recovered *between* video frames (temporal super-resolution,
§53) — timing precision far better than one frame. Duration comes from the bar
length divided by fall speed (§11); repeated notes stay distinct because they
arrive as separate tracks with a visible gap (§12).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detect import FrameObservation, LaneRun
from .keyboard import KeyboardGeometry
from .model import NoteEvent, Hand


@dataclass
class _Sample:
    t: float
    frame: int
    y_top: float
    y_bottom: float
    val: float
    sat: float
    cluster: int
    top_clamped: bool
    bottom_clamped: bool


@dataclass
class Track:
    id: int
    midi: int
    samples: list[_Sample] = field(default_factory=list)
    missing: int = 0

    @property
    def last(self) -> _Sample:
        return self.samples[-1]


class Tracker:
    """Links per-lane runs frame-to-frame into :class:`Track` objects."""

    def __init__(self, geom: KeyboardGeometry, fps: float):
        self.geom = geom
        self.fps = fps
        self.strike_y = float(geom.strike_y)
        self.roll_h = float(geom.strike_y)
        self._tracks: dict[int, list[Track]] = {}   # midi -> active tracks
        self._closed: list[Track] = []
        self._next_id = 1
        self._v_est = self.strike_y * 0.5  # bootstrap fall speed px/s (refined)
        self._v_samples: list[float] = []

    def process(self, obs: FrameObservation, dt: float) -> None:
        for midi, runs in obs.lane_runs.items():
            active = self._tracks.setdefault(midi, [])
            self._match_lane(midi, active, runs, obs, dt)
        # age & close tracks that went unmatched this frame
        for midi, active in self._tracks.items():
            if midi in obs.lane_runs:
                continue
            for tr in active:
                tr.missing += 1
            self._reap(active)

    def _match_lane(self, midi, active, runs, obs, dt):
        runs = sorted(runs, key=lambda r: r.y_bottom)
        predicted = []
        for tr in active:
            pred = tr.last.y_bottom + self._v_est * dt
            predicted.append((pred, tr))
        used = set()
        gate = self._v_est * dt * 0.9 + max(4.0, self.roll_h * 0.04)
        for r in runs:
            best, best_d = None, gate
            for pred, tr in predicted:
                if id(tr) in used:
                    continue
                d = abs(r.y_bottom - pred)
                if d < best_d:
                    best_d, best = d, tr
            if best is not None:
                used.add(id(best))
                self._append(best, r, obs)
            else:
                self._start(midi, r, obs)
        # tracks with no run this frame
        for pred, tr in predicted:
            if id(tr) not in used:
                tr.missing += 1
        self._reap(active)

    def _append(self, tr: Track, r: LaneRun, obs: FrameObservation):
        prev = tr.last
        s = self._mk_sample(r, obs)
        # accumulate velocity evidence from clean (unclamped) leading edges
        if not s.bottom_clamped and not prev.bottom_clamped:
            dtt = s.t - prev.t
            if dtt > 0:
                v = (s.y_bottom - prev.y_bottom) / dtt
                if v > 0:
                    self._v_samples.append(v)
                    if len(self._v_samples) >= 12:
                        self._v_est = float(np.median(self._v_samples[-400:]))
        tr.samples.append(s)
        tr.missing = 0

    def _start(self, midi, r, obs):
        tr = Track(id=self._next_id, midi=midi)
        self._next_id += 1
        tr.samples.append(self._mk_sample(r, obs))
        self._tracks[midi].append(tr)

    def _mk_sample(self, r: LaneRun, obs: FrameObservation) -> _Sample:
        return _Sample(
            t=obs.time, frame=obs.index,
            y_top=r.y_top, y_bottom=r.y_bottom, val=r.val, sat=r.sat,
            cluster=r.cluster,
            top_clamped=r.y_top <= 1.0,
            bottom_clamped=r.y_bottom >= self.roll_h - 2.0,
        )

    def _reap(self, active: list[Track]):
        keep = []
        for tr in active:
            if tr.missing >= 3:
                self._closed.append(tr)
            else:
                keep.append(tr)
        active[:] = keep

    def finalize(self) -> tuple[list[Track], float]:
        for active in self._tracks.values():
            self._closed.extend(active)
        self._tracks.clear()
        v = float(np.median(self._v_samples)) if self._v_samples else self._v_est
        return self._closed, max(v, 1e-3)


# ---------------------------------------------------------------------------
# Track -> NoteEvent
# ---------------------------------------------------------------------------

def build_notes(tracks: list[Track], geom: KeyboardGeometry, fps: float,
                v_global: float, min_track_frames: int = 2) -> list[NoteEvent]:
    strike_y = float(geom.strike_y)
    notes: list[NoteEvent] = []
    nid = 1
    frame_dt = 1.0 / fps
    for tr in tracks:
        if len(tr.samples) < min_track_frames:
            # 1-frame artefact — keep only if it clearly straddled the line,
            # otherwise drop as flicker (§35). We drop; a real note persists.
            continue
        v = _fit_velocity(tr, v_global)
        onset = _crossing_time(tr, strike_y, v, edge="bottom")
        if onset is None:
            continue
        length_px, len_conf, top_clamped = _bar_length(tr)
        dur = length_px / v if v > 0 else frame_dt
        end = onset + dur

        n = NoteEvent(
            id=nid, midi=tr.midi, start=onset, end=end,
            track_id=tr.id,
            lane_x=geom.lanes[tr.midi].center if tr.midi in geom.lanes else None,
            source_frames=[s.frame for s in tr.samples],
        )
        nid += 1
        # velocity from core brightness (relative; calibrated later, §23)
        core_val = float(np.median([s.val for s in tr.samples]))
        n.velocity = int(np.clip(round(core_val / 255.0 * 110 + 12), 1, 127))
        n.velocity_observed = False

        # confidences
        n.pitch_confidence = geom.confidence
        n.timing_confidence = _timing_confidence(tr, v, fps)
        n.duration_confidence = len_conf
        n.detection_confidence = float(np.clip(len(tr.samples) / 6.0, 0.4, 1.0))
        if top_clamped:
            n.flag("duration_unbounded_top")  # bar entered before fully visible (§37)
            n.duration_confidence = min(n.duration_confidence, 0.4)
        if len(tr.samples) <= 2:
            n.flag("short_track")
        notes.append(n)
    notes.sort(key=lambda x: (x.start, x.midi))
    # re-id in temporal order for stable references
    for i, n in enumerate(notes):
        n.id = i + 1
    return notes


def _fit_velocity(tr: Track, v_global: float) -> float:
    pts = [(s.t, s.y_bottom) for s in tr.samples
           if not s.bottom_clamped]
    if len(pts) >= 3:
        ts = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
        (slope, _), *_ = np.linalg.lstsq(A, ys, rcond=None)
        if slope > 1e-3:
            # blend with global to stay robust on short tracks
            w = min(1.0, len(pts) / 8.0)
            return float(w * slope + (1 - w) * v_global)
    return v_global


def _crossing_time(tr: Track, strike_y: float, v: float, edge: str) -> Optional[float]:
    """Sub-frame time at which the given edge reaches the strike line (§53)."""
    key = (lambda s: s.y_bottom) if edge == "bottom" else (lambda s: s.y_top)
    samples = tr.samples
    # find bracket where edge crosses strike_y
    for a, b in zip(samples, samples[1:]):
        ya, yb = key(a), key(b)
        if ya <= strike_y <= yb and yb != ya:
            frac = (strike_y - ya) / (yb - ya)
            return a.t + frac * (b.t - a.t)
    # never bracketed (bar exited between frames / clamped): extrapolate from
    # the last clean sample using velocity.
    last = None
    for s in reversed(samples):
        if not s.bottom_clamped:
            last = s
            break
    last = last or samples[-1]
    yb = key(last)
    return last.t + (strike_y - yb) / v if v > 0 else last.t


def _bar_length(tr: Track) -> tuple[float, float, bool]:
    """Median visible bar length in px, a confidence, and whether the top edge
    was clamped at the roll ceiling (length under-measured, §37)."""
    lengths = [s.y_bottom - s.y_top for s in tr.samples]
    top_clamped = any(s.top_clamped for s in tr.samples)
    if not lengths:
        return 0.0, 0.3, top_clamped
    med = float(np.median(lengths))
    spread = float(np.std(lengths))
    conf = float(np.clip(1.0 - spread / (med + 1e-6), 0.3, 1.0))
    return med, conf, top_clamped


def _timing_confidence(tr: Track, v: float, fps: float) -> float:
    # more clean pre-strike samples + steady velocity => higher confidence
    clean = sum(1 for s in tr.samples if not s.bottom_clamped)
    return float(np.clip(0.5 + 0.1 * clean, 0.5, 0.99))
