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
            # cap the prediction at the strike line: once a note's leading edge
            # reaches it, the edge is pinned there while the note plays, so the
            # same track must keep matching the (clamped) run rather than being
            # abandoned and re-detected as a spurious repeat.
            pred = min(self.roll_h - 1.0, tr.last.y_bottom + self._v_est * dt)
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
        # note-on  = leading (bottom) edge crosses the strike line
        # note-off = trailing (top) edge crosses the strike line
        # both edges are directly observed while above the line (§10, §11).
        onset, on_conf, on_extrap = _edge_crossing(tr, strike_y, v, "bottom")
        offset, off_conf, off_extrap = _edge_crossing(tr, strike_y, v, "top")
        if onset is None:
            continue
        length_px, len_conf, top_clamped = _bar_length(tr)
        if offset is None or offset <= onset:
            # fall back to bar-length / speed if the trailing edge was unseen
            offset = onset + (length_px / v if v > 0 else frame_dt)
            off_extrap = True
        end = offset

        clusters = [s.cluster for s in tr.samples]
        dom_cluster = int(np.bincount(clusters).argmax()) if clusters else 0
        n = NoteEvent(
            id=nid, midi=tr.midi, start=onset, end=end,
            track_id=tr.id, color_cluster=dom_cluster,
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
        n.timing_confidence = on_conf
        n.duration_confidence = min(off_conf, len_conf)
        n.detection_confidence = float(np.clip(len(tr.samples) / 6.0, 0.4, 1.0))
        if on_extrap:
            n.flag("onset_extrapolated")   # note began before it was visible (§37)
            n.timing_confidence = min(n.timing_confidence, 0.5)
        if off_extrap:
            n.flag("offset_extrapolated")
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


def _edge_crossing(tr: Track, strike_y: float, v: float, edge: str):
    """Sub-frame time an edge reaches the strike line, with a confidence and an
    ``extrapolated`` flag (§10, §11, §53).

    The edge is tracked while it is above the line (clean, unclamped). We fit a
    line to those clean samples and solve for y == strike_y. A direct bracket
    (edge observed on both sides of the line across two frames) yields the
    highest confidence; a pure extrapolation from approach samples is flagged.
    Returns (time | None, confidence, extrapolated)."""
    if edge == "bottom":
        key = lambda s: s.y_bottom
        clean = [s for s in tr.samples if not s.bottom_clamped]
    else:
        key = lambda s: s.y_top
        clean = [s for s in tr.samples if not s.top_clamped]
    samples = tr.samples

    # 1) linear fit over clean approach samples, solved for y == strike_y.
    #    Constant fall speed makes this exact and sub-frame accurate, and it
    #    sidesteps the 1-2 px clamp at the bottom of the roll (§53).
    if len(clean) >= 3:
        ts = np.array([s.t for s in clean])
        ys = np.array([key(s) for s in clean])
        A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
        if slope > 1e-3:
            t_cross = ts[0] + (strike_y - intercept) / slope
            residual = float(np.sqrt(np.mean((A @ [slope, intercept] - ys) ** 2)))
            extrap_px = strike_y - ys.max()
            extrap = extrap_px > slope * (2.0 / max(1e-6, len(clean)))  # >~2 frames out
            conf = float(np.clip(0.92 - residual / 20.0, 0.55, 0.95))
            return float(t_cross), conf, bool(extrap and extrap_px > 25)

    # 2) direct bracket across the line (includes the clamp sample)
    for a, b in zip(samples, samples[1:]):
        ya, yb = key(a), key(b)
        if ya < strike_y <= yb and yb != ya:
            frac = (strike_y - ya) / (yb - ya)
            return a.t + frac * (b.t - a.t), 0.9, False
        clamped_b = (b.bottom_clamped if edge == "bottom" else b.top_clamped)
        if clamped_b and ya < strike_y and yb >= strike_y - 3:
            frac = min(1.0, (strike_y - ya) / max(1e-6, (yb - ya)))
            return a.t + frac * (b.t - a.t), 0.75, False

    # 3) last resort: single sample + global velocity
    if samples:
        s = clean[-1] if clean else samples[-1]
        y = key(s)
        if v > 0:
            return s.t + (strike_y - y) / v, 0.4, True
    return None, 0.3, True


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
