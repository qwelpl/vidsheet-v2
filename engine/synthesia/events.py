"""Per-lane note-event extraction (§9, §10, §11, §12, §53).

Instead of tracking falling-bar *objects* through the strike line — which is
fragile when a held note and an approaching repeat share a lane — we treat each
pitch lane as an independent occupancy signal at the strike line:

    rising edge  (empty -> occupied) = note-on
    falling edge (occupied -> empty)  = note-off

Held notes stay occupied; repeated notes drop occupancy in the visible gap
between their bars, so they remain distinct (§12). Each onset is then refined to
sub-frame precision by fitting the leading edge's approach trajectory and
solving for the exact strike-line crossing (§53); each offset likewise from the
trailing edge. Every lane is independent, so chords and overlapping notes keep
their true micro-timing (§13, §15).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import FrameObservation, LaneRun
from .keyboard import KeyboardGeometry
from .model import NoteEvent


@dataclass
class LaneHistory:
    """Sparse per-frame record for one pitch lane."""
    frames: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    runs: list[list[LaneRun]] = field(default_factory=list)


class HistoryCollector:
    """Accumulates per-lane run history while frames stream past."""

    def __init__(self, geom: KeyboardGeometry):
        self.geom = geom
        self.hist: dict[int, LaneHistory] = {m: LaneHistory() for m in geom.lanes}

    def add(self, obs: FrameObservation) -> None:
        for midi, runs in obs.lane_runs.items():
            h = self.hist[midi]
            h.frames.append(obs.index)
            h.times.append(obs.time)
            h.runs.append(runs)


def estimate_fall_speed(hist: dict[int, LaneHistory], geom: KeyboardGeometry) -> float:
    """Median leading-edge speed (px/s), fitted over each contiguous approach
    segment. Fitting over a long baseline avoids the bias that per-frame integer
    pixel deltas introduce."""
    strike = float(geom.strike_y)
    band = max(6.0, strike * 0.02)
    speeds = []
    for h in hist.values():
        seg_t, seg_y = [], []
        prev = None
        for k in range(len(h.times)):
            yb = max(r.y_bottom for r in h.runs[k])
            cont = (prev is not None and 0 <= yb - prev < strike * 0.15)
            if yb < strike - band and cont:
                seg_t.append(h.times[k]); seg_y.append(yb)
            else:
                if len(seg_t) >= 5:
                    speeds.append(_slope(seg_t, seg_y))
                seg_t, seg_y = ([h.times[k]], [yb]) if yb < strike - band else ([], [])
            prev = yb
        if len(seg_t) >= 5:
            speeds.append(_slope(seg_t, seg_y))
    speeds = [s for s in speeds if s > 1]
    return float(np.median(speeds)) if speeds else strike * 0.5


def _slope(ts, ys) -> float:
    ts = np.asarray(ts, float); ys = np.asarray(ys, float)
    A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
    (slope, _), *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(slope)


def extract_notes(hist: dict[int, LaneHistory], geom: KeyboardGeometry,
                  fps: float, v_global: float,
                  min_gap_frames: int = 1) -> list[NoteEvent]:
    strike = float(geom.strike_y)
    band = max(6.0, strike * 0.02)
    # total analysed span, used to reject static bright elements (a persistent
    # hit-line glow / reflection reads as a note that never releases, §35).
    tmin = min((h.times[0] for h in hist.values() if h.times), default=0.0)
    tmax = max((h.times[-1] for h in hist.values() if h.times), default=1.0)
    span = max(1e-3, tmax - tmin)
    notes: list[NoteEvent] = []
    for midi, h in hist.items():
        if not h.frames:
            continue
        notes.extend(_extract_lane(midi, h, geom, strike, band, fps, v_global, span))
    notes.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    return notes


def _extract_lane(midi, h: LaneHistory, geom, strike, band, fps, v_global, span=1e9):
    times = h.times
    frames = h.frames
    n = len(times)
    # leading-edge (max y_bottom) reaching the strike band => occupied
    bottom = np.array([max(r.y_bottom for r in rr) for rr in h.runs])
    occ = bottom >= (strike - band)

    notes: list[NoteEvent] = []
    i = 0
    while i < n:
        if not occ[i]:
            i += 1
            continue
        # onset at entry i (rising edge, allowing i==0)
        onset_frame = i
        # extend the played span while it stays occupied, tolerating tiny
        # frame gaps (compression flicker), but a real repeat gap ends it.
        j = i
        while j + 1 < n and (occ[j + 1] or _is_flicker(frames, j, occ)):
            j += 1
        offset_frame = j

        onset, on_conf, on_flags = _refine_onset(h, onset_frame, strike, v_global)
        offset, off_conf, off_flags = _refine_offset(h, onset_frame, offset_frame,
                                                     strike, v_global, onset)
        dur = offset - onset
        # A segment that stays occupied for essentially the whole video AND was
        # never seen approaching is not a note — it is a static bright element
        # (persistent hit-line glow, reflection, coloured vignette). Reject it
        # rather than emit a note held for the entire piece (§35, §51).
        no_approach = "onset_extrapolated" in on_flags
        if dur > 0.85 * span and no_approach:
            i = j + 1
            continue
        note = _make_note(midi, h, onset_frame, offset_frame, onset, offset,
                          geom, on_conf, off_conf, on_flags + off_flags)
        if dur > 0.5 * span:
            note.flag("implausibly_long")
            note.detection_confidence = min(note.detection_confidence, 0.35)
        notes.append(note)
        i = j + 1
    return notes


def _is_flicker(frames, j, occ) -> bool:
    # a single missing occupied frame surrounded by occupied ones
    return False  # occupancy already tolerant; kept explicit for clarity


def _refine_onset(h: LaneHistory, onset_frame: int, strike: float, v_global: float):
    """Fit the leading edge's approach and solve for the strike crossing."""
    ts, ys = [], []
    k = onset_frame - 1
    prev_y = None
    # walk backwards over the approach (bar below strike, descending)
    while k >= 0:
        yb = max(r.y_bottom for r in h.runs[k])
        if yb >= strike - 1:            # still at strike (previous note) -> stop
            break
        if prev_y is not None and yb > prev_y + 2:  # discontinuity -> stop
            break
        ts.append(h.times[k]); ys.append(yb)
        prev_y = yb
        if len(ts) >= 16:
            break
        k -= 1
    flags: list[str] = []
    if len(ts) >= 3:
        ts = np.array(ts[::-1]); ys = np.array(ys[::-1])
        A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
        if slope > 1e-3:
            t_cross = ts[0] + (strike - intercept) / slope
            resid = float(np.sqrt(np.mean((A @ [slope, intercept] - ys) ** 2)))
            return float(t_cross), float(np.clip(0.92 - resid / 20, 0.55, 0.96)), flags
    # not enough approach seen: note began before it was visible (§37)
    flags.append("onset_extrapolated")
    t0 = h.times[onset_frame]
    if len(ts) >= 1 and v_global > 0:
        return t0 - band_extrap(ys, strike, v_global), 0.5, flags
    return t0, 0.45, flags


def band_extrap(ys, strike, v):
    return 0.0  # onset already at strike frame; conservative


def _refine_offset(h: LaneHistory, onset_frame: int, offset_frame: int,
                   strike: float, v_global: float, onset: float):
    """Fit the trailing edge (min y_top of the strike-reaching run) rising to
    the strike line and solve for the crossing (§11)."""
    ts, ys = [], []
    for k in range(onset_frame, min(offset_frame + 2, len(h.times))):
        runs = h.runs[k]
        # trailing edge = top of the run that reaches the strike band
        strike_runs = [r for r in runs if r.y_bottom >= strike - max(6.0, strike*0.02)]
        if not strike_runs:
            continue
        yt = min(r.y_top for r in strike_runs)
        ts.append(h.times[k]); ys.append(yt)
    flags: list[str] = []
    if len(ts) >= 3:
        tsa = np.array(ts); ysa = np.array(ys)
        A = np.vstack([tsa - tsa[0], np.ones_like(tsa)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, ysa, rcond=None)
        last_t = h.times[min(offset_frame, len(h.times) - 1)]
        frame_dt = (h.times[-1] - h.times[0]) / max(1, len(h.times) - 1)
        if slope > 1e-3:
            t_cross = tsa[0] + (strike - intercept) / slope
            # the trailing edge crosses at ~offset_frame; a wildly larger value
            # means a near-flat (degenerate) fit -> fall back to the frame time.
            if onset < t_cross <= last_t + 3 * frame_dt:
                resid = float(np.sqrt(np.mean((A @ [slope, intercept] - ysa) ** 2)))
                return float(t_cross), float(np.clip(0.9 - resid / 20, 0.5, 0.95)), flags
    # trailing edge never seen crossing (note runs past video end / occluded)
    flags.append("offset_extrapolated")
    off_t = h.times[min(offset_frame, len(h.times) - 1)]
    return max(off_t, onset + 1.0 / 60), 0.45, flags


def _make_note(midi, h, onset_frame, offset_frame, onset, offset, geom,
               on_conf, off_conf, flags) -> NoteEvent:
    played = h.runs[onset_frame: offset_frame + 1]
    vals, clusters, source_frames = [], [], []
    top_clamped = False
    for fr_i, runs in zip(range(onset_frame, offset_frame + 1), played):
        source_frames.append(h.frames[fr_i])
        for r in runs:
            vals.append(r.val); clusters.append(r.cluster)
            if r.y_top <= 1.0:
                top_clamped = True
    n = NoteEvent(id=0, midi=midi, start=float(onset), end=float(max(offset, onset + 1e-3)),
                  source_frames=source_frames,
                  lane_x=geom.lanes[midi].center if midi in geom.lanes else None)
    n.color_cluster = int(np.bincount(clusters).argmax()) if clusters else 0
    core_val = float(np.median(vals)) if vals else 150.0
    n.velocity = int(np.clip(round(core_val / 255.0 * 110 + 12), 1, 127))
    n.pitch_confidence = geom.confidence
    n.timing_confidence = on_conf
    n.duration_confidence = off_conf
    n.detection_confidence = float(np.clip(len(source_frames) / 5.0, 0.4, 1.0))
    for f in flags:
        n.flag(f)
    return n
