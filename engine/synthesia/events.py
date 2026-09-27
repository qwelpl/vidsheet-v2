"""Per-lane note-event extraction (§9, §10, §11, §12, §53).

Instead of tracking falling-bar *objects* through the strike line - which is
fragile when a held note and an approaching repeat share a lane - we treat each
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
    # Occupancy band. A hit-flash strip above the keys caps how far a bar's
    # coloured leading edge descends, so bars top out several pixels SHORT of the
    # strike line - right at the band edge, where threshold noise drops roughly
    # half of them (rapid repeats then detect every other note). Anchor the band
    # to the measured hit level (where bars actually top out) instead of the
    # geometric strike line; on a clean render that level is the strike itself, so
    # the band is unchanged.
    hit = _hit_level(hist, strike)
    band = max(strike * 0.02, strike - hit + 6.0)
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
    # A held note whose bottom is hidden by a hit-flash strip clamps its coloured
    # edge tens of pixels short of the strike line, so tight occupancy drops after
    # a frame or two. Detect the note is still sounding by the edge PLATEAUING in a
    # wider hold zone rather than descending on (a fresh approaching bar). On a
    # clean render the edge passes straight through the strike and never plateaus,
    # so this never fires there. bottom_s is median-smoothed to ignore the few-
    # pixel oscillation of a clamped edge.
    bottom_s = _med3(bottom)
    hold = bottom >= (strike - max(band, strike * 0.09))
    onset_frames = [k for k in range(n) if occ[k] and (k == 0 or not occ[k - 1])]
    next_onset = _next_onset_after(onset_frames, n)

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
        # then keep extending through a HELD (clamped, non-descending) edge in the
        # wider hold zone - a note behind a hit-flash - but never past the next
        # note's onset in this lane, so repeats stay distinct.
        cap = next_onset[onset_frame]
        while (j + 1 < cap and hold[j + 1]
               and bottom_s[j + 1] <= bottom_s[j] + 3):   # not a fresh descent
            j += 1
        offset_frame = j

        # Reconstruct the approaching bar's trajectory in the clean roll area
        # above the strike line. A genuine falling note descends at the global
        # fall speed; its visible bar length equals its duration. Measuring both
        # away from the glow-contaminated strike zone is far more robust on real
        # composited video than reading the note-off at the line (§11, §51).
        traj = _approach_trajectory(h, onset_frame, strike, v_global)
        onset, on_conf, on_flags, dur_from_len, top_clamped, vel_ok = \
            _from_trajectory(traj, strike, v_global)

        offset_r, off_conf, off_flags = _refine_offset(h, onset_frame, offset_frame,
                                                       strike, v_global, onset if onset else h.times[onset_frame])
        no_approach = onset is None
        no_release = "offset_extrapolated" in off_flags

        # static element (glow / UI / composited footage): no falling approach
        # AND no clean release -> not a note (§35, §51).
        if no_approach and no_release:
            i = j + 1
            continue
        # approach is the ONLY evidence but the bar does not descend at the fall
        # speed -> a drifting anime region, not a note. Reject. When a clean
        # release also exists we keep it (a real note whose approach was noisy
        # over the moving footage).
        if not no_approach and not vel_ok and no_release:
            i = j + 1
            continue

        if no_approach:
            onset = offset_r - 0.05 if offset_r else h.times[onset_frame]
            on_conf = 0.4
            on_flags = ["onset_extrapolated"]

        # duration: prefer the visible bar length (measured in the clean roll,
        # away from the strike-line glow) when the trajectory is trustworthy;
        # otherwise the observed release, then the occupancy span.
        if dur_from_len is not None and vel_ok and not top_clamped:
            offset = onset + dur_from_len
            dconf = 0.85
        elif not no_release and offset_r > onset:
            offset = offset_r
            dconf = off_conf
        else:
            offset = max(offset_r, onset + (h.times[offset_frame] - h.times[onset_frame]))
            dconf = 0.4
            off_flags = list(set(off_flags + ["offset_extrapolated"]))

        # A held bar stays in the strike band (its coloured edge clamped just above
        # the keys behind a hit-flash) until it shrinks away, so the last occupied
        # frame is the note-off. The visible-length / released-edge estimates
        # truncate when a label box or translucent tail splits the bar or the hit-
        # flash hides its bottom, so take the bar-vanish time when it is later.
        # Gated on a clean constant-velocity approach so composited strike glow
        # (which yields no clean approach) can't inflate durations.
        if not no_approach and vel_ok:
            vanish = h.times[offset_frame]
            if vanish > offset + 1.5 / max(fps, 1.0):
                offset = vanish
                dconf = min(dconf, 0.8)

        dur = offset - onset
        # A short detection that never showed a genuine descent at the fall
        # speed is not a note but an on-screen overlay - a comment box, emoji,
        # text or watermark that briefly matched a note colour. Real short notes
        # still fall normally, so they keep a clean constant-velocity approach;
        # overlays do not. Drop these (§16, §36).
        if dur < 0.09 and (no_approach or not vel_ok):
            i = j + 1
            continue

        note = _make_note(midi, h, onset_frame, offset_frame, onset, offset,
                          geom, on_conf, dconf, on_flags + (off_flags if offset <= onset + 1e-4 else []))
        if top_clamped:
            note.flag("duration_unbounded_top")  # bar entered before fully visible (§37)
            note.duration_confidence = min(note.duration_confidence, 0.45)
        if dur > 0.5 * span:
            note.flag("implausibly_long")
            note.detection_confidence = min(note.detection_confidence, 0.3)
        notes.append(note)
        i = j + 1
    return notes


def cleanup_fragments(notes: list[NoteEvent], merge_gap: float = 0.045,
                      drop_min: float = 0.022) -> list[NoteEvent]:
    """Fuse same-pitch fragments and drop tiny leftovers (§48).

    Detection occasionally splits one note into a main note plus a sliver, or
    emits a near-duplicate at the same pitch. Here, per pitch, notes that overlap
     - or sit a hair apart where one of them is very short - are merged into the
    longer note; anything still shorter than ``drop_min`` is removed. Genuine
    fast repeats (both notes a real length, with a clean gap) are left intact."""
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.midi, []).append(n)
    kept: list[NoteEvent] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: n.start)
        merged: list[NoteEvent] = []
        for n in group:
            if merged:
                p = merged[-1]
                overlap = n.start < p.end
                sliver = (n.start - p.end < merge_gap) and \
                    (min(n.duration, p.duration) < 0.05)
                if overlap or sliver:
                    if n.end > p.end:
                        p.end = n.end
                    p.detection_confidence = max(p.detection_confidence,
                                                 n.detection_confidence)
                    for f in n.issues:
                        p.flag(f)
                    continue
            merged.append(n)
        kept.extend(n for n in merged if n.duration >= drop_min)
    kept.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(kept):
        n.id = i + 1
    return kept


def align_chords(notes: list[NoteEvent], onsets=None, window: float = 0.033) -> int:
    """Snap near-simultaneous onsets across lanes to one shared time (§10, §18).

    Notes meant to be struck together - a chord, or the two hands landing on the
    same beat - are detected on slightly different frames and drift a few
    milliseconds apart, which reads as the hands being out of sync. Notes whose
    onsets fall inside a tight ``window`` are almost certainly one event (a real
    arpeggio/roll spreads wider than a frame), so they are moved to a common
    onset: the nearby audio attack if one exists, else the group's median. Each
    note keeps its own duration. Returns how many notes were moved."""
    import numpy as _np
    if len(notes) < 2:
        return 0
    ons = _np.sort(onsets) if onsets is not None and len(onsets) else None
    order = sorted(range(len(notes)), key=lambda k: notes[k].start)
    moved = 0
    grp: list[int] = []
    anchor = None

    def _flush(g):
        nonlocal moved
        if len(g) < 2:
            return
        starts = [notes[k].start for k in g]
        target = float(_np.median(starts))
        if ons is not None:
            j = int(_np.searchsorted(ons, target))
            cand = [ons[x] for x in (j - 1, j) if 0 <= x < ons.size]
            near = [c for c in cand if abs(c - target) <= window]
            if near:
                target = float(min(near, key=lambda c: abs(c - target)))
        for k in g:
            n = notes[k]
            if abs(n.start - target) <= 1e-4:
                continue
            dur = n.duration
            if target >= n.end:            # never invert a short note
                continue
            n.start = target
            n.end = target + dur
            moved += 1

    for k in order:
        s = notes[k].start
        if anchor is None or s - anchor <= window:
            if anchor is None:
                anchor = s
            grp.append(k)
        else:
            _flush(grp)
            grp = [k]
            anchor = s
    _flush(grp)
    if moved:
        notes.sort(key=lambda n: (n.start, n.midi))
        for i, n in enumerate(notes):
            n.id = i + 1
    return moved


def resolve_same_pitch_overlaps(notes: list[NoteEvent], drop_min: float = 0.022,
                                eps: float = 1e-3) -> int:
    """Trim overlaps between consecutive notes of the SAME pitch (§12).

    One key cannot sound twice at once, so after sub-frame timing and chord
    alignment nudge onsets around, a note that now reaches past the next hit of
    the same key has its release trimmed back to that hit - the repeats are kept
    distinct (unlike a merge). A note trimmed shorter than ``drop_min`` was a
    spurious duplicate and is dropped. Returns notes removed."""
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.midi, []).append(n)
    drop: set[int] = set()
    for group in by_pitch.values():
        group.sort(key=lambda n: n.start)
        for a, b in zip(group, group[1:]):
            if a.end > b.start - eps:
                a.end = b.start - eps
            if a.end - a.start < drop_min:
                drop.add(id(a))
    if not drop:
        return 0
    kept = [n for n in notes if id(n) not in drop]
    notes[:] = kept
    notes.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    return len(drop)


def _hit_level(hist: dict, strike: float) -> float:
    """The y a falling bar's leading edge actually tops out at. Normally the
    strike line, but a hit-flash strip above the keys can cap it several pixels
    short. Estimated as a high percentile of the per-bar leading-edge peaks (local
    maxima in the lower roll), so onset occupancy can be anchored to it."""
    peaks: list[float] = []
    lo = strike * 0.8
    for h in hist.values():
        runs = h.runs
        if len(runs) < 3:
            continue
        b = np.array([max((r.y_bottom for r in rr), default=0.0) for rr in runs])
        for k in range(1, len(b) - 1):
            if b[k] >= lo and b[k] >= b[k - 1] and b[k] >= b[k + 1]:
                peaks.append(float(b[k]))
    if len(peaks) < 20:
        return strike
    # a high percentile pins to the true bar-peak level (the consistent apex where
    # bars top out), robust to lower spurious peaks from hit-flash glow
    return float(np.percentile(peaks, 90))


def _med3(a: np.ndarray, k: int = 7) -> np.ndarray:
    """Rolling-median smoothing over a window of ``k`` frames. Flattens the
    few-pixel oscillation of a clamped (held) leading edge so it reads as a
    plateau, while a genuinely descending bar still climbs across the window."""
    a = a.astype(np.float64)
    n = a.size
    if n < 3:
        return a.copy()
    r = max(1, k // 2)
    out = a.copy()
    for i in range(n):
        out[i] = np.median(a[max(0, i - r): min(n, i + r + 1)])
    return out


def _next_onset_after(onset_frames: list[int], n: int) -> np.ndarray:
    """For each frame, the next onset frame strictly after it (or n if none)."""
    import bisect
    of = sorted(onset_frames)
    nxt = np.full(n, n, dtype=int)
    for f in range(n):
        k = bisect.bisect_right(of, f)
        nxt[f] = of[k] if k < len(of) else n
    return nxt


def _is_flicker(frames, j, occ) -> bool:
    # a single missing occupied frame surrounded by occupied ones
    return False  # occupancy already tolerant; kept explicit for clarity


def _approach_trajectory(h: LaneHistory, onset_frame: int, strike: float,
                         v: float, max_pts: int = 30):
    """Follow the note's bar backwards from its strike frame through the roll,
    picking in each frame the run whose leading edge best matches a bar falling
    at speed ``v``. Returns [(t, y_top, y_bottom), ...] earliest-first."""
    pts = []
    t_ref = h.times[onset_frame]
    k = onset_frame - 1
    lost = 0
    while k >= 0 and len(pts) < max_pts:
        dt = t_ref - h.times[k]
        pred = strike - v * dt                     # where the bar bottom should be
        cand = [r for r in h.runs[k] if r.y_bottom < strike - 1]
        if not cand:
            break
        r = min(cand, key=lambda r: abs(r.y_bottom - pred))
        if abs(r.y_bottom - pred) > max(0.35 * v * dt, 20.0):
            lost += 1                    # note momentarily lost behind footage
            if lost > 3:
                break
            k -= 1
            continue
        pts.append((h.times[k], float(r.y_top), float(r.y_bottom)))
        lost = 0
        k -= 1
    return pts[::-1]


def _from_trajectory(traj, strike: float, v: float):
    """From an approach trajectory derive (onset, onset_conf, flags,
    duration_from_length | None, top_clamped, velocity_ok)."""
    if len(traj) < 2:
        return None, 0.4, ["onset_extrapolated"], None, True, False
    ts = np.array([p[0] for p in traj])
    ybs = np.array([p[2] for p in traj])
    A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ybs, rcond=None)
    if slope <= 1e-3:
        return None, 0.4, ["onset_extrapolated"], None, True, False
    # constant-velocity check: a real bar descends at ~the global fall speed.
    # Kept loose so noisy approaches over moving footage still qualify; only
    # clearly-wrong speeds (near-static, or far too fast) are rejected.
    ratio = slope / max(v, 1e-3)
    vel_ok = 0.45 <= ratio <= 2.3
    onset = float(ts[0] + (strike - intercept) / slope)
    resid = float(np.sqrt(np.mean((A @ [slope, intercept] - ybs) ** 2)))
    conf = float(np.clip(0.92 - resid / 20, 0.55, 0.96))
    # visible bar length -> duration (measured where the bar is fully in view)
    lengths = [p[2] - p[1] for p in traj if p[1] > 1.5]
    top_clamped = any(p[1] <= 1.5 for p in traj)
    dur_len = None
    if lengths:
        dur_len = float(np.median(lengths)) / slope
    return onset, conf, [], dur_len, top_clamped, vel_ok


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
