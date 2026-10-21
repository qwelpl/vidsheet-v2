"""Tempo, beat grid and rhythmic quantization (§19, §20).

Raw performance timing is authoritative and is never overwritten. This module
derives a *separate* notation-friendly quantized view: it estimates tempo from
inter-onset intervals, lays a beat grid, and snaps a copy of the notes to it
while recording the timing error so ambiguous passages can be surfaced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import NoteEvent, TempoSegment


@dataclass
class TempoAnalysis:
    bpm: float
    beat_period: float
    beat_phase: float          # time of beat 0
    time_signature: tuple[int, int]
    segments: list[TempoSegment]
    confidence: float
    # Drift-tracking beat grid (§19, §20). A live performance speeds up and slows
    # down; a single global bpm+phase then slides a fraction of a beat across the
    # piece and scatters onsets onto off-grid (32nd/dotted) positions - the "timing
    # is off" look. These sampled arrays describe a monotone time -> sixteenth-count
    # map that bends with the local tempo, so ``beat_at`` places every note on the
    # beat grid it was actually played against. ``None`` falls back to the linear
    # global grid (too few notes to track).
    grid_ts: object = None       # np.ndarray of sample times (s)
    grid_phi: object = None      # np.ndarray of cumulative sixteenth count at ts
    grid_delta: float = 0.0      # sub-sixteenth phase so onsets land on integers
    grid_shift: int = 0          # downbeat offset in sixteenths (0..3)

    def beat_at(self, t: float) -> float:
        """Fractional beat (quarter) index of time ``t`` on the tracked grid."""
        if self.grid_ts is None:
            return (t - self.beat_phase) / self.beat_period
        phi = float(np.interp(t, self.grid_ts, self.grid_phi))
        return (phi - self.grid_delta - self.grid_shift) / 4.0

    def to_dict(self) -> dict:
        return {
            "bpm": round(self.bpm, 3),
            "beat_period": self.beat_period,
            "beat_phase": self.beat_phase,
            "time_signature": list(self.time_signature),
            "confidence": self.confidence,
            "segments": [s.to_dict() for s in self.segments],
        }


def estimate_tempo(notes: list[NoteEvent], fps: float) -> TempoAnalysis:
    onsets = np.array(sorted(n.start for n in notes))
    if len(onsets) < 4:
        return TempoAnalysis(120.0, 0.5, 0.0, (4, 4), [TempoSegment(0.0, 120.0)], 0.2)

    iois = np.diff(onsets)
    iois = iois[(iois > 0.04) & (iois < 3.0)]
    if len(iois) < 3:
        return TempoAnalysis(120.0, 0.5, 0.0, (4, 4), [TempoSegment(0.0, 120.0)], 0.2)

    # Find the tempo whose sixteenth grid the onsets sit on most tightly (§19).
    # The correct tempo is a RAZOR-SHARP peak: over a 3-minute piece a 1 bpm
    # error drifts the grid most of a beat by the end, so the fit at the true
    # bpm towers over its neighbours. That sharpness means we must scan finely -
    # a coarse pulse estimate lands in the flat basin beside the needle and a
    # local refine can't climb to it. So: a cheap coarse pass over the musical
    # range to find the basin, then a fine pass to pin it, phase optimised at
    # each step. Concentration is onset count near a grid line, which does not
    # inflate with a finer grid the way a fractional score does, so staying
    # inside a musical bpm range is enough to avoid the double/half-tempo trap.
    coarse = np.arange(65.0, 170.01, 0.2)
    cc = np.array([_grid_conc(onsets, 60.0 / b)[0] for b in coarse])
    b0 = float(coarse[int(cc.argmax())])
    fine = np.arange(max(65.0, b0 - 0.4), min(170.0, b0 + 0.4) + 1e-9, 0.01)
    best_conc, best_period, phase = -1.0, 60.0 / b0, 0.0
    for b in fine:
        period = 60.0 / b
        c, ph = _grid_conc(onsets, period)
        if c > best_conc:
            best_conc, best_period, phase = c, period, ph

    # A denser grid always fits at least as well, so the sharpest-concentration
    # tempo is prone to a double-tempo error: a piece of mostly eighth notes reads
    # as mostly quarters one octave up, with the real notes sitting on the beat and
    # the faster grid lines populated only by onset jitter. Fold the tempo down
    # while the lines that WOULD vanish on halving (the odd sixteenth positions)
    # carry only a sparse jitter tail - but keep the fast tempo when those lines
    # hold real notes (a genuinely fast piece of sixteenth runs), so the two are
    # told apart by the data, not a fixed cutoff (§19).
    best_period, phase, best_conc = _fold_double_tempo(onsets, best_period, phase,
                                                       best_conc)
    bpm = 60.0 / best_period
    conf = float(np.clip(best_conc, 0.2, 0.98))

    beats = list(np.arange(phase, onsets[-1] + best_period, best_period))
    seg = TempoSegment(start=float(onsets[0]), bpm=float(bpm), beats=[float(b) for b in beats])
    ta = TempoAnalysis(bpm=float(bpm), beat_period=float(best_period),
                       beat_phase=float(phase), time_signature=(4, 4),
                       segments=[seg], confidence=conf)

    # Track the local tempo so a drifting/rubato performance still notates cleanly.
    grid = _build_beat_map(onsets, best_period)
    if grid is not None:
        ta.grid_ts, ta.grid_phi, ta.grid_delta, ta.grid_shift = grid
    return ta


def _local_period(seg: np.ndarray, p0: float, span: float = 0.12) -> float:
    """Best beat period for a short window of onsets, searched within ``span`` of
    the global period ``p0`` so a window can track the local tempo but never jump
    to a half/double-tempo lattice."""
    bpm0 = 60.0 / p0
    best_c, best_p = -1.0, p0
    for b in np.arange(bpm0 * (1 - span), bpm0 * (1 + span) + 1e-9, 0.1):
        c, _ = _grid_conc(seg, 60.0 / b)
        if c > best_c:
            best_c, best_p = c, 60.0 / b
    return best_p


def _build_beat_map(onsets: np.ndarray, p0: float,
                    win: float = 6.0, hop: float = 1.5, dt: float = 0.01):
    """Build a monotone time -> sixteenth-count map that follows the local tempo.

    Local beat period is estimated in overlapping windows, interpolated between
    window centres, and integrated (sixteenths/second) into a cumulative count
    ``phi(t)``. A single sub-sixteenth phase ``delta`` then slides the whole grid
    so the onsets land on integer sixteenths, and ``shift`` picks the downbeat.
    Returns ``(ts, phi, delta, shift)`` or ``None`` when too few windows qualify.
    """
    t0, t1 = float(onsets[0]), float(onsets[-1])
    if t1 - t0 < win or len(onsets) < 12:
        return None
    cs, ps = [], []
    c = t0
    while c <= t1 + 1e-9:
        seg = onsets[(onsets >= c - win / 2) & (onsets <= c + win / 2)]
        if len(seg) >= 6:
            cs.append(c)
            ps.append(_local_period(seg, p0))
        c += hop
    if len(cs) < 2:
        return None
    cs = np.asarray(cs, float)
    ps = np.asarray(ps, float)

    # Fine time axis padded past both ends (note releases can run past the last
    # onset); sixteenth rate = 4 / local beat period, held flat beyond the windows.
    ts = np.arange(t0 - 2.0, t1 + 10.0, dt)
    step16 = np.interp(ts, cs, ps, left=ps[0], right=ps[-1]) / 4.0
    rate = 1.0 / step16
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)])

    # Integration pins the local TEMPO but its phase drifts: a sub-percent bias in
    # the per-window period accumulates over minutes into a whole-sixteenth slip,
    # so a single global offset cannot line the grid up (the fractional parts come
    # out uniform). Re-lock the phase locally instead: measure the residual
    # sixteenth offset in each window (circular mean), unwrap it so it tracks the
    # slip continuously, and subtract that slowly-varying correction from phi. This
    # is the per-window phase fit, made into one monotone grid.
    po = np.interp(onsets, ts, phi)
    cw, phasors = [], []
    c = t0
    while c <= t1 + 1e-9:
        m = (onsets >= c - win / 2) & (onsets <= c + win / 2)
        if int(m.sum()) >= 6:
            ang = 2 * np.pi * (po[m] - np.floor(po[m]))
            cw.append(c)
            phasors.append(np.mean(np.exp(1j * ang)))
        c += hop
    if len(cw) < 2:
        return None
    delta_w = np.unwrap(np.angle(np.asarray(phasors))) / (2 * np.pi)
    delta_t = np.interp(ts, np.asarray(cw), delta_w,
                        left=delta_w[0], right=delta_w[-1])
    phi = phi - delta_t

    # Pick the downbeat phase (which sixteenth is beat 0) by the shift that puts
    # the most onsets on a beat line; ties keep 0.
    po = np.interp(onsets, ts, phi)
    idx = np.round(po).astype(int)
    shift, best = 0, -1
    for q in range(4):
        hit = int(np.sum((idx - q) % 4 == 0))
        if hit > best:
            best, shift = hit, q
    return ts, phi, 0.0, int(shift)


def _odd_line_frac(onsets: np.ndarray, period: float, phase: float,
                   sub: int = 4, tol: float = 0.025) -> float:
    """Of the onsets that sit on the sixteenth grid, the fraction landing on an
    ODD grid line (a half-beat-offset position that disappears when the tempo is
    halved). A double-tempo error leaves these lines nearly empty - the real
    notes are on the even lines and only jitter strays onto the odd ones - whereas
    a genuinely fast piece populates them with real sixteenths."""
    step = period / sub
    g = np.round((onsets - phase) / step)
    err = np.abs((onsets - phase) / step - g) * step
    on = err <= tol
    if on.sum() == 0:
        return 1.0
    return float((on & (g.astype(int) % 2 != 0)).sum()) / float(on.sum())


def _fold_double_tempo(onsets: np.ndarray, period: float, phase: float,
                       conc: float, min_bpm: float = 55.0,
                       odd_thresh: float = 0.2) -> tuple[float, float, float]:
    """Halve the tempo (double ``period``) while the faster grid's odd lines carry
    only a sparse jitter tail, so a mostly-eighth-note piece detected an octave too
    fast is brought back to its true beat. Stops as soon as the odd lines are
    populated (a real fast piece) or the tempo would drop below ``min_bpm``."""
    while _odd_line_frac(onsets, period, phase) < odd_thresh and \
            60.0 / (period * 2.0) >= min_bpm:
        period *= 2.0
        conc, phase = _grid_conc(onsets, period)
    return period, phase, conc


def _grid_conc(onsets: np.ndarray, period: float, sub: int = 4,
               tol: float = 0.02, nph: int = 64) -> tuple[float, float]:
    """For a given beat ``period``, return the best achievable grid
    concentration and the phase that achieves it: the largest fraction of
    onsets landing within ``tol`` seconds of a sub-beat grid line (``sub``
    lines per beat), maximised over phase. Vectorised over the phase grid."""
    step = period / sub
    phs = np.linspace(0.0, step, nph, endpoint=False)
    g = (onsets[None, :] - phs[:, None]) / step
    err = np.abs(g - np.round(g)) * step
    conc = (err <= tol).mean(axis=1)
    i = int(conc.argmax())
    return float(conc[i]), float(phs[i])


# division grid in beats (incl. triplets) used for notation snapping
_DIVISIONS = [4, 2, 1, 0.5, 1/3, 0.25, 1/6, 0.125]


def quantize(notes: list[NoteEvent], tempo: TempoAnalysis,
             strength: float = 1.0) -> list[dict]:
    """Return a quantized *copy* (list of dicts) - raw notes stay untouched
    (§20). Each entry records the snapped start/end in beats and the ms error."""
    period = tempo.beat_period
    phase = tempo.beat_phase
    out = []
    for n in notes:
        beat = tempo.beat_at(n.start)
        end_beat = tempo.beat_at(n.end)
        qbeat, sdur = _snap(beat)
        qend, _ = _snap(end_beat)
        if qend <= qbeat:
            qend = qbeat + min(_DIVISIONS, key=lambda d: abs(d))
        # map the snapped beat back to a time through the grid (near-identity for
        # the linear fallback; drift-aware on the tracked grid)
        q_start_time = _beat_to_time(tempo, qbeat)
        err_ms = (q_start_time - n.start) * 1000.0
        out.append({
            "id": n.id, "midi": n.midi,
            "beat": qbeat, "end_beat": qend,
            "quantized_start": q_start_time,
            "error_ms": round(err_ms, 2),
            "ambiguous": abs(err_ms) > (period * 1000 * 0.12),
        })
    return out


def _beat_to_time(tempo: TempoAnalysis, beat: float) -> float:
    """Inverse of ``beat_at``: time at a given fractional beat index."""
    if tempo.grid_ts is None:
        return tempo.beat_phase + beat * tempo.beat_period
    phi = beat * 4.0 + tempo.grid_delta + tempo.grid_shift
    return float(np.interp(phi, tempo.grid_phi, tempo.grid_ts))


def _snap(beat: float) -> tuple[float, float]:
    best = round(beat)
    best_err = abs(beat - best)
    for d in _DIVISIONS:
        q = round(beat / d) * d
        e = abs(beat - q)
        if e < best_err:
            best_err, best = e, q
    return best, best_err
