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

    bpm = 60.0 / best_period
    conf = float(np.clip(best_conc, 0.2, 0.98))

    beats = list(np.arange(phase, onsets[-1] + best_period, best_period))
    seg = TempoSegment(start=float(onsets[0]), bpm=float(bpm), beats=[float(b) for b in beats])
    return TempoAnalysis(bpm=float(bpm), beat_period=float(best_period),
                         beat_phase=float(phase), time_signature=(4, 4),
                         segments=[seg], confidence=conf)


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
        beat = (n.start - phase) / period
        end_beat = (n.end - phase) / period
        qbeat, sdur = _snap(beat)
        qend, _ = _snap(end_beat)
        if qend <= qbeat:
            qend = qbeat + min(_DIVISIONS, key=lambda d: abs(d))
        q_start_time = phase + qbeat * period
        err_ms = (q_start_time - n.start) * 1000.0
        out.append({
            "id": n.id, "midi": n.midi,
            "beat": qbeat, "end_beat": qend,
            "quantized_start": q_start_time,
            "error_ms": round(err_ms, 2),
            "ambiguous": abs(err_ms) > (period * 1000 * 0.12),
        })
    return out


def _snap(beat: float) -> tuple[float, float]:
    best = round(beat)
    best_err = abs(beat - best)
    for d in _DIVISIONS:
        q = round(beat / d) * d
        e = abs(beat - q)
        if e < best_err:
            best_err, best = e, q
    return best, best_err
