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

    # The beat is an integer multiple of the piece's dominant pulse. A plain
    # grid search is degenerate - _grid_score rewards onsets near ANY grid line,
    # so a finer (faster) grid always fits at least as well and the search runs
    # off to the bpm ceiling, doubling or quadrupling the true tempo (§19). So:
    # find the modal inter-onset interval (the pulse), then pick the multiple of
    # it whose bpm lands in a musical range, breaking ties by grid fit.
    bins = np.arange(0.04, 2.0, 0.02)
    hist, edges = np.histogram(iois, bins=bins)
    pulse = float(edges[int(hist.argmax())] + 0.01)  # centre of the modal bin

    cands = []
    for m in (1, 2, 3, 4, 6):
        period = pulse * m
        bpm = 60.0 / period
        if 65.0 <= bpm <= 170.0:
            cands.append(period)
    if cands:
        # grid fit barely separates octaves on a rubato piece, so among the
        # candidates that fit within a tolerance of the best, take the FASTEST
        # (smallest period): straight eighths are far more common than an
        # all-triplet or half-speed reading, and a faster beat keeps the
        # notation grid cells small enough that genuinely separate notes never
        # round together into one chord.
        best_fit = max(_grid_score(onsets, p) for p in cands)
        best_period = min(p for p in cands
                          if _grid_score(onsets, p) >= best_fit - 0.01)
    else:
        # pulse outside any sane octave: fall back to a bounded grid search
        best_period, best = 0.5, -1.0
        for bpm in np.arange(65, 170.5, 0.5):
            period = 60.0 / bpm
            s = _grid_score(onsets, period)
            if s > best:
                best, best_period = s, period
    best_score = _grid_score(onsets, best_period)
    phase = _best_phase(onsets, best_period)
    bpm = 60.0 / best_period
    conf = float(np.clip(best_score, 0.2, 0.98))

    beats = list(np.arange(phase, onsets[-1] + best_period, best_period))
    seg = TempoSegment(start=float(onsets[0]), bpm=float(bpm), beats=[float(b) for b in beats])
    return TempoAnalysis(bpm=float(bpm), beat_period=float(best_period),
                         beat_phase=float(phase), time_signature=(4, 4),
                         segments=[seg], confidence=conf)


def _grid_score(onsets: np.ndarray, period: float) -> float:
    phase_frac = ((onsets / period) % 1.0)
    # reward onsets near any sixteenth-grid position (0, 1/4, 1/2, 3/4). The old
    # score rewarded only beats and eighths, so a run of real sixteenths scored
    # HIGHER at half-tempo (where they land on eighths) than at the true tempo
    # (where they land on the unrewarded quarter positions) - the main driver of
    # the octave error that fused fast notes into chords.
    err = np.abs(phase_frac - np.round(phase_frac * 4.0) / 4.0)
    return float(1.0 - 4.0 * err.mean())


def _best_phase(onsets: np.ndarray, period: float) -> float:
    best_phase, best = 0.0, -1.0
    for p in np.linspace(0, period, 24, endpoint=False):
        frac = (((onsets - p) / period) % 1.0)
        err = np.minimum(frac, 1 - frac)
        score = -err.mean()
        if score > best:
            best, best_phase = score, p
    return float(onsets[0] - ((onsets[0] - best_phase) % period))


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
