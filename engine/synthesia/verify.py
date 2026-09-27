"""Self-verification, accuracy metrics and automatic error search
(§24, §25, §48, §67).

Two independent checks:

  * ``visual_compare`` re-derives, for sampled frames, which lanes the
    reconstruction says should hold a bar and compares that against what was
    actually detected in the frame. Disagreements localise missing notes and
    false positives (§24).
  * ``search_errors`` scans the note list for the suspicious patterns listed in
    §48 (1-frame notes, near-duplicate re-attacks, octave jumps, unsupported
    events...) and flags them for review - never deletes them.

Nothing here fabricates or "rounds up" accuracy; every number is measured
(§25, §68).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import FrameObservation
from .keyboard import KeyboardGeometry
from .model import NoteEvent, ReconstructionReport, midi_to_name
from .renderer import predicted_presence


@dataclass
class FrameDiff:
    time: float
    matched: int
    missing: list[int]        # midis present in original, absent in recon
    false_positive: list[int]  # midis in recon, absent in original


def visual_compare(notes: list[NoteEvent], observations: list[FrameObservation],
                   geom: KeyboardGeometry, v: float,
                   iou_tol: float = 0.35) -> list[FrameDiff]:
    diffs: list[FrameDiff] = []
    for obs in observations:
        pred = predicted_presence(notes, geom, v, obs.time)
        detected = {m: (min(r.y_top for r in runs), max(r.y_bottom for r in runs))
                    for m, runs in obs.lane_runs.items()}
        matched = 0
        missing, fp = [], []
        for m, span in detected.items():
            if m in pred and _overlap(span, pred[m]) >= iou_tol:
                matched += 1
            else:
                missing.append(m)
        for m in pred:
            if m not in detected:
                fp.append(m)
        diffs.append(FrameDiff(obs.time, matched, missing, fp))
    return diffs


def _overlap(a, b) -> float:
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def search_errors(notes: list[NoteEvent], fps: float,
                  min_dur: float = 0.02) -> None:
    """Annotate notes in place with §48 issue flags."""
    frame_dt = 1.0 / fps
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.midi, []).append(n)

    for n in notes:
        if n.duration < max(min_dur, frame_dt * 0.75):
            n.flag("very_short")
            n.duration_confidence = min(n.duration_confidence, 0.4)

    # near-duplicate re-attacks: same pitch within a few ms (§12/§48)
    for m, group in by_pitch.items():
        group.sort(key=lambda x: x.start)
        for a, b in zip(group, group[1:]):
            if b.start - a.start < 0.03:
                b.flag("possible_duplicate")
                b.detection_confidence = min(b.detection_confidence, 0.5)
            if a.end > b.start + 0.005:
                b.flag("overlaps_same_pitch")

    # unexpected octave jumps within one hand/voice (context anomaly)
    ordered = sorted(notes, key=lambda x: x.start)
    for i in range(1, len(ordered) - 1):
        prev, cur, nxt = ordered[i - 1], ordered[i], ordered[i + 1]
        if abs(cur.midi - prev.midi) == 12 and abs(cur.midi - nxt.midi) == 12 \
                and prev.midi == nxt.midi:
            cur.flag("octave_jump_isolated")


def build_report(notes: list[NoteEvent], diffs: list[FrameDiff],
                 geom: KeyboardGeometry, onset_uncertainty_ms: float,
                 ground_truth: bool = False) -> ReconstructionReport:
    r = ReconstructionReport()
    r.notes_total = len(notes)
    r.high_confidence = sum(1 for n in notes if n.overall_confidence >= 0.8)
    r.medium_confidence = sum(1 for n in notes if 0.55 <= n.overall_confidence < 0.8)
    r.low_confidence = sum(1 for n in notes if n.overall_confidence < 0.55)
    r.needs_review = sum(1 for n in notes if n.issues)
    r.possible_duplicates = sum(1 for n in notes if "possible_duplicate" in n.issues)
    r.timing_conflicts = sum(1 for n in notes if n.timing_confidence < 0.55)
    r.pitch_conflicts = sum(1 for n in notes if n.pitch_confidence < 0.6)

    # missing-note estimate from the visual comparison
    persistent_missing = 0
    for d in diffs:
        persistent_missing += len(d.missing)
    r.possible_missing = int(persistent_missing / max(1, len(diffs)) * 2)

    r.mean_onset_uncertainty_ms = round(onset_uncertainty_ms, 2)
    r.keyboard_low_midi = geom.low_midi
    r.keyboard_high_midi = geom.high_midi
    r.visual_verified = len(diffs) > 0
    r.ground_truth_compared = ground_truth
    match_total = sum(d.matched for d in diffs)
    det_total = match_total + persistent_missing
    ratio = match_total / det_total if det_total else 1.0
    r.notes = (f"Visual agreement {ratio*100:.1f}% over {len(diffs)} sampled frames."
               if diffs else "No visual verification performed.")
    return r


def compare_ground_truth(recon: list[NoteEvent], truth: list[dict],
                         onset_tol: float = 0.05) -> dict:
    """Compare against known MIDI ground truth (§65). Greedy match by pitch and
    nearest onset within tolerance."""
    truth_by_pitch: dict[int, list[dict]] = {}
    for t in truth:
        truth_by_pitch.setdefault(t["midi"], []).append(dict(t, used=False))
    tp = 0
    onset_err = []
    dur_err = []
    for n in sorted(recon, key=lambda x: x.start):
        cands = truth_by_pitch.get(n.midi, [])
        best, best_d = None, onset_tol
        for c in cands:
            if c["used"]:
                continue
            d = abs(c["start"] - n.start)
            if d < best_d:
                best_d, best = d, c
        if best is not None:
            best["used"] = True
            tp += 1
            onset_err.append(abs(best["start"] - n.start))
            dur_err.append(abs((best["end"] - best["start"]) - n.duration))
    fp = len(recon) - tp
    fn = sum(1 for lst in truth_by_pitch.values() for c in lst if not c["used"])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "mean_onset_error_ms": round(float(np.mean(onset_err)) * 1000, 2) if onset_err else None,
        "mean_duration_error_ms": round(float(np.mean(dur_err)) * 1000, 2) if dur_err else None,
    }
