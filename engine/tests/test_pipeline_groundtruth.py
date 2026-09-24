"""End-to-end ground-truth regression (§65, §66).

A synthetic Synthesia clip is generated from a known note list, analysed, and
scored against that ground truth. These thresholds are the regression guard:
accuracy must never silently drop below them.
"""
import os
import tempfile

import pytest

from synthesia import analyze, Options
from synthesia.synth import (generate_video, demo_piece, scale_piece,
                             SynthConfig, SynthNote)
from synthesia.keyboard import build_geometry, temporal_median
from synthesia import videoio


def _run(notes, cfg):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "clip.mp4")
    truth = generate_video(path, notes, cfg)
    result = analyze(path, Options.for_preset("maximum"), truth=truth)
    return result


def test_geometry_88_key():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "clip.mp4")
    cfg = SynthConfig()
    generate_video(path, scale_piece(), cfg)
    meta = videoio.probe(path)
    frames = [fr.image.copy() for fr in videoio.stream_frames(meta, step=5)][:40]
    geom = build_geometry(temporal_median(frames))
    assert geom.low_midi == 21 and geom.high_midi == 108
    assert len(geom.lanes) == 88
    assert geom.confidence > 0.9


def test_demo_piece_accuracy():
    result = _run(demo_piece(), SynthConfig())
    g = result.ground_truth
    assert g["recall"] == 1.0
    assert g["precision"] >= 0.97
    assert g["f1"] >= 0.98
    assert g["mean_onset_error_ms"] < 20      # sub-frame at 60 fps
    assert g["mean_duration_error_ms"] < 25


def test_chromatic_scale_all_pitches():
    notes = [SynthNote(60 + i, 1.0 + i * 0.3, 1.0 + i * 0.3 + 0.28, "right", 90)
             for i in range(13)]
    result = _run(notes, SynthConfig())
    g = result.ground_truth
    assert g["recall"] == 1.0
    assert g["false_positive"] == 0


def test_repeated_notes_stay_distinct():
    # five repeats of the same pitch must not merge into one (§12)
    notes = [SynthNote(64, 1.0 + i * 0.4, 1.0 + i * 0.4 + 0.3, "right", 95)
             for i in range(5)]
    result = _run(notes, SynthConfig())
    same = [n for n in result.notes if n.midi == 64]
    assert len(same) == 5


def test_30fps_source():
    cfg = SynthConfig(fps=30.0)
    result = _run(demo_piece(), cfg)
    g = result.ground_truth
    assert g["recall"] >= 0.97
    assert g["mean_onset_error_ms"] < 35


def test_no_hallucinated_notes_on_silence():
    # a clip with a single note must not invent extras (§51)
    notes = [SynthNote(72, 1.0, 1.5, "right", 100)]
    result = _run(notes, SynthConfig())
    assert len(result.notes) == 1
    assert result.notes[0].midi == 72
