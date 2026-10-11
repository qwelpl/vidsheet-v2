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


def _run(notes, cfg, detector="auto"):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "clip.mp4")
    truth = generate_video(path, notes, cfg)
    opts = Options.for_preset("maximum")
    opts.detector = detector
    result = analyze(path, opts, truth=truth)
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


def test_keyboard_band_spans_black_keys():
    # The keyboard band must reach up past the black-key region, not stop at the
    # white aprons below it. A row MEAN is dragged down through the black-key band
    # (dark blacks interleaved) and stops short; a high per-row percentile follows
    # the white-key columns up to the strike line. Also survive a dark bottom
    # letterbox that would zero a bottom-row brightness anchor.
    import numpy as np
    from synthesia.keyboard import detect_keyboard_band

    h, w = 300, 520
    img = np.zeros((h, w), np.uint8)          # dark roll + letterbox
    top_true, bottom_true = 150, 280
    img[top_true:bottom_true, :] = 245        # white keys, full band
    # black keys occupy the UPPER half of the band, every other 20px column
    for x in range(0, w, 40):
        img[top_true:top_true + 65, x:x + 20] = 15
    # dark bottom letterbox below the keyboard
    img[bottom_true:, :] = 0

    top, bottom = detect_keyboard_band(img)
    assert top <= top_true + 8, f"band top {top} cut into black keys ({top_true})"
    assert abs(bottom - bottom_true) <= 8


def test_keylight_far_cover_rejects_strike_line_glow():
    # A key-highlight press fills the whole key body, so its FAR half (away from
    # the strike line) is as solidly covered as the near half. A strike-line glow
    # or particle-smoke halo pools at the near edge and decays before the far half.
    # ``_far_cover`` scores the far half only, so a top-weighted halo collapses to
    # ~0 while a real press stays high - the discriminator carries no colour- or
    # video-specific constant, only the key's own midline (§55).
    import numpy as np
    from synthesia.keylight import _far_cover

    press = np.ones((20, 6), bool)                     # solid, full key body
    assert _far_cover(press) == 1.0

    halo = np.zeros((20, 6), bool)
    halo[:7, :] = True                                 # colour only near the top
    assert _far_cover(halo) == 0.0                     # far half empty -> rejected

    # a genuine press whose near rows are darkened by the strike-line edge must
    # still register: the far half is what counts.
    edge_dimmed = np.ones((20, 6), bool)
    edge_dimmed[:3, :] = False
    assert _far_cover(edge_dimmed) == 1.0


def test_keylight_repeats_split_on_coverage_dip():
    # A key-highlight repeat re-strikes a key that only *dims* between hits (its
    # colour coverage dips and recovers without ever fully clearing) and whose
    # bass attack may be too quiet for a global audio onset. The illumination
    # envelope alone must separate the repeats; a steadily-held key must not
    # split (§12).
    from synthesia.keylight import _restrike_times

    fps = 60.0
    dt = 1.0 / fps
    hit = [0.2, 0.5, 0.9, 0.9, 0.9, 0.5, 0.2]      # one strike, dips at the seam
    env = hit * 4                                    # four repeats, no full unlit
    t = [k * dt for k in range(len(env))]
    splits = _restrike_times(env, t, 0, len(env) - 1, dt, 2)
    assert len(splits) == 3                          # 4 notes -> 3 boundaries

    held = [0.9] * 40                                # one sustained note
    t2 = [k * dt for k in range(len(held))]
    assert _restrike_times(held, t2, 0, len(held) - 1, dt, 2) == []


def test_keylight_onset_accuracy_and_hand_sync():
    # Key-highlight path must place onsets sub-frame and keep a two-hand chord
    # tight, not skewed by which frame each key happened to light on (§10).
    left = [SynthNote(48 + i * 4, 1.0, 1.8, "left", 80) for i in range(3)]   # chord
    right = [SynthNote(72 + i * 4, 1.0, 1.8, "right", 90) for i in range(3)]  # chord
    result = _run(left + right, SynthConfig(), detector="keylight")
    g = result.ground_truth
    assert g["recall"] == 1.0
    assert g["mean_onset_error_ms"] < 20
    # every note nominally at t=1.0 should land within a frame of the others
    starts = sorted(n.start for n in result.notes)
    assert starts[-1] - starts[0] < 0.02


def test_musicxml_measures_are_time_complete():
    # Every voice in every measure must sum to exactly one measure of divisions - 
    # overlapping/held notes flattened to a single voice must not overflow or
    # push later onsets off their beat (the "timing really off" regression).
    import xml.etree.ElementTree as ET
    from synthesia.exporters import _build_musicxml
    from synthesia.model import NoteEvent, Hand
    from synthesia.tempo import TempoAnalysis

    tempo = TempoAnalysis(120.0, 0.5, 0.0, (4, 4), [], 0.9)
    # dense overlaps: a held left-hand note under a right-hand run, plus a chord
    notes = [
        NoteEvent(1, 48, 0.0, 2.0, Hand.LEFT, 80),      # 2s held bass
        NoteEvent(2, 60, 0.0, 0.5, Hand.RIGHT, 90),
        NoteEvent(3, 64, 0.0, 0.5, Hand.RIGHT, 90),     # chord w/ 60
        NoteEvent(4, 67, 0.5, 0.9, Hand.RIGHT, 90),     # overlaps into next onset
        NoteEvent(5, 72, 0.75, 1.5, Hand.RIGHT, 90),
        NoteEvent(6, 50, 2.0, 4.0, Hand.LEFT, 80),
    ]
    xml = _build_musicxml(notes, tempo)
    root = ET.fromstring(xml)
    divisions = int(root.find(".//attributes/divisions").text)
    beats = int(root.find(".//attributes/time/beats").text)
    per = divisions * beats
    for meas in root.iter("measure"):
        sums = {1: 0, 2: 0}
        for note in meas.findall("note"):
            if note.find("chord") is not None:
                continue
            st = note.find("staff")
            sums[int(st.text) if st is not None else 1] += int(note.find("duration").text)
        for staff, tot in sums.items():
            assert tot == per, f"measure {meas.get('number')} staff {staff}: {tot} != {per}"


def test_no_hallucinated_notes_on_silence():
    # a clip with a single note must not invent extras (§51)
    notes = [SynthNote(72, 1.0, 1.5, "right", 100)]
    result = _run(notes, SynthConfig())
    assert len(result.notes) == 1
    assert result.notes[0].midi == 72
