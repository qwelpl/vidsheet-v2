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
    # the server ships this dict as JSON; a stray numpy scalar breaks the UI
    import json
    json.dumps(result.to_dict())
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


def test_white_boundaries_survive_printed_key_labels():
    # Note-name letters / fingering numbers printed ON the keys read as dark
    # columns in the lower keyboard strip and used to spawn phantom separators,
    # halving the apparent key width and shattering the octave pattern. The
    # boundary finder must lock to the keys' own dominant period and ignore the
    # off-grid intruders.
    import numpy as np
    from synthesia.keyboard import _find_white_boundaries, _dominant_period

    w = 660
    period = 60
    strip = np.full((20, w), 240, np.uint8)         # bright white-key bodies
    for x in range(period, w, period):              # true separators every 60 px
        strip[:, x - 1:x + 1] = 20
    # dark labels printed mid-key at irregular offsets (the contamination)
    for cx in range(period // 2, w, period):
        strip[8:16, cx - 4:cx + 4] = 30

    assert abs(_dominant_period(strip.max() - strip.mean(axis=0)) - period) <= 2
    b = _find_white_boundaries(strip)
    assert 9 <= len(b) <= 11, f"expected ~10 boundaries, got {len(b)}"
    assert abs(float(np.median(np.diff(b))) - period) <= 3


def test_black_group_phase_anchors_C():
    # The octave phase (which white key is C) must come from the black-key 2/3
    # grouping: the two-group (C#,D#) is the lone narrow gap bracketed by the two
    # wide gaps that cross the black-less E-F and B-C white pairs. Its left key is
    # C#, so the white key just left of it is C.
    import numpy as np
    from synthesia.keyboard import _phase_from_black_groups, _WHITE_SEMITONES

    P = 60.0
    # two octaves of white keys C..B, C index 0 and 7
    white = np.arange(0, 14) * P + 40.0
    # black keys at C#,D#,F#,G#,A# in each octave (offsets within the octave)
    offs = [0.6, 1.6, 3.6, 4.6, 5.6]
    black = np.array(sorted(white[0] + o * P for base in (0, 7)
                            for o in [x + base for x in offs]))
    phase = _phase_from_black_groups(white, black, P)
    # white[0] is C, so its pitch class must be C (semitone 0)
    assert phase is not None and _WHITE_SEMITONES[phase % 7] == 0


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


def test_dim_keyboard_band_grows_to_key_tops():
    # Grey keys under a bright hit-line glow: the brightness tests break inside
    # the black-key zone, so the band stopped at the white aprons below the
    # blacks (rows 449-472 on the real clip). Key texture (sharp key edges) must
    # carry it up to the key tops and down to the frame edge, and stop at the
    # smooth glow above.
    import numpy as np
    from synthesia.keyboard import _grow_band_by_texture

    h, w = 480, 854
    g = np.full((h, w), 10, np.uint8)
    for y in range(340, 386):                        # glow ramping into the line
        g[y, :] = int(10 + 235 * ((y - 340) / 45.0) ** 2)
    g[386:, :] = 70                                  # dim grey white keys
    for x in np.arange(10, w, 16.5):                 # white separators
        g[386:, int(x)] = 25
    for x in np.arange(5, w, 16.5 * 7 / 5):          # black keys
        g[386:448, int(x):int(x) + 9] = 15
    top, bottom = _grow_band_by_texture(g, 449, 472)
    assert top == 386, top
    assert bottom == h - 1


def test_boundary_lattice_drops_strays_and_fills_gaps():
    # A mostly-regular separator comb with a stray board-edge peak, a black-key
    # edge read as a separator, and two missing separators must come back as the
    # one even lattice - a stray must not shift the keys after it.
    import numpy as np
    from synthesia.keyboard import _fit_boundary_lattice

    p = 16.8
    true = 10.0 + p * np.arange(50)
    peaks = np.sort(np.concatenate([np.delete(true, [3, 20, 21]), [1.5, 39.1]]))
    b = _fit_boundary_lattice(peaks, float(np.median(np.diff(peaks))))
    assert len(b) == len(true)
    assert np.max(np.abs(b - true)) < 1.0


def test_static_glow_band_found_above_strike():
    # A wide static glow sitting on the strike line is found from the temporal
    # median; a clean dark roll yields None (nothing changes there).
    import numpy as np
    from synthesia.pipeline import _static_glow_top
    from synthesia.keyboard import _fallback_88

    h, w = 480, 854
    geom = _fallback_88(w, h, 386, 479)
    clean = np.full((h, w, 3), 8, np.uint8)
    clean[386:] = 200
    assert _static_glow_top(clean, geom) is None
    glow = clean.copy()
    for y in range(340, 386):
        glow[y] = int(8 + 237 * ((y - 340) / 45.0) ** 2)
    top = _static_glow_top(glow, geom)
    assert top is not None and 330 <= top <= 352, top


def test_audio_skew_tracks_drift_across_the_song():
    # The picture-vs-audio offset can drift between sections. One song-wide
    # median then drags a whole section's notes onto attacks tens of ms off;
    # the skew must be measured locally so each section stays on its own bars.
    import numpy as np
    from synthesia.model import NoteEvent
    from synthesia.pipeline import _align_audio_to_visual

    vis = np.arange(0.5, 60.0, 0.25)
    skew = np.where(vis < 20, -0.017, 0.037)        # section-dependent offset
    audio = vis - skew
    notes = [NoteEvent(id=i, midi=60, start=float(s), end=float(s) + 0.1)
             for i, s in enumerate(vis)]
    aligned = np.sort(np.asarray(_align_audio_to_visual(notes, audio)))
    early = np.abs(aligned[vis < 14] - vis[vis < 14])
    late = np.abs(aligned[vis > 26] - vis[vis > 26])
    assert early.max() < 0.005 and late.max() < 0.005


def test_snap_keeps_precise_visual_onsets_off_neighbouring_attacks():
    # Clean bars already time notes to a few ms. A note whose own soft attack
    # the flux picker missed must not jump to an unrelated attack 70 ms away
    # (another sound with no bar of its own); notes that do have an attack still
    # snap onto it.
    import numpy as np
    from synthesia.model import NoteEvent
    from synthesia.audio import snap_onsets

    starts = np.arange(1.0, 11.0, 0.2)              # 50 notes, 0.2 s apart
    audio = np.sort(np.append(starts + 0.003, 5.16))  # attacks agree to 3 ms
    notes = [NoteEvent(id=i, midi=60 + (i % 2) * 7, start=float(s), end=float(s) + 0.05)
             for i, s in enumerate(starts)]
    want = {id(n): s + 0.003 for n, s in zip(notes, starts)}
    lone = NoteEvent(id=99, midi=72, start=5.09, end=5.14)   # attack missed
    notes.append(lone)
    snap_onsets(notes, audio)
    assert abs(lone.start - 5.09) < 1e-9
    assert all(abs(n.start - want[id(n)]) < 1e-6 for n in notes if n is not lone)
