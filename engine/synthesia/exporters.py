"""Exporters: MIDI, CSV, JSON, MusicXML (§22, §58).

MIDI uses high PPQ so precise sub-frame timing survives (§22). Hands map to
separate channels; sustain-pedal events export as CC64 (§17). Raw note metadata
and confidences are preserved in the JSON export (§40, §58) - nothing is thrown
away after producing the MIDI.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Optional

from .model import NoteEvent, Hand, midi_to_name
from .tempo import TempoAnalysis
from .key import detect_key, build_speller

PPQ = 960


def write_midi(path: str, notes: list[NoteEvent], tempo: TempoAnalysis,
               pedal_events: Optional[list[tuple[float, bool]]] = None) -> None:
    import mido

    mid = mido.MidiFile(ticks_per_beat=PPQ)
    spb = 60.0 / tempo.bpm  # seconds per beat

    def sec_to_tick(t: float) -> int:
        return int(round(t / spb * PPQ))

    # tempo/meta track
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo.bpm), time=0))
    num, den = tempo.time_signature
    meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))

    # collect events: (tick, priority, message)
    events: list[tuple[int, int, "mido.Message"]] = []
    for n in notes:
        # MIDI data bytes are 0..127. A real piano note always is, but a bad
        # keyboard-geometry detection can hand us a lane pitch outside that range;
        # skip it rather than letting mido abort the whole export.
        if not (0 <= n.midi <= 127):
            continue
        ch = 0 if n.hand == Hand.LEFT else 1
        vel = int(min(127, max(0, n.velocity)))
        events.append((sec_to_tick(n.start), 1,
                       mido.Message("note_on", note=n.midi, velocity=vel, channel=ch)))
        events.append((sec_to_tick(n.end), 0,
                       mido.Message("note_off", note=n.midi, velocity=0, channel=ch)))
    if pedal_events:
        for t, down in pedal_events:
            events.append((sec_to_tick(t), 2,
                           mido.Message("control_change", control=64,
                                        value=127 if down else 0, channel=1)))

    events.sort(key=lambda e: (e[0], e[1]))
    track = mido.MidiTrack()
    mid.tracks.append(track)
    last = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last)
        last = tick
        track.append(msg)
    mid.save(path)


def write_csv(path: str, notes: list[NoteEvent]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pitch", "midi", "start", "end", "duration", "hand",
                    "velocity", "detection_conf", "pitch_conf", "timing_conf",
                    "duration_conf", "hand_conf", "issues"])
        for n in notes:
            w.writerow([n.id, n.name, n.midi, f"{n.start:.6f}", f"{n.end:.6f}",
                        f"{n.duration:.6f}", n.hand.value, n.velocity,
                        f"{n.detection_confidence:.3f}", f"{n.pitch_confidence:.3f}",
                        f"{n.timing_confidence:.3f}", f"{n.duration_confidence:.3f}",
                        f"{n.hand_confidence:.3f}", ";".join(n.issues)])


def write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def notes_to_json(notes: list[NoteEvent]) -> list[dict]:
    return [n.to_dict() for n in notes]


# ---------------------------------------------------------------------------
# MusicXML (§21) - partwise grand staff, quantized to the beat grid
# ---------------------------------------------------------------------------

def write_musicxml(path: str, notes: list[NoteEvent], tempo: TempoAnalysis) -> None:
    xml = _build_musicxml(notes, tempo)
    with open(path, "w") as f:
        f.write(xml)


def _build_musicxml(notes: list[NoteEvent], tempo: TempoAnalysis) -> str:
    divisions = 12  # divisions per quarter: resolves 16ths (3) and triplets (4)
    period = tempo.beat_period
    phase = tempo.beat_phase
    beats_per_measure, beat_type = tempo.time_signature

    per_measure = divisions * beats_per_measure

    def _raw_div(t: float) -> int:
        # Rhythmic quantization for notation: snap the beat position to the
        # nearest eighth / eighth-triplet / sixteenth before mapping to
        # divisions. Rounding straight to the fine division grid prints onset
        # jitter as ragged 7/12 + 5/12 values; a plain run of eighths then reads
        # as a mess instead of even eighth notes. Raw MIDI timing is untouched.
        beat = (t - phase) / period
        best = round(beat)
        for s in (2, 3, 4):           # eighths, triplets, sixteenths per beat
            q = round(beat * s) / s
            if abs(beat - q) < abs(beat - best):
                best = q
        return int(round(best * divisions))

    # Anchor to the first onset's measure so the score doesn't start with a run
    # of empty leading measures (or negative divisions when a note precedes the
    # estimated beat-0 phase). Whole-measure shift keeps the downbeat grid intact.
    offset = 0
    if notes:
        min_d = min(_raw_div(n.start) for n in notes)
        offset = (min_d // per_measure) * per_measure

    def to_div(t: float) -> int:
        return _raw_div(t) - offset

    left = sorted([n for n in notes if n.hand != Hand.RIGHT], key=lambda n: n.start)
    right = sorted([n for n in notes if n.hand == Hand.RIGHT], key=lambda n: n.start)

    key = detect_key(notes)
    spell = build_speller(key.fifths)

    out = io.StringIO()
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.write('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
              '"http://www.musicxml.org/dtds/partwise.dtd">\n')
    out.write('<score-partwise version="3.1">\n')
    out.write('  <part-list>\n'
              '    <score-part id="P1"><part-name>Piano</part-name></score-part>\n'
              '  </part-list>\n')
    out.write('  <part id="P1">\n')
    _write_measures(out, left, right, divisions, beats_per_measure, beat_type,
                    to_div, tempo.bpm, key, spell)
    out.write('  </part>\n</score-partwise>\n')
    return out.getvalue()


def _write_measures(out, left, right, divisions, beats_per_measure, beat_type,
                    to_div, bpm, key, spell):
    all_notes = left + right
    if not all_notes:
        end_div = divisions * beats_per_measure
    else:
        end_div = max(to_div(n.end) for n in all_notes)
    per_measure = divisions * beats_per_measure
    n_measures = max(1, (end_div + per_measure - 1) // per_measure)

    for m in range(n_measures):
        out.write(f'    <measure number="{m + 1}">\n')
        if m == 0:
            out.write('      <attributes>\n')
            out.write(f'        <divisions>{divisions}</divisions>\n')
            out.write(f'        <key><fifths>{key.fifths}</fifths>'
                      f'<mode>{key.mode}</mode></key>\n')
            out.write(f'        <time><beats>{beats_per_measure}</beats>'
                      f'<beat-type>{beat_type}</beat-type></time>\n')
            out.write('        <staves>2</staves>\n')
            out.write('        <clef number="1"><sign>G</sign><line>2</line></clef>\n')
            out.write('        <clef number="2"><sign>F</sign><line>4</line></clef>\n')
            out.write('      </attributes>\n')
            out.write(f'      <direction placement="above"><direction-type>'
                      f'<metronome><beat-unit>quarter</beat-unit>'
                      f'<per-minute>{int(round(bpm))}</per-minute></metronome>'
                      f'</direction-type><sound tempo="{int(round(bpm))}"/></direction>\n')
        m0, m1 = m * per_measure, (m + 1) * per_measure
        legato = max(2, divisions // 2)  # absorb trailing gaps below an eighth
        _write_staff(out, right, 1, m0, m1, per_measure, to_div, divisions, spell,
                     min_rest_div=legato)
        out.write(f'      <backup><duration>{per_measure}</duration></backup>\n')
        _write_staff(out, left, 2, m0, m1, per_measure, to_div, divisions, spell,
                     min_rest_div=legato)
        out.write('    </measure>\n')


def _write_staff(out, notes, staff, m0, m1, per_measure, to_div, divisions,
                 spell, min_rest_div=2):
    # Gather onsets in this measure, grouped by start division (chords). A single
    # MusicXML voice is monophonic, so overlapping notes are flattened: each note
    # is spaced to the NEXT onset, guaranteeing every onset lands on its true
    # division (a held note is truncated rather than shoved past the next attack,
    # which is what previously pushed everything late and overflowed the measure).
    #
    # A single key cannot sound twice at one instant: when rounding fuses two
    # strikes of the SAME pitch into one division, shift the later strike to the
    # next free division so it stays a distinct note instead of an impossible
    # unison chord. Distinct pitches that round together remain a real chord.
    events: dict[int, list] = {}
    for n in sorted(notes, key=lambda n: n.start):
        sd = to_div(n.start)
        if not (m0 <= sd < m1):
            continue
        pos = sd - m0
        while any(x.midi == n.midi for x in events.get(pos, ())) and pos + 1 < per_measure:
            pos += 1
        events.setdefault(pos, []).append(n)
    positions = sorted(events)
    cursor = 0
    for k, pos in enumerate(positions):
        if pos > cursor:
            _write_rest(out, pos - cursor, staff, divisions)
            cursor = pos
        chord = events[pos]
        next_pos = positions[k + 1] if k + 1 < len(positions) else per_measure
        note_dur = min(to_div(n.end) - to_div(n.start) for n in chord)
        dur = max(1, min(note_dur, next_pos - pos, per_measure - pos))
        # Absorb a sub-beat trailing gap (< min_rest_div) into the note as legato:
        # consecutive notes are near-contiguous in real time, and rounding their
        # end down while the next onset rounds up otherwise leaves a spurious
        # 16th-rest between them (staccato pauses peppered through the score).
        if 0 < next_pos - (pos + dur) < min_rest_div:
            dur = next_pos - pos
        for i, n in enumerate(chord):
            _write_pitch(out, n, dur, staff, divisions, spell, is_chord=(i > 0))
        cursor = pos + dur
        # a genuine gap (>= min_rest_div) to the next onset stays a rest so the
        # next onset still lands exactly on its division
        if cursor < next_pos:
            _write_rest(out, next_pos - cursor, staff, divisions)
            cursor = next_pos
    if cursor < per_measure:
        _write_rest(out, per_measure - cursor, staff, divisions)


def _dur_type(dur: int, divisions: int) -> str:
    # standard note value <= dur, measured in divisions-per-quarter units
    q = divisions
    for d, name in ((4 * q, "whole"), (2 * q, "half"), (q, "quarter"),
                    (max(1, q // 2), "eighth"), (max(1, q // 4), "16th"),
                    (max(1, q // 8), "32nd")):
        if dur >= d:
            return name
    return "32nd"


def _write_pitch(out, n, dur, staff, divisions, spell, is_chord):
    step, alter, octave = spell(n.midi)
    out.write('      <note>\n')
    if is_chord:
        out.write('        <chord/>\n')
    out.write('        <pitch>\n')
    out.write(f'          <step>{step}</step>\n')
    if alter:
        out.write(f'          <alter>{alter}</alter>\n')
    out.write(f'          <octave>{octave}</octave>\n')
    out.write('        </pitch>\n')
    out.write(f'        <duration>{dur}</duration>\n')
    out.write(f'        <voice>{staff}</voice>\n')
    out.write(f'        <type>{_dur_type(dur, divisions)}</type>\n')
    out.write(f'        <staff>{staff}</staff>\n')
    out.write('      </note>\n')


def _write_rest(out, dur, staff, divisions):
    out.write('      <note>\n        <rest/>\n')
    out.write(f'        <duration>{dur}</duration>\n')
    out.write(f'        <voice>{staff}</voice>\n')
    out.write(f'        <type>{_dur_type(dur, divisions)}</type>\n')
    out.write(f'        <staff>{staff}</staff>\n      </note>\n')
