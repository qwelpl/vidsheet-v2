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

try:
    import numpy as np
    _HAS_NP = True
except Exception:  # numpy is always present in practice; guard keeps import safe
    _HAS_NP = False

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
    period = tempo.beat_period
    phase = tempo.beat_phase
    beat_at = tempo.beat_at          # drift-tracking time -> fractional beat
    beats_per_measure, beat_type = tempo.time_signature

    # Choose the notation resolution from the data instead of assuming sixteenths.
    # A fast piece whose onsets concentrate on the 32nd grid clearly better than
    # the 16th grid needs 24 divisions/quarter to print those 32nds; otherwise
    # 12 is plenty and avoids fabricating 32nds from jitter. Never go finer than
    # the detected timing can carry (a 32nd below ~one video frame is noise).
    # 12 and 24 are both divisible by 3, so eighth-note triplets print cleanly in
    # either (triplet-eighth = divisions/3 ticks).
    fine_sub, divisions = 4, 12
    if len(notes) >= 16:
        starts = [n.start for n in notes]

        # Concentration is measured in BEAT space on the tracked grid, so a note's
        # fit is judged against the tempo it was played at, not a drifting global
        # one (tol = 0.02 s expressed in beats).
        bpos = [beat_at(s) for s in starts]
        tol_beat = 0.02 / period

        def _conc(sub: int) -> float:
            step = 1.0 / sub
            hits = sum(1 for b in bpos
                       if abs(b - round(b / step) * step) <= tol_beat)
            return hits / len(bpos)

        if period / 8 >= 0.028 and _conc(8) >= _conc(4) + 0.08:
            fine_sub, divisions = 8, 24   # 32nd=3, 16th=6, eighth=12, quarter=24

    per_measure = divisions * beats_per_measure

    # Decide, per quarter-note beat, whether its onsets read as straight (duple)
    # or as an eighth-note triplet. A beat is a triplet only when its onsets sit
    # on the /3 grid clearly better than on the straight /fine_sub grid - a dense
    # run that fits neither stays duple, and jitter never flips a beat to triplet.
    # Committing the whole beat to one grid stops a smooth run being shredded into
    # dotted-sixteenth+thirty-second scraps because one note got yanked to a
    # triplet position while its neighbours stayed straight (the "timing is off"
    # look on triplet-heavy pieces). Triplets print as real tuplets below.
    triplet_beats = _detect_triplet_beats(notes, beat_at, fine_sub)

    def _beat_is_triplet(beat_i: int) -> bool:
        return beat_i in triplet_beats

    def _raw_div(t: float) -> int:
        beat = beat_at(t)
        bi = int(np.floor(beat)) if _HAS_NP else int(beat // 1)
        if bi in triplet_beats:                     # snap to the eighth-triplet grid
            frac = round((beat - bi) * 3) / 3
        else:                                       # snap to the straight grid
            frac = round((beat - bi) * fine_sub) / fine_sub
        return int(round((bi + frac) * divisions))

    # Anchor to the first onset's measure so the score doesn't start with a run
    # of empty leading measures (or negative divisions when a note precedes the
    # estimated beat-0 phase). Whole-measure shift keeps the downbeat grid intact.
    offset = 0
    if notes:
        min_d = min(_raw_div(n.start) for n in notes)
        offset = (min_d // per_measure) * per_measure

    def to_div(t: float) -> int:
        return _raw_div(t) - offset

    # Triplet flag in OUTPUT division coordinates: a position's beat is the
    # quarter it falls in. The offset is a whole number of measures (hence of
    # beats), so output beat index = absolute beat index - offset/divisions.
    beat_shift = offset // divisions

    def _pos_is_triplet(out_pos: int) -> bool:
        return _beat_is_triplet(out_pos // divisions + beat_shift)

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
                    to_div, tempo.bpm, key, spell, divisions // fine_sub,
                    _pos_is_triplet)
    out.write('  </part>\n</score-partwise>\n')
    return out.getvalue()


def _detect_triplet_beats(notes, beat_at, fine_sub) -> set:
    """Return the set of quarter-beat indices whose onsets read as an eighth-note
    triplet. A beat qualifies when it carries at least two onsets that lie on the
    three-per-beat grid distinctly better than on the straight grid; everything
    else (including dense runs that fit neither cleanly) stays duple."""
    if not _HAS_NP or len(notes) < 8:
        return set()
    by_beat: dict[int, list[float]] = {}
    for n in notes:
        beat = beat_at(n.start)
        bi = int(np.floor(beat))
        by_beat.setdefault(bi, []).append(beat - bi)

    def _cost(fracs, sub):
        return float(np.mean([abs(f * sub - round(f * sub)) for f in fracs]))

    trip = set()
    for bi, fracs in by_beat.items():
        if len(fracs) < 2:
            continue
        c3 = _cost(fracs, 3)
        cs = _cost(fracs, fine_sub)
        # clear triplet fit and a real margin over the straight grid; the finer
        # /fine_sub grid is also tested so a beat that is really sixteenths is not
        # mistaken for a loose triplet.
        if c3 <= 0.12 and c3 + 0.06 < cs:
            trip.add(bi)
    return trip


def _write_measures(out, left, right, divisions, beats_per_measure, beat_type,
                    to_div, bpm, key, spell, grid_step, pos_is_triplet):
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
                     grid_step, pos_is_triplet, min_rest_div=legato)
        out.write(f'      <backup><duration>{per_measure}</duration></backup>\n')
        _write_staff(out, left, 2, m0, m1, per_measure, to_div, divisions, spell,
                     grid_step, pos_is_triplet, min_rest_div=legato)
        out.write('    </measure>\n')


def _write_staff(out, notes, staff, m0, m1, per_measure, to_div, divisions,
                 spell, grid_step, pos_is_triplet, min_rest_div=2):
    # Gather onsets in this measure, grouped by start division (chords). A single
    # MusicXML voice is monophonic, so overlapping notes are flattened: each note
    # is spaced to the NEXT onset, guaranteeing every onset lands on its true
    # division (a held note is truncated rather than shoved past the next attack,
    # which is what previously pushed everything late and overflowed the measure).
    #
    # A chord is distinct, non-adjacent pitches at one instant. When rounding to
    # the grid fuses a step-neighbour (<=2 semitones, incl. a same-pitch repeat)
    # into one position, it is a melodic run tone, not a chord tone, so shift it
    # to the next grid cell to keep it a separate note rather than a cluster/
    # unison chord. Wider intervals rounding together stay a real chord.
    events: dict[int, list] = {}
    for n in sorted(notes, key=lambda n: n.start):
        sd = to_div(n.start)
        if not (m0 <= sd < m1):
            continue
        pos = sd - m0
        while any(abs(x.midi - n.midi) <= 2 for x in events.get(pos, ())) \
                and pos + grid_step < per_measure:
            pos += grid_step
        events.setdefault(pos, []).append(n)
    positions = sorted(events)

    # First pass: lay out (pos, dur, chord) segments and the rests between them,
    # exactly as the straight layout did; the tuplet wrapping is a second pass so
    # grouping logic stays separate from spacing/legato logic.
    segs: list[dict] = []  # {pos, dur, chord|None}

    def _add_rest(pos, dur):
        if dur > 0:
            segs.append({"pos": pos, "dur": dur, "chord": None})

    cursor = 0
    for k, pos in enumerate(positions):
        if pos > cursor:
            _add_rest(cursor, pos - cursor)
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
        # Snap the printed length to a readable value: a tuplet unit inside a
        # triplet beat, a clean straight value (power-of-two + single dot) else.
        dur = _clean_dur(dur, divisions, next_pos - pos,
                         triplet=pos_is_triplet(m0 + pos))
        segs.append({"pos": pos, "dur": dur, "chord": chord})
        cursor = pos + dur
        if cursor < next_pos:                       # genuine gap stays a rest
            _add_rest(cursor, next_pos - cursor)
            cursor = next_pos
    if cursor < per_measure:
        _add_rest(cursor, per_measure - cursor)

    # Second pass: emit, opening a <tuplet> bracket on the first triplet segment
    # of a beat and closing it on the last segment that ends within that beat.
    # Every note/rest inside gets the 3:2 <time-modification> so durations stay
    # mathematically exact and the measure still sums to a whole number of beats.
    open_tuplet_end = -1  # absolute end division of the currently open tuplet
    for si, seg in enumerate(segs):
        abspos = m0 + seg["pos"]
        trip = pos_is_triplet(abspos) and _is_tuplet_dur(seg["dur"], divisions)
        start_t = stop_t = False
        if trip:
            if seg["pos"] >= open_tuplet_end - m0 or open_tuplet_end < 0:
                # start a new tuplet spanning this beat
                beat_start = (abspos // divisions) * divisions
                open_tuplet_end = beat_start + divisions
                start_t = True
            # does the next segment continue this same beat as a tuplet?
            nxt = segs[si + 1] if si + 1 < len(segs) else None
            cont = (nxt is not None and pos_is_triplet(m0 + nxt["pos"])
                    and _is_tuplet_dur(nxt["dur"], divisions)
                    and (m0 + nxt["pos"]) < open_tuplet_end)
            if not cont:
                stop_t = True
                open_tuplet_end = -1
        if seg["chord"] is None:
            if trip:
                _write_rest(out, seg["dur"], staff, divisions,
                            triplet=True, tup_start=start_t, tup_stop=stop_t)
            else:
                # split a straight gap into clean rest values on metric
                # boundaries instead of one rest whose type lies about its length
                _emit_clean_rests(out, seg["pos"], seg["dur"], staff, divisions)
        else:
            for i, n in enumerate(seg["chord"]):
                _write_pitch(out, n, seg["dur"], staff, divisions, spell,
                             is_chord=(i > 0), triplet=trip,
                             tup_start=(start_t and i == 0),
                             tup_stop=(stop_t and i == 0))


def _clean_values(divisions: int) -> list[tuple[int, str, int]]:
    """Readable note durations in divisions: each power-of-two note value and its
    single-dotted (x1.5) form - the only rhythms a player expects to read. No
    double dots. Longest first."""
    q = divisions
    bases = [(4 * q, "whole"), (2 * q, "half"), (q, "quarter"),
             (max(1, q // 2), "eighth"), (max(1, q // 4), "16th")]
    if q >= 24:
        bases.append((q // 8, "32nd"))
    out: list[tuple[int, str, int]] = []
    for b, name in bases:
        out.append((b, name, 0))
        if b % 2 == 0:                       # dotted form only if it is integral
            out.append((b * 3 // 2, name, 1))
    out.sort(key=lambda x: -x[0])
    return out


def _clean_dur(dur: int, divisions: int, max_dur: int, triplet: bool = False) -> int:
    """Snap a raw duration to the nearest readable note value that still fits the
    space to the next onset, so every printed note is a clean type/dot instead of
    a ragged value whose ``<type>`` contradicts its ``<duration>``. Any remainder
    becomes a rest, keeping onsets exactly on their divisions.

    In a triplet beat the eighth-triplet unit (``divisions/3``) and its double are
    added to the candidates, so a note that fills one or two triplet slots keeps
    its true length instead of being rounded to a dotted-sixteenth."""
    cand = [v for v, _, _ in _clean_values(divisions) if v <= max_dur]
    if triplet:
        u = divisions // 3
        for tv in (u, 2 * u):
            if 0 < tv <= max_dur and tv not in cand:
                cand.append(tv)
    if not cand:
        return max(1, min(dur, max_dur))
    return min(cand, key=lambda v: abs(v - dur))


def _is_tuplet_dur(dur: int, divisions: int) -> bool:
    """True when a duration is an eighth-triplet unit (one or two slots) that must
    print with a 3:2 time-modification. A full-beat (quarter) or longer note in a
    triplet beat is an ordinary straight value and stays untupleted."""
    u = divisions // 3
    return u > 0 and dur in (u, 2 * u)


def _tuplet_type(dur: int, divisions: int) -> str:
    """Printed note type for a tuplet duration: the straight type whose value is
    1.5x the tuplet duration (a triplet eighth prints as an eighth, a note of two
    triplet slots as a quarter)."""
    return _note_value(dur * 3 // 2, divisions)[0]


def _note_value(dur: int, divisions: int) -> tuple[str, int]:
    """Map a (already cleaned) duration to a (type, dots) pair. Matches the exact
    readable values; a non-matching residual falls back to the nearest base type
    with no dot (never a double dot)."""
    for v, name, dots in _clean_values(divisions):
        if dur == v:
            return name, dots
    for v, name, _ in _clean_values(divisions):
        if dur >= v:
            return name, 0
    return "16th", 0


def _dur_type(dur: int, divisions: int) -> str:
    return _note_value(dur, divisions)[0]


def _time_mod_xml(indent: str) -> str:
    return (f'{indent}<time-modification><actual-notes>3</actual-notes>'
            f'<normal-notes>2</normal-notes><normal-type>eighth</normal-type>'
            f'</time-modification>\n')


def _tuplet_notations(tup_start: bool, tup_stop: bool) -> str:
    inner = ''
    if tup_start:
        inner += '<tuplet type="start" bracket="yes"/>'
    if tup_stop:
        inner += '<tuplet type="stop"/>'
    if not inner:
        return ''
    return f'        <notations>{inner}</notations>\n'


def _write_pitch(out, n, dur, staff, divisions, spell, is_chord,
                 triplet=False, tup_start=False, tup_stop=False):
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
    if triplet:
        typ, dots = _tuplet_type(dur, divisions), 0
    else:
        typ, dots = _note_value(dur, divisions)
    out.write(f'        <type>{typ}</type>\n')
    out.write('        <dot/>\n' * dots)
    if triplet:
        out.write(_time_mod_xml('        '))
    out.write(f'        <staff>{staff}</staff>\n')
    out.write(_tuplet_notations(tup_start, tup_stop))
    out.write('      </note>\n')


def _emit_clean_rests(out, pos, dur, staff, divisions):
    """Emit a straight gap as one or more clean rest values, each sitting on a
    boundary that matches its length, so no rest's ``<type>`` contradicts its
    duration. Every grid gap is a multiple of the smallest clean value, so this
    always terminates exactly."""
    vals = [v for v, _, _ in _clean_values(divisions)]  # longest first
    cur, end = pos, pos + dur
    while cur < end:
        rem = end - cur
        chosen = None
        for v in vals:                 # prefer a value aligned to its own grid
            if v <= rem and cur % v == 0:
                chosen = v
                break
        if chosen is None:             # not aligned: take the largest that fits
            for v in vals:
                if v <= rem:
                    chosen = v
                    break
        if chosen is None:             # sub-value sliver (shouldn't occur)
            chosen = rem
        _write_rest(out, chosen, staff, divisions)
        cur += chosen


def _write_rest(out, dur, staff, divisions,
                triplet=False, tup_start=False, tup_stop=False):
    if triplet:
        typ, dots = _tuplet_type(dur, divisions), 0
    else:
        typ, dots = _note_value(dur, divisions)
    out.write('      <note>\n        <rest/>\n')
    out.write(f'        <duration>{dur}</duration>\n')
    out.write(f'        <voice>{staff}</voice>\n')
    out.write(f'        <type>{typ}</type>\n')
    out.write('        <dot/>\n' * dots)
    if triplet:
        out.write(_time_mod_xml('        '))
    out.write(f'        <staff>{staff}</staff>\n')
    out.write(_tuplet_notations(tup_start, tup_stop))
    out.write('      </note>\n')
