"""Key-signature estimation and enharmonic spelling.

The transcriber only knows MIDI pitch classes (black keys are always spelled as
sharps). To print a readable score we need two things the raw notes don't carry:

1. the key signature (how many sharps/flats to hang on the staff), and
2. per-note spelling consistent with that key (a D# in Eb major is really Eb).

Key is estimated with the Krumhansl-Schmuckler profile correlation over a
duration-weighted pitch-class histogram. Spelling walks the line of fifths so
every pitch class gets the name nearest the key's diatonic window.
"""
from __future__ import annotations

from dataclasses import dataclass

# Krumhansl-Kessler key profiles (major / minor), rooted on C / A respectively
# as published; rotated for each candidate tonic during correlation.
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                  5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                  4.75, 3.98, 2.69, 3.34, 3.17]

_LETTERS = "FCGDAEB"            # line-of-fifths order of letter names
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class KeyEstimate:
    fifths: int                 # -7..7, signed count for <fifths>
    mode: str                   # "major" | "minor"
    tonic_pc: int               # 0..11, pitch class of the tonic


def _correlate(hist: list[float], profile: list[float], shift: int) -> float:
    rotated = [profile[(i - shift) % 12] for i in range(12)]
    n = 12
    mh = sum(hist) / n
    mp = sum(rotated) / n
    num = sum((hist[i] - mh) * (rotated[i] - mp) for i in range(n))
    dh = sum((hist[i] - mh) ** 2 for i in range(n)) ** 0.5
    dp = sum((rotated[i] - mp) ** 2 for i in range(n)) ** 0.5
    if dh == 0 or dp == 0:
        return 0.0
    return num / (dh * dp)


def _tonic_to_fifths(tonic_pc: int, mode: str) -> int:
    # Minor keys share the signature of their relative major (tonic + 3).
    pc = tonic_pc if mode == "major" else (tonic_pc + 3) % 12
    # Map pitch class to the signed signature in -5..6, preferring flats for the
    # ambiguous enharmonics (Db over C#, etc.).
    return ((7 * pc + 5) % 12) - 5


def detect_key(notes) -> KeyEstimate:
    """Estimate the key from note events (uses .midi and .duration)."""
    hist = [0.0] * 12
    for n in notes:
        weight = max(n.duration, 1e-3)
        hist[n.midi % 12] += weight
    if sum(hist) == 0:
        return KeyEstimate(fifths=0, mode="major", tonic_pc=0)

    best = (-2.0, 0, "major")
    for tonic in range(12):
        maj = _correlate(hist, _MAJOR_PROFILE, tonic)
        if maj > best[0]:
            best = (maj, tonic, "major")
        minor = _correlate(hist, _MINOR_PROFILE, tonic)
        if minor > best[0]:
            best = (minor, tonic, "minor")

    _, tonic_pc, mode = best
    return KeyEstimate(fifths=_tonic_to_fifths(tonic_pc, mode),
                       mode=mode, tonic_pc=tonic_pc)


def build_speller(fifths: int):
    """Return f(midi) -> (step, alter, octave) spelling each pitch for the key.

    Walks the line of fifths: position p has pitch class (7*p) % 12. For every
    pitch class we choose the position nearest the centre of the key's diatonic
    window ([fifths-1, fifths+5]), so diatonic notes stay natural and chromatic
    notes pick the alteration idiomatic to the key.
    """
    centre = fifths + 2
    spelling: dict[int, tuple[str, int]] = {}
    for pc in range(12):
        p0 = (7 * pc) % 12                 # a line-of-fifths position for this pc
        # Shift by multiples of 12 to land nearest the window centre.
        p = p0 + round((centre - p0) / 12) * 12
        letter = _LETTERS[(p + 1) % 7]
        alter = (p + 1) // 7            # Python // floors, so negatives give flats
        spelling[pc] = (letter, alter)

    def spell(midi: int) -> tuple[str, int, int]:
        step, alter = spelling[midi % 12]
        # Octave from the exact spelled pitch so B#/Cb cross the octave right.
        octave = (midi - _LETTER_PC[step] - alter) // 12 - 1
        return step, alter, octave

    return spell
