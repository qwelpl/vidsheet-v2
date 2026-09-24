"""Core data model for the Synthesia reconstruction engine.

Every note in the reconstruction carries its full provenance: which frames it
was seen in, how it was tracked, and separate confidence values for pitch,
timing, duration and hand. Raw (unquantized) performance timing is *never*
discarded — quantization is a derived, separate representation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------

_NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Semitone offsets inside an octave that correspond to black keys.
BLACK_KEY_SEMITONES = {1, 3, 6, 8, 10}


def midi_to_name(midi: int) -> str:
    """MIDI number -> scientific pitch name, e.g. 60 -> 'C4'."""
    octave = midi // 12 - 1
    return f"{_NOTE_NAMES_SHARP[midi % 12]}{octave}"


def name_to_midi(name: str) -> int:
    """'C4' / 'C#4' -> MIDI number. Inverse of :func:`midi_to_name`."""
    name = name.strip()
    i = 1
    if len(name) > 1 and name[1] == "#":
        i = 2
    pitch_class = name[:i]
    octave = int(name[i:])
    return _NOTE_NAMES_SHARP.index(pitch_class) + (octave + 1) * 12


def is_black_key(midi: int) -> bool:
    return (midi % 12) in BLACK_KEY_SEMITONES


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VISUAL_ONLY = "visual"
    AUDIO_CONFIRMED = "audio_confirmed"
    CONFLICT = "conflict"
    CORRECTED = "corrected"


# ---------------------------------------------------------------------------
# Note event
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """A single reconstructed note.

    Times are in seconds of the source video (presentation time). ``velocity``
    is 1..127. Confidences are 0..1. Flags in ``issues`` mark anything a human
    should review; the note is still preserved (never silently dropped).
    """

    id: int
    midi: int
    start: float
    end: float
    hand: Hand = Hand.UNKNOWN
    velocity: int = 80

    # confidences, kept independent so the UI can surface *what* is uncertain
    detection_confidence: float = 1.0
    pitch_confidence: float = 1.0
    timing_confidence: float = 1.0
    duration_confidence: float = 1.0
    hand_confidence: float = 1.0
    velocity_observed: bool = False

    # provenance
    source_frames: list[int] = field(default_factory=list)
    track_id: Optional[int] = None
    color_cluster: Optional[int] = None  # theme colour cluster (hand hint, §16)
    lane_x: Optional[float] = None  # sub-pixel horizontal centre in the roll
    verification: VerificationStatus = VerificationStatus.UNVERIFIED
    manually_corrected: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def name(self) -> str:
        return midi_to_name(self.midi)

    @property
    def overall_confidence(self) -> float:
        return min(
            self.detection_confidence,
            self.pitch_confidence,
            self.timing_confidence,
            self.duration_confidence,
        )

    def flag(self, issue: str) -> None:
        if issue not in self.issues:
            self.issues.append(issue)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hand"] = self.hand.value
        d["verification"] = self.verification.value
        d["name"] = self.name
        d["duration"] = self.duration
        d["overall_confidence"] = round(self.overall_confidence, 4)
        return d


# ---------------------------------------------------------------------------
# Video / geometry metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    title: str = ""
    source_url: str = ""
    variable_fps: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TempoSegment:
    start: float  # seconds
    bpm: float
    beats: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReconstructionReport:
    """Aggregate accuracy statistics. Deliberately avoids any '100% accurate'
    claim unless compared against known ground truth."""

    notes_total: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    needs_review: int = 0
    possible_missing: int = 0
    possible_duplicates: int = 0
    pitch_conflicts: int = 0
    timing_conflicts: int = 0
    mean_onset_uncertainty_ms: float = 0.0
    keyboard_low_midi: int = 21
    keyboard_high_midi: int = 108
    visual_verified: bool = False
    ground_truth_compared: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x == mu else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)
