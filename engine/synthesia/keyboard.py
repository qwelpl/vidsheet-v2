"""Keyboard detection and pitch-lane geometry (§5, §6, §7, §30, §57).

The keyboard is the anchor for the whole reconstruction: every horizontal
position in the falling-note roll is mapped to an exact MIDI note through the
detected key geometry, so octave errors that plague audio-only systems are
avoided (§57). Detection uses the characteristic black-key groups of two and
three rather than dividing the screen into equal slices (§6).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

from .model import midi_to_name, is_black_key, name_to_midi


# White-key pattern within one octave (C D E F G A B): is there a black key to
# the immediate right of this white key? E->F and B->C have none.
_OCTAVE_GAP_PATTERN = [1, 1, 0, 1, 1, 1, 0]  # index 0 == C
_WHITE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]     # C D E F G A B semitone offsets


@dataclass
class Lane:
    midi: int
    center: float      # sub-pixel horizontal centre in roll coordinates
    half_width: float
    is_black: bool

    @property
    def x0(self) -> float:
        return self.center - self.half_width

    @property
    def x1(self) -> float:
        return self.center + self.half_width


@dataclass
class KeyboardGeometry:
    width: int
    height: int
    keyboard_top: float          # y of top of keyboard == strike line
    keyboard_bottom: float
    lanes: dict[int, Lane]
    low_midi: int
    high_midi: int
    note_direction: str = "down"  # notes travel toward the keyboard
    confidence: float = 1.0
    manual: bool = False

    @property
    def strike_y(self) -> float:
        return self.keyboard_top

    def midi_at(self, x: float, prefer_black: Optional[bool] = None) -> tuple[int, float]:
        """Map a horizontal position to the nearest lane. Returns (midi, conf).

        A footprint's *center* is matched to the closest lane centre. When the
        caller knows the note is narrow (black-key candidate) it can bias the
        match with ``prefer_black`` to resolve the white/black overlap (§7)."""
        best_midi = self.low_midi
        best_d = float("inf")
        second = float("inf")
        for m, lane in self.lanes.items():
            d = abs(lane.center - x)
            if prefer_black is True and not lane.is_black:
                d *= 1.6
            elif prefer_black is False and lane.is_black:
                d *= 1.6
            if d < best_d:
                second = best_d
                best_d = d
                best_midi = m
            elif d < second:
                second = d
        # confidence: how much closer the winner is than the runner-up
        lane = self.lanes[best_midi]
        margin = (second - best_d) / max(1.0, lane.half_width)
        conf = float(np.clip(0.5 + 0.5 * margin, 0.4, 1.0))
        return best_midi, conf

    def lane_span(self, midi: int) -> tuple[float, float]:
        lane = self.lanes[midi]
        return lane.x0, lane.x1

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "keyboard_top": self.keyboard_top,
            "keyboard_bottom": self.keyboard_bottom,
            "strike_y": self.strike_y,
            "low_midi": self.low_midi,
            "high_midi": self.high_midi,
            "low_name": midi_to_name(self.low_midi),
            "high_name": midi_to_name(self.high_midi),
            "note_direction": self.note_direction,
            "confidence": self.confidence,
            "manual": self.manual,
            "lanes": [
                {"midi": l.midi, "name": midi_to_name(l.midi), "center": l.center,
                 "half_width": l.half_width, "is_black": l.is_black}
                for l in sorted(self.lanes.values(), key=lambda l: l.midi)
            ],
        }


# ---------------------------------------------------------------------------
# Static-background estimate (median over time isolates the still keyboard)
# ---------------------------------------------------------------------------

def temporal_median(frames: list[np.ndarray]) -> np.ndarray:
    """Median across sampled frames. Moving falling notes average away while the
    static keyboard stays sharp — the ideal image to measure geometry from."""
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def detect_keyboard_band(gray: np.ndarray) -> tuple[int, int]:
    """Find the vertical extent of the keyboard.

    Keyboard rows are bright (white keys) and rich in vertical edges (key
    separators + black keys). We score each row and take the tallest bright,
    high-edge band in the lower half of the frame (§5 bottom-keyboard case)."""
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_energy = np.abs(gx).mean(axis=1)
    brightness = gray.mean(axis=1)

    edge_n = edge_energy / (edge_energy.max() + 1e-6)
    bright_n = brightness / 255.0
    score = 0.5 * bright_n + 0.5 * edge_n

    # search lower 60% of the frame for the keyboard
    lo = int(h * 0.40)
    best = (lo, h - 1, -1.0)
    thr = np.percentile(score[lo:], 55)
    y = h - 1
    while y >= lo:
        if score[y] >= thr:
            y2 = y
            while y >= lo and score[y] >= thr * 0.75:
                y -= 1
            top = y + 1
            band_score = float(score[top:y2 + 1].mean()) * (y2 - top)
            if band_score > best[2]:
                best = (top, y2, band_score)
        y -= 1
    return best[0], best[1]


def _find_white_boundaries(strip: np.ndarray) -> np.ndarray:
    """Given a horizontal strip from the lower (all-white) keyboard region,
    return sub-pixel x positions of the dark separators between white keys."""
    col = strip.mean(axis=0)
    inv = col.max() - col                       # separators are dark -> peaks
    inv = cv2.GaussianBlur(inv.reshape(1, -1), (1, 5), 0).ravel()
    w = inv.shape[0]
    # adaptive threshold; separators are the clearly-dark local maxima
    thr = inv.mean() + 0.6 * inv.std()
    peaks = []
    i = 1
    while i < w - 1:
        if inv[i] >= thr and inv[i] >= inv[i - 1] and inv[i] >= inv[i + 1]:
            j = i
            while j < w - 1 and inv[j + 1] == inv[i]:
                j += 1
            # sub-pixel centroid of the peak neighbourhood
            a, b = max(0, i - 2), min(w, j + 3)
            wsum = inv[a:b].sum()
            centroid = (np.arange(a, b) * inv[a:b]).sum() / (wsum + 1e-6)
            peaks.append(centroid)
            i = j + 2
        else:
            i += 1
    return np.array(peaks, dtype=np.float64)


def _has_black_between(upper: np.ndarray, x0: float, x1: float) -> bool:
    """Is there a black key between two adjacent white-key centres? Look at the
    upper keyboard region for a persistently dark column."""
    a, b = int(round(x0)), int(round(x1))
    if b - a < 3:
        return False
    seg = upper[:, a:b].mean(axis=0)
    darkest = seg.min()
    surround = float(upper[:, max(0, a - 2):a + 1].mean() + upper[:, b:b + 3].mean()) / 2.0
    return darkest < surround - 30 and darkest < 110


def build_geometry(
    median_bgr: np.ndarray,
    kb_top: Optional[int] = None,
    kb_bottom: Optional[int] = None,
) -> KeyboardGeometry:
    """Full geometry model from a static (median) frame."""
    h, w = median_bgr.shape[:2]
    gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)

    if kb_top is None or kb_bottom is None:
        kb_top, kb_bottom = detect_keyboard_band(gray)

    band_h = kb_bottom - kb_top
    # lower ~30% of the band is below the black keys -> only white separators
    lower = gray[kb_bottom - max(3, band_h // 3): kb_bottom, :]
    upper = gray[kb_top: kb_top + max(3, band_h // 2), :]

    boundaries = _find_white_boundaries(lower)
    if len(boundaries) < 3:
        # fallback: assume full 88-key board evenly spaced
        return _fallback_88(w, h, kb_top, kb_bottom)

    # white-key centres = midpoints between consecutive boundaries, plus the
    # two edge keys before the first / after the last boundary.
    step = float(np.median(np.diff(boundaries)))
    starts = np.concatenate([[boundaries[0] - step], boundaries])
    ends = np.concatenate([boundaries, [boundaries[-1] + step]])
    white_centers = (starts + ends) / 2.0
    white_hw = step / 2.0

    # black-key presence between adjacent white keys
    gaps = [
        _has_black_between(upper, white_centers[i], white_centers[i + 1])
        for i in range(len(white_centers) - 1)
    ]

    phase = _match_phase(gaps)
    lanes = _assign_pitches(white_centers, white_hw, gaps, phase)
    low = min(lanes),
    lo_midi = min(lanes)
    hi_midi = max(lanes)

    return KeyboardGeometry(
        width=w, height=h,
        keyboard_top=float(kb_top), keyboard_bottom=float(kb_bottom),
        lanes=lanes, low_midi=lo_midi, high_midi=hi_midi,
        confidence=_geometry_confidence(gaps, phase),
    )


def _match_phase(gaps: list[bool]) -> int:
    """Find which white key is C by matching the observed gap sequence against
    the octave pattern [1,1,0,1,1,1,0] at all 7 rotations (§6)."""
    obs = np.array([1 if g else 0 for g in gaps], dtype=int)
    best_phase, best_score = 0, -1
    for phase in range(7):
        pat = np.array([_OCTAVE_GAP_PATTERN[(i + phase) % 7] for i in range(len(obs))])
        score = int((pat == obs).sum())
        if score > best_score:
            best_score, best_phase = score, phase
    return best_phase


def _geometry_confidence(gaps: list[bool], phase: int) -> float:
    obs = np.array([1 if g else 0 for g in gaps], dtype=int)
    pat = np.array([_OCTAVE_GAP_PATTERN[(i + phase) % 7] for i in range(len(obs))])
    if len(obs) == 0:
        return 0.3
    return float((pat == obs).mean())


def _assign_pitches(white_centers, white_hw, gaps, phase) -> dict[int, Lane]:
    """Assign each white key a MIDI number from the matched phase, place black
    keys between the appropriate white pairs (narrower, §6)."""
    lanes: dict[int, Lane] = {}
    # white key index 0 has octave position (phase). Choose an octave so the
    # lowest key lands in a sane MIDI range; refined later against 88-key board.
    n = len(white_centers)
    # anchor: assume the board's lowest key sits around octave giving midi ~ 21+
    # We compute white-key semitone offset from C, accumulate octaves.
    base_octave = 2  # provisional; corrected by range clamp below
    for i in range(n):
        pos = (i + phase) % 7
        octave_advance = (i + phase) // 7
        semitone = _WHITE_SEMITONES[pos]
        midi = (base_octave + 1) * 12 + semitone + octave_advance * 12
        lanes[midi] = Lane(midi, float(white_centers[i]), float(white_hw), False)
    # black keys
    for i in range(n - 1):
        if not gaps[i]:
            continue
        left_midi = _white_midi(i, phase, base_octave)
        black_midi = left_midi + 1
        cx = (white_centers[i] + white_centers[i + 1]) / 2.0
        lanes[black_midi] = Lane(black_midi, float(cx), float(white_hw * 0.58), True)

    # shift the whole board so the lowest key is a plausible piano key.
    lo = min(lanes)
    # snap lowest white key onto the nearest real piano key >= A0 (21)
    target_lo = _nearest_playable_low(lanes)
    shift = target_lo - lo
    if shift:
        lanes = {m + shift: Lane(m + shift, l.center, l.half_width, l.is_black)
                 for m, l in lanes.items()}
    return lanes


def _white_midi(i: int, phase: int, base_octave: int) -> int:
    pos = (i + phase) % 7
    octave_advance = (i + phase) // 7
    return (base_octave + 1) * 12 + _WHITE_SEMITONES[pos] + octave_advance * 12


def _nearest_playable_low(lanes: dict[int, Lane]) -> int:
    """Pick a lowest-MIDI so the board maps onto standard piano ranges. Common
    boards: 88 (A0=21), 76 (E1=28), 61 (C2=36), 49 (C2), 25. We keep the
    detected pitch classes and only translate by whole octaves."""
    lo = min(lanes)
    hi = max(lanes)
    span = hi - lo
    lo_pc = lo % 12
    # candidate lowest keys for common boards with matching pitch class
    candidates = [21, 28, 29, 36, 48]  # A0, E1, F1, C2, C3
    best = lo
    best_cost = 1e9
    for cand in candidates:
        if (cand % 12) != lo_pc and (cand % 12) != (lo_pc):
            # allow octave translation to reach this pitch class
            k = round((cand - lo) / 12.0)
            trans = lo + k * 12
            if trans % 12 != lo_pc:
                continue
            cand = trans
        # only whole-octave translations keep pitch classes intact
        if (cand - lo) % 12 != 0:
            continue
        if cand < 21 or cand + span > 108:
            continue
        cost = abs(cand - 21) + abs((cand + span) - 108)
        if cost < best_cost:
            best_cost, best = cost, cand
    return best


def _fallback_88(w: int, h: int, kb_top: int, kb_bottom: int) -> KeyboardGeometry:
    """Even-spaced 88-key fallback when separator detection fails. Flagged with
    low confidence so the UI prompts for manual calibration (§30, §69)."""
    lanes: dict[int, Lane] = {}
    n_white = 52  # 88-key board has 52 white keys
    white_w = w / n_white
    wi = 0
    for midi in range(21, 109):
        if is_black_key(midi):
            continue
        cx = (wi + 0.5) * white_w
        lanes[midi] = Lane(midi, cx, white_w / 2, False)
        wi += 1
    # place black keys midway between neighbouring whites
    whites = sorted(lanes)
    for a, b in zip(whites, whites[1:]):
        if b - a == 2:  # a black key sits between
            bm = a + 1
            cx = (lanes[a].center + lanes[b].center) / 2
            lanes[bm] = Lane(bm, cx, white_w * 0.58 / 2 * 2 / 2, True)
    return KeyboardGeometry(
        width=w, height=h, keyboard_top=float(kb_top), keyboard_bottom=float(kb_bottom),
        lanes=lanes, low_midi=21, high_midi=108, confidence=0.35,
    )


def manual_geometry(
    width: int, height: int, left: float, right: float,
    strike_y: float, keyboard_bottom: float,
    first_key: str, last_key: str,
) -> KeyboardGeometry:
    """Build geometry from user calibration (§30). Keys are laid out from the
    real white-key layout between ``first_key`` and ``last_key``."""
    lo = name_to_midi(first_key)
    hi = name_to_midi(last_key)
    whites = [m for m in range(lo, hi + 1) if not is_black_key(m)]
    n_white = len(whites)
    white_w = (right - left) / max(1, n_white)
    lanes: dict[int, Lane] = {}
    for i, m in enumerate(whites):
        cx = left + (i + 0.5) * white_w
        lanes[m] = Lane(m, cx, white_w / 2, False)
    for m in range(lo, hi + 1):
        if not is_black_key(m):
            continue
        left_white = m - 1 if (m - 1) in lanes else None
        right_white = m + 1 if (m + 1) in lanes else None
        if left_white and right_white:
            cx = (lanes[left_white].center + lanes[right_white].center) / 2
        elif left_white:
            cx = lanes[left_white].center + white_w / 2
        else:
            cx = lanes[right_white].center - white_w / 2
        lanes[m] = Lane(m, cx, white_w * 0.58, True)
    return KeyboardGeometry(
        width=width, height=height, keyboard_top=float(strike_y),
        keyboard_bottom=float(keyboard_bottom), lanes=lanes,
        low_midi=lo, high_midi=hi, confidence=1.0, manual=True,
    )
