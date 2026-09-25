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
    """Find the vertical extent of the keyboard (§5 bottom-keyboard case).

    On a temporal-median frame the falling-note roll averages to a dark
    background while the keyboard stays a bright, contiguous block at the
    bottom. We therefore walk upward from the last row through the connected
    bright region: the keyboard body (white keys, with black keys interleaved)
    stays bright until the dark roll begins."""
    h, w = gray.shape
    brightness = cv2.GaussianBlur(gray.mean(axis=1).reshape(-1, 1), (1, 1), 0).ravel()
    brightness = np.convolve(brightness, np.ones(3) / 3, mode="same")

    # brightness of the bottom-most rows anchors the "white key" level
    kb_level = float(np.median(brightness[int(h * 0.95):]))
    if kb_level < 90:  # bottom isn't bright -> keyboard may not be flush (rare)
        kb_level = float(brightness.max())
    thr = max(90.0, kb_level * 0.55)

    bottom = h - 1
    # ignore a possible thin dark border at the very bottom
    while bottom > h * 0.5 and brightness[bottom] < thr:
        bottom -= 1
    top = bottom
    while top > int(h * 0.30) and brightness[top - 1] >= thr:
        top -= 1
    # guard against absurdly thin detection
    if bottom - top < h * 0.04:
        top = max(int(h * 0.30), bottom - int(h * 0.16))
    return top, bottom


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


def _detect_black_centers(gray: np.ndarray, kb_top: int, kb_bottom: int) -> list[float]:
    """Sub-pixel x-centres of the black keys.

    Black keys are wide dark blobs occupying the upper part of the keyboard. We
    locate the rows where a large fraction of columns are dark (the black-key
    band, distinct from the white-key body where only thin separators are dark),
    then extract the dark blobs of black-key width in that band."""
    band = gray[kb_top:kb_bottom, :]
    h, w = band.shape
    dark_frac = (band < 100).mean(axis=1)          # per-row fraction dark
    # black-key band = the FIRST contiguous run of dark-rich rows from the top
    # (the black keys). A dark strip below the keyboard, if the band overshoots,
    # is a separate later run and must be ignored.
    r0 = 0
    while r0 < h and dark_frac[r0] <= 0.18:
        r0 += 1
    if r0 >= h:
        return []
    r1 = r0
    while r1 < h and dark_frac[r1] > 0.18:
        r1 += 1
    if r1 - r0 < 3:
        return []
    strip = band[r0 + 1: r1 - 1, :]
    col = strip.mean(axis=0)
    thr = 0.5 * (float(np.percentile(col, 85)) + float(np.percentile(col, 15)))
    dark = col < min(thr, 120)
    centers = []
    i = 0
    while i < w:
        if dark[i]:
            j = i
            while j < w and dark[j]:
                j += 1
            if (j - i) >= 6:                        # black-key width, not a seam
                seg = (col.max() - col[i:j])
                s = seg.sum()
                cx = (np.arange(i, j) * seg).sum() / s if s > 0 else (i + j) / 2
                centers.append(float(cx))
            i = j
        else:
            i += 1
    return centers


def _black_in_gap(centers: list[float], x0: float, x1: float) -> bool:
    lo, hi = min(x0, x1), max(x0, x1)
    return any(lo < c < hi for c in centers)


def _has_black_between(upper: np.ndarray, x0: float, x1: float) -> bool:
    """Is there a black key between two adjacent white-key centres?

    A black key is a *wide* dark block centred between the two white centres,
    unlike the thin (1-2 px) separator line that also sits there when there is
    no black key. We therefore test the mean brightness of the central band, not
    the single darkest column (which the separator would always trip)."""
    span = x1 - x0
    if span < 4:
        return False
    ca = int(round(x0 + 0.30 * span))
    cb = int(round(x1 - 0.30 * span))
    if cb <= ca:
        ca, cb = int(round((x0 + x1) / 2)) - 1, int(round((x0 + x1) / 2)) + 2
    central = float(upper[:, ca:cb].mean())
    white_ref = float(np.percentile(upper, 85))  # white-key body brightness
    return central < white_ref * 0.5


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

    # Detect the black keys directly (dark blobs in the upper keyboard) rather
    # than assuming they sit at white-key midpoints. This yields their true
    # x-centres — essential when a renderer offsets or narrows them — and a
    # reliable gap pattern for the C-phase (§6).
    black_centers = _detect_black_centers(gray, kb_top, kb_bottom)
    if black_centers:
        gaps = [_black_in_gap(black_centers, white_centers[i], white_centers[i + 1])
                for i in range(len(white_centers) - 1)]
    else:
        gaps = [_has_black_between(upper, white_centers[i], white_centers[i + 1])
                for i in range(len(white_centers) - 1)]

    phase = _match_phase(gaps)
    lanes = _assign_pitches(white_centers, white_hw, gaps, phase,
                            black_centers if black_centers else None)
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


def _assign_pitches(white_centers, white_hw, gaps, phase,
                    black_centers=None) -> dict[int, Lane]:
    """Assign each white key a MIDI number, then place black keys.

    Relative semitone positions come straight from the gap pattern: adjacent
    white keys differ by 2 semitones when a black key sits between them and by 1
    otherwise. Phase only fixes which pitch class the first white key is. The
    whole board is then transposed by whole octaves onto a standard range so
    pitch classes are preserved but the octave is anchored (§6). When actual
    black-key centres were detected, black lanes are placed there (matching where
    black-key notes really fall) instead of at the white-key midpoint."""
    n = len(white_centers)
    # cumulative relative semitones from the first white key
    rel = [0]
    for i in range(n - 1):
        rel.append(rel[-1] + (2 if gaps[i] else 1))
    first_pc = _WHITE_SEMITONES[phase % 7]

    lanes: dict[int, Lane] = {}
    base = 60  # provisional; corrected by transpose below
    for i in range(n):
        midi = base + first_pc + rel[i]
        lanes[midi] = Lane(midi, float(white_centers[i]), float(white_hw), False)
    white_midis = list(lanes.keys())
    black_hw = white_hw * 0.62
    for i in range(n - 1):
        if not gaps[i]:
            continue
        black_midi = white_midis[i] + 1
        lo, hi = white_centers[i], white_centers[i + 1]
        cx = (lo + hi) / 2.0
        if black_centers:                      # snap to the real black key
            cand = [c for c in black_centers if lo < c < hi]
            if cand:
                cx = min(cand, key=lambda c: abs(c - (lo + hi) / 2))
        lanes[black_midi] = Lane(black_midi, float(cx), float(black_hw), True)

    shift = _octave_transpose(min(lanes), max(lanes))
    if shift:
        lanes = {m + shift: Lane(m + shift, l.center, l.half_width, l.is_black)
                 for m, l in lanes.items()}
    return lanes


def _octave_transpose(lo: int, hi: int) -> int:
    """Whole-octave shift bringing [lo, hi] onto the piano (21..108), centred so
    both ends sit inside range. Preserves pitch classes."""
    span = hi - lo
    best_k, best_cost = 0, 1e18
    for k in range(-10, 11):
        nlo, nhi = lo + k * 12, hi + k * 12
        if nlo < 21 or nhi > 108:
            continue
        cost = abs(nlo - 21) + abs(108 - nhi)
        if cost < best_cost:
            best_cost, best_k = cost, k
    if best_cost == 1e18:  # board wider than 88 keys shouldn't happen; clamp low
        best_k = round((21 - lo) / 12.0)
    return best_k * 12


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
