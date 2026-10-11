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
    static keyboard stays sharp - the ideal image to measure geometry from."""
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
    # White keys stay bright all the way up to the strike line even where black
    # keys sit between them. A ROW MEAN, however, is dragged down through the
    # black-key band (dark blacks interleaved with the white gaps) and stops the
    # upward walk short - capturing only the white aprons below the black keys and
    # wrecking pitch detection. A high per-row percentile follows the white-key
    # columns instead, so the band spans the full keyboard including black keys.
    brightness = np.percentile(gray, 80, axis=1).astype(np.float64)
    brightness = np.convolve(brightness, np.ones(3) / 3, mode="same")

    # a bright key level anchored from the lower half (robust to a dark bottom
    # letterbox, which would otherwise zero out the bottom-row anchor)
    kb_level = float(np.percentile(brightness[int(h * 0.5):], 90))
    if kb_level < 90:  # keyboard may not be bright (rare)
        kb_level = float(brightness.max())
    thr = max(90.0, kb_level * 0.55)

    bottom = h - 1
    # ignore a possible thin dark border at the very bottom
    while bottom > h * 0.5 and brightness[bottom] < thr:
        bottom -= 1
    top = bottom
    while top > int(h * 0.15) and brightness[top - 1] >= thr:
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
    peaks = np.array(peaks, dtype=np.float64)
    # White keys are a regular comb, so the true separators sit at one dominant
    # pitch. Anything printed ON the keys - note-name letters, fingering numbers,
    # a logo - also reads as a dark column here and injects extra, irregular
    # "separators" that halve the apparent key width and shatter the octave
    # pattern. Lock the boundaries to the comb's own period (recovered from the
    # darkness profile, not assumed) so those off-grid intruders are dropped.
    return _regularise_boundaries(peaks, inv)


def _dominant_period(inv: np.ndarray, pmin: int = 6) -> float:
    """Fundamental spatial period of the darkness profile via autocorrelation -
    the white-key pitch. Robust to clutter printed on the keys: letters add noise
    at assorted spacings but the regular separator comb dominates the periodicity.
    Returns 0 if no clear period stands out.

    The fundamental is the SMALLEST-lag autocorrelation peak, not the tallest: a
    comb's harmonics (2x, 3x ... the key pitch) often autocorrelate higher than
    the pitch itself, so a plain argmax would lock onto a multiple and halve the
    key count. We take the first local maximum that still rises to at least half
    the tallest peak in range."""
    x = inv.astype(np.float64) - float(inv.mean())
    if x.shape[0] < 2 * pmin or not np.any(x):
        return 0.0
    ac = np.correlate(x, x, "full")[x.shape[0] - 1:]
    pmax = min(len(ac) - 2, len(inv) // 2)
    if pmax <= pmin:
        return 0.0
    maxima = [l for l in range(pmin, pmax + 1)
              if ac[l] >= ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > 0]
    if not maxima:
        return 0.0
    floor = 0.5 * max(ac[l] for l in maxima)
    lag = next((l for l in maxima if ac[l] >= floor), maxima[0])
    # parabolic refinement around the integer peak for a sub-pixel period
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    if denom != 0:
        lag = lag + 0.5 * (a - c) / denom
    return float(lag)


def _regularise_boundaries(peaks: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Keep only separators consistent with the keyboard's dominant key pitch.

    Walk a comb of the detected period across the span the raw peaks cover; at
    each grid slot take the nearest raw peak if one falls within a third of a key,
    otherwise synthesise the slot from the grid. Extra peaks from markings printed
    on the keys fall between slots and are discarded; a genuinely clean keyboard,
    whose peaks already sit one period apart, is reproduced unchanged."""
    if len(peaks) < 3:
        return peaks
    period = _dominant_period(inv)
    med = float(np.median(np.diff(peaks)))
    # Only intervene when the raw peaks disagree with the comb period (markings
    # present): a clean keyboard already matches it and is left exactly as-is.
    if period <= 0 or med <= 0 or abs(med - period) <= 0.15 * period:
        return peaks
    lo, hi = float(peaks[0]), float(peaks[-1])
    tol = period / 3.0
    grid: list[float] = []
    x = lo
    while x <= hi + tol:
        near = peaks[np.abs(peaks - x) <= tol]
        if len(near):
            pick = float(near[np.argmin(np.abs(near - x))])
            grid.append(pick)
            x = pick + period
        else:
            grid.append(x)
            x += period
    return np.array(grid, dtype=np.float64)


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


def _has_black_between_wide(upper: np.ndarray, x0: float, x1: float) -> bool:
    """Alternative black-key test by dark-region WIDTH rather than midpoint
    brightness. A black key is ~half a white key wide; the separator line between
    white keys is a few px. Counting dark columns across the whole centre-to-centre
    span catches a black key wherever it sits - real black keys are offset from the
    midpoint between front white-key centres, so the fixed central window of
    ``_has_black_between`` misses them on some keyboards - while a bare separator
    never reaches a key's width. Used as the fallback when the midpoint test's
    octave pattern comes out weak (see ``build_geometry``)."""
    span = x1 - x0
    if span < 4:
        return False
    a, b = int(round(x0)), int(round(x1))
    col = upper[:, max(0, a):max(1, b)].mean(axis=0)
    white_ref = float(np.percentile(upper, 85))
    dark_w = int((col < white_ref * 0.5).sum())
    # a black key is ~0.5-0.6 of a white key wide; a separator is a few px. The
    # 0.30 floor sits well below a key yet far above any separator.
    return dark_w >= 0.30 * span


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

    # black-key presence between adjacent white keys. The midpoint-brightness
    # test is crisp on a standard keyboard; the dark-width test survives black
    # keys offset from the front-centre midpoint (some stylised renders). Keep
    # whichever yields the cleaner octave pattern so neither keyboard regresses.
    gaps = _best_gaps(upper, white_centers)

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


def _best_gaps(upper: np.ndarray, white_centers: np.ndarray) -> list[bool]:
    """Black-key gap sequence from whichever detector matches the octave pattern
    best: the midpoint-brightness test (sharp on standard keyboards) or the
    dark-width test (robust to offset black keys on stylised renders)."""
    best_gaps: list[bool] = []
    best_score = -1.0
    for test in (_has_black_between, _has_black_between_wide):
        gaps = [test(upper, white_centers[i], white_centers[i + 1])
                for i in range(len(white_centers) - 1)]
        score = _geometry_confidence(gaps, _match_phase(gaps))
        if score > best_score:
            best_score, best_gaps = score, gaps
    return best_gaps


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
    """Assign each white key a MIDI number, then place black keys.

    Relative semitone positions come straight from the gap pattern: adjacent
    white keys differ by 2 semitones when a black key sits between them and by 1
    otherwise. Phase only fixes which pitch class the first white key is. The
    whole board is then transposed by whole octaves onto a standard range so
    pitch classes are preserved but the octave is anchored (§6)."""
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
    for i in range(n - 1):
        if not gaps[i]:
            continue
        black_midi = white_midis[i] + 1
        cx = (white_centers[i] + white_centers[i + 1]) / 2.0
        lanes[black_midi] = Lane(black_midi, float(cx), float(white_hw * 0.58), True)

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
    if best_cost == 1e18:
        # Board wider than 88 keys (a mis-detection): no whole-octave shift fits
        # both ends inside 21..108. Centre the board on the valid range rather
        # than anchoring the low end, which otherwise leaves the top lane past
        # 108 - and, on a badly over-wide board, past 127, where it becomes an
        # illegal MIDI note that crashes the exporter.
        mid = (lo + hi) / 2.0
        best_k = round((64.5 - mid) / 12.0)
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
