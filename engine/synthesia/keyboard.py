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


def _keyboard_extent(lower: np.ndarray) -> tuple[int, int]:
    """Horizontal [left, right] span of the keyboard body in a white-key strip.

    The white keys are a bright contiguous block; letterbox / margins are dark.
    Returns the first and last column whose brightness clears a mid threshold."""
    col = lower.mean(axis=0)
    thr = max(60.0, 0.5 * float(np.percentile(col, 90)))
    bright = np.where(col >= thr)[0]
    if len(bright) < 8:
        return 0, lower.shape[1] - 1
    return int(bright[0]), int(bright[-1])


def _detect_black_centers(gray: np.ndarray, kb_top: int, kb_bottom: int) -> np.ndarray:
    """Sub-pixel x centres of the black keys, read from the upper keyboard band.

    Black keys are tall, high-contrast dark blocks that survive the temporal
    median cleanly even when the thin white-key separators below are too faint
    to find. We threshold the mean-brightness profile of a strip taken over the
    black-key body and return the centroid of each dark run."""
    band_h = kb_bottom - kb_top
    upper = gray[kb_top + int(band_h * 0.12): kb_top + int(band_h * 0.38), :]
    prof = upper.mean(axis=0)
    thr = (float(prof.max()) + float(prof.min())) / 2.0
    mask = prof < thr
    w = len(mask)
    min_run = max(3, int(band_h * 0.02))
    centers = []
    i = 0
    while i < w:
        if mask[i]:
            j = i
            while j < w and mask[j]:
                j += 1
            if j - i >= min_run:
                seg = thr - prof[i:j]            # weight by darkness
                centers.append((np.arange(i, j) * seg).sum() / (seg.sum() + 1e-6))
            i = j
        else:
            i += 1
    return np.array(centers, dtype=np.float64)


# Black-key position within an octave, in white-key units (C == 0):
# C#=0.5 D#=1.5  F#=3.5 G#=4.5 A#=5.5
_BLACK_WC_3 = [3.5, 4.5, 5.5]   # F# G# A#  (group of three)
_BLACK_WC_2 = [0.5, 1.5]        # C# D#     (group of two)


def _geometry_from_black_keys(
    gray: np.ndarray, w: int, h: int, kb_top: int, kb_bottom: int,
) -> Optional[KeyboardGeometry]:
    """Geometry anchored on the black keys (§6). Robust where the white-key
    separators are too faint to detect: the 2-and-3 black-key groups give both
    the pitch-class phase and, via a linear fit, the exact white-key pitch.

    Returns ``None`` when the black pattern is too sparse or irregular to trust,
    so the caller can fall back to white-separator detection."""
    centers = _detect_black_centers(gray, kb_top, kb_bottom)
    if len(centers) < 5:
        return None
    gaps = np.diff(centers)
    unit = float(np.median(gaps[gaps < np.median(gaps) * 1.3]))
    if unit <= 0:
        return None
    wide = gaps > 1.35 * unit

    # split the black keys into their 2- and 3-key groups at the wide gaps
    groups: list[list[float]] = []
    cur = [centers[0]]
    for k, g in enumerate(gaps):
        if wide[k]:
            groups.append(cur)
            cur = [centers[k + 1]]
        else:
            cur.append(centers[k + 1])
    groups.append(cur)
    sizes = [len(g) for g in groups]
    if 3 not in sizes:
        return None

    # The groups alternate 3,2,3,2,... A group of three is F#G#A#, a group of
    # two is C#D#. Anchor the alternation on the first full three-group and read
    # the type of every group from its parity; partial edge groups then follow.
    anchor = sizes.index(3)
    types = [3 if (gi - anchor) % 2 == 0 else 2 for gi in range(len(groups))]
    ocs, oc = [], 0
    for t in types:
        ocs.append(oc)
        if t == 3:          # F#G#A# closes an octave's black keys
            oc += 1

    xs, wcs = [], []
    for gi, (grp, t, o) in enumerate(zip(groups, types, ocs)):
        full = _BLACK_WC_3 if t == 3 else _BLACK_WC_2
        n = len(grp)
        if n > len(full):
            return None                         # impossible group size
        if n < len(full):
            sub = full[len(full) - n:] if gi == 0 else full[:n]
        else:
            sub = full
        for cx, wc in zip(grp, sub):
            xs.append(cx)
            wcs.append(o * 7 + wc)
    xs = np.array(xs)
    wcs = np.array(wcs)

    A = np.vstack([wcs, np.ones_like(wcs)]).T
    (white_w, intercept), *_ = np.linalg.lstsq(A, xs, rcond=None)
    if white_w <= 1:
        return None
    resid = float(np.sqrt(np.mean((A @ [white_w, intercept] - xs) ** 2)))
    if resid > 0.30 * white_w:                  # fit too loose to trust
        return None

    # white keys at integer white-coords that land inside the keyboard body
    band_h = kb_bottom - kb_top
    lower = gray[kb_bottom - max(3, band_h // 3): kb_bottom, :]
    left_edge, right_edge = _keyboard_extent(lower)
    half = white_w / 2.0
    i_lo = int(np.floor((left_edge - intercept) / white_w))
    i_hi = int(np.ceil((right_edge - intercept) / white_w))
    lanes: dict[int, Lane] = {}
    base = 60
    for i in range(i_lo, i_hi + 1):
        cx = white_w * i + intercept
        if cx < left_edge - half * 0.5 or cx > right_edge + half * 0.5:
            continue
        midi = base + 12 * (i // 7) + _WHITE_SEMITONES[i % 7]
        lanes[midi] = Lane(midi, float(cx), float(half), False)
    if len(lanes) < 10:
        return None

    # black keys between whites a tone apart
    white_midis = sorted(lanes)
    for a, b in zip(white_midis, white_midis[1:]):
        if b - a == 2:
            bm = a + 1
            cx = (lanes[a].center + lanes[b].center) / 2.0
            lanes[bm] = Lane(bm, float(cx), float(white_w * 0.29), True)

    shift = _octave_transpose(min(lanes), max(lanes))
    if shift:
        lanes = {m + shift: Lane(m + shift, l.center, l.half_width, l.is_black)
                 for m, l in lanes.items()}
    lo_midi, hi_midi = min(lanes), max(lanes)
    # confidence: tight fit + regular alternation
    conf = float(np.clip(1.0 - resid / white_w, 0.55, 0.95))
    return KeyboardGeometry(
        width=w, height=h,
        keyboard_top=float(kb_top), keyboard_bottom=float(kb_bottom),
        lanes=lanes, low_midi=lo_midi, high_midi=hi_midi, confidence=conf,
    )


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

    # Black keys are the most reliable landmark (tall, high-contrast blocks),
    # so anchor on them first. White-key separators are often too faint on a
    # temporal-median frame to count correctly - under-counting them silently
    # mis-scales the whole board (§6). Fall back to separator detection only
    # when the black pattern is too sparse / irregular to fit.
    black_geo = _geometry_from_black_keys(gray, w, h, kb_top, kb_bottom)
    if black_geo is not None:
        return black_geo

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
    """Whole-octave shift bringing [lo, hi] onto the piano (21..108).

    Absolute octave can't be read from key geometry alone - only the pitch
    classes are known - so several whole-octave placements fit inside the piano
    range. Synthesia tutorials frame the played range around middle C, so we
    pick the placement whose mid-point sits closest to MIDI 60, breaking ties by
    how centred it is in 21..108. Preserves pitch classes."""
    best_k, best_cost = 0, 1e18
    for k in range(-10, 11):
        nlo, nhi = lo + k * 12, hi + k * 12
        if nlo < 21 or nhi > 108:
            continue
        mid = (nlo + nhi) / 2.0
        cost = abs(mid - 60.0) + 0.01 * (abs(nlo - 21) + abs(108 - nhi))
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
