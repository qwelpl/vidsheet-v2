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
    p80 = np.percentile(gray, 80, axis=1).astype(np.float64)
    p80 = np.convolve(p80, np.ones(3) / 3, mode="same")
    # The 80th-percentile path assumes bright white keys. Dark-themed boards
    # (grey keys on black) never clear a fixed bright threshold, so detection
    # latches onto a bright strike-line glow instead and misses the keyboard. The
    # row MEDIAN separates them regardless of brightness: the keyboard's keys fill
    # the row, so its median sits well above the backdrop's, whereas the falling-
    # note roll is sparse bars over that backdrop and its median stays at backdrop
    # level. A row is keyboard if EITHER the bright-key test or the median test
    # fires - the former keeps ordinary bright boards pixel-identical, the latter
    # rescues dark ones.
    med = np.percentile(gray, 50, axis=1).astype(np.float64)
    med = np.convolve(med, np.ones(3) / 3, mode="same")

    top_rows = slice(0, int(h * 0.45))          # the roll / backdrop region
    bot = slice(int(h * 0.5), h)                 # where the keyboard sits
    kb_level = float(np.percentile(p80[bot], 90))
    if kb_level < 90:                            # keyboard may not be bright
        kb_level = float(p80.max())
    thr_bright = max(90.0, kb_level * 0.55)
    bg_med = float(np.percentile(med[top_rows], 85))
    kb_med = float(np.percentile(med[bot], 92))
    thr_med = max(bg_med + 6.0, bg_med + 0.25 * (kb_med - bg_med))

    rows = (p80 >= thr_bright) | (med >= thr_med)
    ys = np.where(rows[int(h * 0.3):])[0]
    if len(ys) == 0:
        return max(int(h * 0.30), h - 1 - int(h * 0.16)), h - 1
    bottom = int(h * 0.3) + int(ys.max())
    top = bottom
    while top > int(h * 0.15) and rows[top - 1]:
        top -= 1
    top, bottom = _grow_band_by_texture(gray, top, bottom)
    # guard against absurdly thin detection
    if bottom - top < h * 0.04:
        top = max(int(h * 0.30), bottom - int(h * 0.16))
    return top, bottom


def _grow_band_by_texture(gray: np.ndarray, top: int, bottom: int) -> tuple[int, int]:
    """Extend a keyboard band through the contiguous rows that carry key texture.

    On a dim board (grey keys under a bright hit-line glow) the brightness tests
    break inside the black-key zone, where dark blacks pull the row median under
    threshold, so the band stops at the white aprons below the blacks. Key rows
    are still unmistakable by their horizontal structure - every key edge is a
    sharp step - while the glow and the median-averaged roll above are smooth.
    Growing through rows whose mean |dI/dx| stays near the band's own keeps the
    walk on the keys and stops it dead at the glow. A band that already spans the
    keyboard is bordered by smooth rows, so it comes back unchanged."""
    h = gray.shape[0]
    if bottom - top < 2:
        return top, bottom
    tex = np.abs(np.diff(gray.astype(np.float64), axis=1)).mean(axis=1)
    thr = 0.3 * float(np.median(tex[top:bottom + 1]))
    if thr <= 0:
        return top, bottom
    while top > int(h * 0.15) and tex[top - 1] >= thr:
        top -= 1
    while bottom < h - 1 and tex[bottom + 1] >= thr:
        bottom += 1
    return top, bottom


def _comb_regularity(b: np.ndarray, width: int) -> float:
    """Score a separator set by how regular (evenly spaced, plentiful) it is. A
    clean white-key comb has many separators at one pitch; clutter or failure
    gives few separators or wildly uneven gaps.

    A full 88-key board has 52 white keys, so a key narrower than width/56 is
    impossible - that comb is sub-key texture or on-key labels, not separators -
    and a key wider than width/8 means the comb collapsed onto a few spurious
    peaks. Either way the comb is rejected."""
    if len(b) < 8:
        return 0.0
    d = np.diff(np.sort(b))
    med = float(np.median(d))
    if med < width / 56.0 or med > width / 8.0:
        return 0.0
    regular = int(np.sum(np.abs(d - med) <= 0.25 * med))
    return regular + regular / max(1, len(d))      # count, tie-broken by fraction


def _best_white_boundaries(gray: np.ndarray, kb_top: int, kb_bottom: int) -> np.ndarray:
    """White-key separators read from whichever sub-band gives the cleanest comb.
    Trying several depths tolerates both on-key labels near the bottom edge and
    dark-themed boards whose separators only resolve low on the keys."""
    band_h = kb_bottom - kb_top
    best: np.ndarray = np.array([])
    best_score = -1.0
    # (top_fraction_of_band, bottom_margin_fraction): strips below the black keys,
    # from a high one (clear of a bottom label row) down to a deep one (needed by
    # dark boards). Fractions are measured UP from kb_bottom.
    for lo, hi in ((0.33, 0.16), (0.30, 0.03), (0.22, 0.03), (0.42, 0.20)):
        y0 = kb_bottom - max(3, int(band_h * lo))
        y1 = kb_bottom - max(0, int(band_h * hi))
        if y1 - y0 < 3:
            continue
        b = _find_white_boundaries(gray[y0:y1, :])
        s = _comb_regularity(b, gray.shape[1])
        if s > best_score:
            best_score, best = s, b
    return best


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
    # White keys are a regular comb, so the true separators sit at one dominant
    # pitch. Note-name letters / fingering numbers printed ON the keys also read
    # as dark columns here and inject extra, irregular "separators" that halve the
    # apparent key width and shatter the octave pattern. Lock the boundaries to
    # the comb's own period (recovered from the signal, not assumed) to drop them.
    return _regularise_boundaries(np.array(peaks, dtype=np.float64), inv)


def _dominant_period(inv: np.ndarray, pmin: int = 6) -> float:
    """Fundamental spatial period of the darkness profile via autocorrelation -
    the white-key pitch. Robust to clutter printed on the keys: letters add noise
    at assorted spacings but the regular separator comb dominates the periodicity.
    The fundamental is the SMALLEST-lag autocorrelation peak that still reaches
    half the tallest peak - a comb's harmonics often autocorrelate higher than the
    pitch itself, so a plain argmax would lock onto a multiple and halve the count.
    Returns 0 if no clear period stands out."""
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
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    if denom != 0:
        lag = lag + 0.5 * (a - c) / denom
    return float(lag)


def _regularise_boundaries(peaks: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Keep only separators consistent with the keyboard's dominant key pitch.

    Walk a comb of the detected period across the span the raw peaks cover; at
    each grid slot take the nearest raw peak within a third of a key, else
    synthesise the slot from the grid. Extra peaks from markings printed on the
    keys fall between slots and are discarded; a clean keyboard, whose peaks
    already sit one period apart, is reproduced unchanged."""
    if len(peaks) < 3:
        return peaks
    period = _dominant_period(inv)
    med = float(np.median(np.diff(peaks)))
    if period <= 0 or med <= 0:
        return peaks
    if abs(med - period) <= 0.15 * period:
        d = np.diff(peaks)
        if np.all(np.abs(d - med) <= 0.25 * med):
            return peaks
        return _fit_boundary_lattice(peaks, med)
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


def _fit_boundary_lattice(peaks: np.ndarray, period: float) -> np.ndarray:
    """Rebuild a mostly-regular comb that has a few stray or missing separators.

    The pitch is right but the comb is broken locally: a vignetted board edge or
    a black key's bottom edge adds an off-grid peak, a pressed-key highlight or a
    low-contrast stretch drops one. A peak-to-peak walk lets one stray peak shift
    every key after it, so instead fit ONE lattice x0 + k*p to the peaks that agree
    with each other (white keys are evenly spaced), then emit every slot across
    the peaks' span - snapped to a nearby real peak for sub-pixel accuracy, else
    taken from the lattice."""
    n = len(peaks)
    # The median gap is skewed low by doubled peaks; a small pitch error grows by
    # one key's worth over a 52-key board, so take the mean of the on-pitch gaps.
    d = np.diff(peaks)
    on_pitch = np.abs(d - period) <= 0.25 * period
    if np.any(on_pitch):
        period = float(d[on_pitch].mean())
    k_rel = (peaks[None, :] - peaks[:, None]) / period
    on_grid = np.abs(k_rel - np.round(k_rel)) <= 0.2
    anchor = int(np.argmax(on_grid.sum(axis=1)))
    x0, p = float(peaks[anchor]), period
    for _ in range(3):
        # least-squares lattice over the peaks that sit on the current lattice
        k = np.round((peaks - x0) / p)
        inl = np.abs(peaks - (x0 + k * p)) <= 0.2 * p
        if inl.sum() < 2:
            break
        A = np.stack([np.ones(int(inl.sum())), k[inl]], axis=1)
        (x0, p), *_ = np.linalg.lstsq(A, peaks[inl], rcond=None)
    k = np.round((peaks - x0) / p)
    inl = np.abs(peaks - (x0 + k * p)) <= 0.2 * p
    tol = p / 3.0
    k_lo = int(np.ceil((peaks[0] - tol - x0) / p))
    k_hi = int(np.floor((peaks[-1] + tol - x0) / p))
    snap = 0.15 * p
    good = peaks[inl]
    out: list[float] = []
    for kk in range(k_lo, k_hi + 1):
        x = x0 + kk * p
        j = int(np.argmin(np.abs(good - x)))
        out.append(float(good[j]) if abs(good[j] - x) <= snap else float(x))
    return np.array(out, dtype=np.float64) if n else peaks


def _black_key_centers(upper: np.ndarray) -> np.ndarray:
    """x centres of the black keys: solid dark blocks in the upper keyboard band.

    Reading black keys directly as dark blobs (rather than testing each white-key
    gap for darkness) is robust to black keys that sit offset from the midpoint
    between front white-key centres, and gives their true spacing - which carries
    the 2/3 group pattern used to anchor the octave."""
    colmin = upper.min(axis=0).astype(np.float64)    # darkest pixel per column
    white_ref = float(np.percentile(upper, 85))
    dark = colmin < white_ref * 0.45
    centers: list[float] = []
    i, w = 0, len(dark)
    while i < w:
        if dark[i]:
            j = i
            while j < w and dark[j]:
                j += 1
            if j - i >= 6:                            # ignore thin separators
                centers.append((i + j) / 2.0)
            i = j
        else:
            i += 1
    return np.array(centers, dtype=np.float64)


def _gaps_from_black_centers(white_centers: np.ndarray,
                             black_centers: np.ndarray) -> list[bool]:
    """Black-key presence per adjacent white-key pair, decided by whether a
    detected black-key centre falls between the two white centres."""
    return [bool(np.any((black_centers > white_centers[i]) &
                        (black_centers < white_centers[i + 1])))
            for i in range(len(white_centers) - 1)]


def _phase_from_black_groups(white_centers: np.ndarray, black_centers: np.ndarray,
                             period: float) -> Optional[int]:
    """Octave phase (pitch class of the first white key) from the black-key 2/3
    grouping. Black keys fall in groups of two (C#,D#) and three (F#,G#,A#); the
    two-group is bracketed by the two wide gaps that cross the black-less E-F and
    B-C white pairs. Its left key is C#, so the white key just left of it is C.
    Returns None when no clean two-group is found."""
    if len(black_centers) < 5 or period <= 0:
        return None
    gaps = np.diff(black_centers)
    wide = gaps > 1.5 * period
    for k in range(1, len(gaps)):
        # black[k]..black[k+1] a narrow (within-group) gap bracketed by wide gaps
        if not wide[k] and wide[k - 1] and k + 1 < len(gaps) and wide[k + 1]:
            c_sharp = black_centers[k]
            left = white_centers[white_centers < c_sharp]
            if len(left):
                i_c = int(len(left) - 1)             # index of C in white_centers
                return (-i_c) % 7
    return None


def _has_black_between(upper: np.ndarray, x0: float, x1: float) -> bool:
    """Is there a black key between two adjacent white-key centres?

    A black key is a *wide* dark block centred between the two white centres,
    unlike the thin (1-2 px) separator line that also sits there when there is
    no black key. We therefore test the mean brightness of the central band, not
    the single darkest column (which the separator would always trip)."""
    span = x1 - x0
    if span < 4:
        return False
    wmax = upper.shape[1]
    ca = int(round(x0 + 0.30 * span))
    cb = int(round(x1 - 0.30 * span))
    if cb <= ca:
        ca, cb = int(round((x0 + x1) / 2)) - 1, int(round((x0 + x1) / 2)) + 2
    ca = max(0, min(ca, wmax - 1))
    cb = max(ca + 1, min(cb, wmax))
    if ca >= cb:
        return False
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
    upper = gray[kb_top: kb_top + max(3, band_h // 2), :]

    # White separators show in the band below the black keys, but the best row to
    # read them from varies: some renders print a note-name letter or fingering
    # number along the lower edge of each key (extra false separators near the
    # bottom), while a dark-themed board's separators only resolve near the bottom
    # where the keys are most uniform. No single strip wins both, so sample several
    # depths and keep the one whose separators form the most regular comb.
    boundaries = _best_white_boundaries(gray, kb_top, kb_bottom)
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
    # Phase-align the whole comb to the white-key bodies. A separator mis-anchored
    # by a column printed on the keys shifts every midpoint by a fraction of a key,
    # dropping the centres partly onto the (dark) black keys; slide all centres by
    # the single offset that maximises upper-band brightness at them. A clean
    # keyboard already sits at that maximum, so this is a no-op there.
    ucol = upper.mean(axis=0)
    shifts = np.linspace(-step / 2.0, step / 2.0, 25)        # index 12 == zero
    bright = [float(ucol[np.clip(np.round(white_centers + d).astype(int),
                                 0, w - 1)].mean()) for d in shifts]
    best_i = int(np.argmax(bright))
    best_shift = float(shifts[best_i])
    # Correct a real misalignment only: apply the shift when the re-phased centres
    # are clearly brighter (on the white bodies) than the as-found centres - which
    # happens when a printed column mis-anchored the separators. A clean keyboard's
    # midpoints already sit at the brightness maximum, so the gain is ~0 and the
    # centres are left exactly in place (pixel-accurate for the regression suite).
    if bright[best_i] > 1.1 * bright[12] and abs(best_shift) > 0.1 * step:
        white_centers = white_centers + best_shift

    # Black-key presence + octave phase, two ways, keeping whichever matches the
    # octave pattern best. (a) classic: test each white-key gap for a dark central
    # band, phase by correlating the gap sequence against [1,1,0,1,1,1,0]. (b) blob:
    # read the black keys as dark blocks directly and anchor the phase off their
    # 2/3 grouping - robust when black keys sit off the white-centre midpoints and
    # when the gap sequence is noisy, which otherwise leaves every pitch class
    # shifted. Confidence (fraction of gaps matching the octave pattern) picks.
    black_centers = _black_key_centers(upper)
    period = float(np.median(np.diff(boundaries))) if len(boundaries) > 1 else 0.0

    candidates: list[tuple[list[bool], int]] = []
    g_classic = [_has_black_between(upper, white_centers[i], white_centers[i + 1])
                 for i in range(len(white_centers) - 1)]
    candidates.append((g_classic, _match_phase(g_classic)))
    if len(black_centers) >= 5:
        g_blob = _gaps_from_black_centers(white_centers, black_centers)
        ph_blob = _phase_from_black_groups(white_centers, black_centers, period)
        candidates.append((g_blob, ph_blob if ph_blob is not None
                           else _match_phase(g_blob)))

    gaps, phase = max(candidates, key=lambda gp: _geometry_confidence(*gp))
    lanes = _assign_pitches(white_centers, white_hw, gaps, phase)
    # A piano spans at most A0..C8 (MIDI 21..108). Separator detection can graft a
    # phantom key or two past the real ends - a board edge, bezel or hit-line glow
    # read as an extra white separator - pushing the range to e.g. 19..110 (92
    # lanes). The black-key 2/3 grouping pitches the real keys correctly, so those
    # phantoms always fall OUTSIDE 21..108; drop them rather than emit more lanes
    # than a real keyboard has (which then mis-maps every pitch downstream).
    lanes = {m: l for m, l in lanes.items() if 21 <= m <= 108}
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
    # The board is a standard piano, so once the phase (pitch class of the first
    # white key) is fixed the black-key pattern is the fixed [1,1,0,1,1,1,0] comb.
    # Per-gap black detection is unreliable on low-contrast / dark-themed keyboards
    # - a missed black there turns a tone into a semitone and compresses every
    # pitch above it - so snap the gaps to that ideal pattern. A clean keyboard's
    # detected gaps already equal it, so this is a no-op there.
    gaps = [bool(_OCTAVE_GAP_PATTERN[(phase + i) % 7]) for i in range(n - 1)]
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
