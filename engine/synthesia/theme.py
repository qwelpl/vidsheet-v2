"""Note-colour theme detection (§8, §31).

Synthesia-style renderers use arbitrary colour schemes (blue/green, cyan/orange,
rainbow, single colour, gradients...). Nothing is hard-coded: candidate note
colours are discovered from the moving, saturated content of the roll region.
Hand assignment from colour is deferred to the pipeline, which also uses spatial
distribution, because colour alone can be ambiguous (§16).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class ColorCluster:
    hue: float          # OpenCV hue 0..179
    sat: float
    val: float
    weight: float       # relative pixel mass
    hand_hint: Optional[str] = None  # filled by pipeline
    achromatic: bool = False   # white/grey note colour (matched by low sat, not hue)


@dataclass
class ThemeModel:
    clusters: list[ColorCluster]
    sat_min: int
    val_min: int
    rainbow: bool = False
    achromatic_sat_max: int = 0   # pixels below this sat are white-hand (0 = none)
    achromatic_val_min: int = 0   # ...and at/above this brightness (rejects dim bg)

    def classify(self, hue: float, sat: float, val: float) -> Optional[int]:
        """Return index of the closest colour cluster, or None if the pixel is
        not note-like (too desaturated / dark -> background or glow tail)."""
        if val < self.val_min:
            return None
        # A white/grey hand carries no hue, so it is matched by its LOW saturation
        # against a bright background-subtracted pixel, not by hue distance (which
        # is meaningless when a pixel is unsaturated). Route it before the sat gate
        # that would otherwise reject it as background.
        if (self.achromatic_sat_max > 0 and sat < self.achromatic_sat_max
                and val >= self.achromatic_val_min):
            for i, c in enumerate(self.clusters):
                if c.achromatic:
                    return i
            return None
        if sat < self.sat_min:
            return None
        if self.rainbow:
            return 0  # a single rainbow class; hand comes from position
        best, best_d = None, 1e9
        for i, c in enumerate(self.clusters):
            if c.achromatic:
                continue
            dh = min(abs(hue - c.hue), 180 - abs(hue - c.hue))
            if dh < best_d:
                best_d, best = dh, i
        return best if best_d <= 22 else None

    def to_dict(self) -> dict:
        return {
            "sat_min": self.sat_min, "val_min": self.val_min, "rainbow": self.rainbow,
            "achromatic_sat_max": self.achromatic_sat_max,
            "achromatic_val_min": self.achromatic_val_min,
            "clusters": [
                {"hue": c.hue, "sat": c.sat, "val": c.val, "weight": c.weight,
                 "hand_hint": c.hand_hint, "achromatic": c.achromatic}
                for c in self.clusters
            ],
        }


def detect_theme(
    frames_hsv_roll: list[np.ndarray],
    sat_min: int = 70,
    val_min: int = 70,
) -> ThemeModel:
    """Cluster note colours from roll-region HSV samples.

    ``frames_hsv_roll`` are HSV crops of the falling-note region across time.
    """
    # Sample note colours with a LOW saturation floor. Note bars are not always
    # vividly saturated - some visualisers use pale/pastel colours (e.g. a gold
    # left hand at saturation ~50) that a fixed sat_min=70 rejects outright,
    # collapsing that hand's notes to just the solid label. The dark background
    # sits near zero saturation, so a low floor still separates it, and the
    # saturation-squared hue weighting below keeps faint fringe from outvoting a
    # real colour. The detector's actual sat_min is derived from the palest
    # detected cluster at the end (never above the caller's cap).
    cap_sat_min = sat_min
    floor = 30
    # When notes are composited over a STILL image (album art, a character
    # montage) rather than a black roll, that backdrop is saturated and unmoving,
    # so clustering every roll pixel lets its colour masquerade as the dominant
    # note colour. Notes are transient at any given pixel, so the per-pixel
    # temporal median across time-spread samples is the backdrop; keep only pixels
    # that deviate from it and the backdrop drops out.
    #
    # But the backdrop is NOT always still - notes are just as often composited
    # over MOVING footage (a music video, an animation). There every pixel changes
    # frame to frame, so a median-deviation mask would keep the whole background
    # and defeat itself. So only subtract the median when the roll is actually
    # static; detect that by how much of it stays put across samples. Over moving
    # footage we fall back to clustering all pixels, which already tolerates a
    # moving backdrop because its ever-changing colours never form a stable hue
    # peak the way a note colour does (per-frame normalisation below).
    bg = None
    if len(frames_hsv_roll) >= 8 and len({f.shape for f in frames_hsv_roll}) == 1:
        med = np.median(np.stack(frames_hsv_roll), axis=0)
        med_v = med[..., 2].astype(np.int16)
        still = float(np.mean([
            np.mean(np.abs(f[..., 2].astype(np.int16) - med_v) < 12)
            for f in frames_hsv_roll
        ]))
        if still > 0.6:  # most of the roll is unchanging -> a still backdrop
            bg = med
    hues = []
    sats = []
    vals = []
    # Per-hue horizontal centroid, saturation-weighted. Two hands with close hues
    # (e.g. a red right hand and an orange left hand only ~8 apart) cannot be told
    # apart by hue alone, but they separate across the keyboard - the left hand's
    # colour sits left, the right hand's right. The two brightness tiers of ONE
    # hand, by contrast, share the same columns. This centroid lets the peak
    # selection below keep two close-but-distinct hands while still merging a
    # single hand's tiers (§7, §16).
    col_sum = np.zeros(180, dtype=np.float64)
    col_wt = np.zeros(180, dtype=np.float64)
    width = 0
    for hsv in frames_hsv_roll:
        hh, ww = hsv.shape[:2]
        width = ww
        h = hsv[..., 0].ravel()
        s = hsv[..., 1].ravel()
        v = hsv[..., 2].ravel()
        m = (s >= floor) & (v >= val_min)
        if bg is not None:
            dv = np.abs(v.astype(np.int16) - bg[..., 2].ravel().astype(np.int16))
            ds = np.abs(s.astype(np.int16) - bg[..., 1].ravel().astype(np.int16))
            m &= (dv >= 35) | (ds >= 40)
        hm = h[m]
        sm = s[m]
        if hm.size:
            cols = np.tile(np.arange(ww), hh)[m].astype(np.float64)
            wsq = (sm.astype(np.float64) / 255.0) ** 2
            np.add.at(col_sum, hm, wsq * cols)
            np.add.at(col_wt, hm, wsq)
        hues.append(hm)
        sats.append(sm)
        vals.append(v[m])
    H = np.concatenate(hues) if hues else np.array([], dtype=np.uint8)
    if H.size < 100:
        # nearly colourless theme (white/grey bars): treat value as the signal
        return ThemeModel(clusters=[ColorCluster(0, 0, 200, 1.0)],
                          sat_min=0, val_min=max(120, val_min), rainbow=False)

    S = np.concatenate(sats)
    V = np.concatenate(vals)
    # Solid note cores are markedly brighter than translucent glow, which shares
    # the same hue/saturation (uniform RGB scaling) and differs only in value.
    # Glow sits well below the bright-core level, so a threshold at half the
    # core brightness keeps cores and rejects glow (§34).
    core_level = float(np.percentile(V, 92))
    base = max(float(val_min), 0.55 * core_level)
    # Some themes render two brightness tiers of note (e.g. a bright and a dark
    # shade of the same hand colour); the dark tier can sit at value ~95 while the
    # bright tier is ~200, so a threshold anchored to the bright cores would drop
    # the whole dark tier as if it were glow. Glow, however, is DESATURATED as it
    # blends toward the background, whereas both note tiers stay strongly
    # saturated. So never threshold above the dimmest strongly-saturated pixels
    # (true cores): this admits a dark note tier while the sat gate still rejects
    # glow. Only ever lowers the threshold, so bright-only themes are unchanged.
    strong = S >= max(cap_sat_min, 120)
    if int(strong.sum()) >= 200:
        core_lo = float(np.percentile(V[strong], 12))
        base = min(base, core_lo - 12.0)
    val_min = int(np.clip(max(70.0, base), 55, 210))
    # Weight the hue histogram by saturation (real note colours are strongly
    # saturated, suppressing desaturated fringe, §8) AND normalise each frame's
    # contribution before summing, so a hue is scored by how CONSISTENTLY it
    # appears across frames rather than by raw pixel count. This stops a few
    # outlier frames - an intro/transition/logo in another colour - from
    # dominating the total and burying a genuine hand colour that is present
    # throughout the performance.
    hist = np.zeros(180, dtype=np.float64)
    for hf, sf in zip(hues, sats):
        if hf.size == 0:
            continue
        wf = (sf.astype(np.float64) / 255.0) ** 2
        hh = np.bincount(hf, weights=wf, minlength=180)
        tot = hh.sum()
        if tot > 0:
            hist += hh / tot
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 9), 0).ravel()

    # rainbow test: hue spread very wide and reasonably flat
    occupied = (hist > hist.max() * 0.15).sum()
    if occupied > 90:
        rainbow_sat = int(np.clip(0.6 * np.percentile(S, 20), 30, cap_sat_min))
        return ThemeModel(clusters=[ColorCluster(90, float(S.mean()), float(V.mean()), 1.0)],
                          sat_min=rainbow_sat, val_min=val_min, rainbow=True)

    # find up to 3 dominant, well-separated hue peaks. Peaks must be at least the
    # classification radius (22, see ``classify``) apart: two centres closer than
    # that are not separable at classify time, and in practice are one hand's two
    # brightness tiers (e.g. a bright core at hue 46 and a dark tier at hue 60).
    # A smaller gap let a single hand split into two clusters and consume both
    # kept slots, dropping the other hand's colour entirely (a pale, less
    # saturated hand is already under-weighted by the sat^2 hue weighting), which
    # collapsed hand assignment to a naive pitch split.
    # Find fine-grained peaks, then merge ones closer than the classify radius
    # UNLESS they separate across the keyboard - two close hues at clearly
    # different horizontal centroids are two hands (red right / orange left), not
    # one hand's two brightness tiers (which share columns).
    def _peak_col(p: int) -> float:
        # Tight window so two close peaks measure DISJOINT pixels; a wide window
        # would blend their columns together and hide the separation.
        idx = [(p + d) % 180 for d in range(-3, 4)]
        w = col_wt[idx].sum()
        return float(col_sum[idx].sum() / w) if w > 0 else -1.0

    raw_peaks = _hue_peaks(hist, min_sep=6, max_peaks=8)
    peaks: list[int] = []
    for p in raw_peaks:
        close = [q for q in peaks
                 if min(abs(p - q), 180 - abs(p - q)) < 22]
        if not close:
            peaks.append(p)
        else:
            cp = _peak_col(p)
            # keep as a separate hand only if it carries real mass AND its colour
            # sits in a clearly different part of the keyboard than the close peak
            spatially_distinct = (
                cp >= 0 and width > 0
                and hist[p] >= 0.30 * hist[peaks[0]]
                and any(_peak_col(q) >= 0
                        and abs(cp - _peak_col(q)) >= 0.15 * width
                        for q in close)
            )
            if spatially_distinct:
                peaks.append(p)
        if len(peaks) >= 3:
            break
    clusters = []
    total = hist.sum()
    for p in peaks:
        lo, hi = (p - 11) % 180, (p + 11) % 180
        if lo <= hi:
            mask = (H >= lo) & (H <= hi)
        else:
            mask = (H >= lo) | (H <= hi)
        weight = float(hist[[(p + d) % 180 for d in range(-11, 12)]].sum() / (total + 1e-6))
        clusters.append(ColorCluster(
            hue=float(p),
            sat=float(S[mask].mean()) if mask.any() else 128.0,
            val=float(V[mask].mean()) if mask.any() else 200.0,
            weight=weight,
        ))
    clusters.sort(key=lambda c: -c.weight)
    # Keep the dominant colour, plus any further colour carrying real mass. A
    # second hand's colour is a substantial fraction of note pixels (the weaker
    # hand in §16 still ran ~0.22), whereas compression fringe, particle tints or
    # residual backdrop bleed sit far below that. A flat "always keep top 2" let
    # such a sliver pose as a second hand, turning a genuinely single-colour theme
    # into a two-colour one; require the runners-up to clear a mass floor instead.
    #
    # BUT the sat^2 weighting above crushes a *pale* second hand: a pastel colour
    # (e.g. a lavender right hand at saturation ~50) can be a quarter of the note
    # pixels by COUNT yet score only a few percent of sat^2 mass, falling under the
    # floor and collapsing the theme to one colour. So also rescue a runner-up that
    # is a clearly distinct hue family from the dominant (>=35 away, where ordinary
    # fringe never sits - it hugs the dominant hue) and holds real pixel-count mass.
    dom_hue = clusters[0].hue

    def _hue_dist(a: float, b: float) -> float:
        return min(abs(a - b), 180 - abs(a - b))

    def _count_share(hue: float) -> float:
        if H.size == 0:
            return 0.0
        d = np.minimum(np.abs(H.astype(np.int16) - hue), 180 - np.abs(H.astype(np.int16) - hue))
        return float((d <= 11).mean())

    kept = [
        c for i, c in enumerate(clusters)
        if i == 0 or c.weight >= 0.12
        or (_hue_dist(c.hue, dom_hue) >= 35 and _count_share(c.hue) >= 0.10)
    ]
    # Detector saturation floor from the palest kept colour, so a pastel hand is
    # admitted while the near-zero background stays out. Never above the cap.
    pale = min((c.sat for c in kept), default=cap_sat_min)
    final_sat_min = int(np.clip(0.55 * pale, 30, cap_sat_min))

    # A WHITE/grey hand (common second hand: one chromatic, one white) carries no
    # hue, so the hue clustering above cannot see it and the theme collapses to one
    # colour - then hand assignment has no colour to split on and falls back to a
    # naive pitch split that scrambles the two hands wherever their ranges overlap.
    # Detect it as a separate achromatic cluster: a substantial mass of bright note
    # pixels whose saturation sits well below the chromatic colour(s). The split is
    # the valley between the two saturation populations, found by Otsu, so nothing
    # is hard-coded; require real mass on the white side and a clearly chromatic
    # colour on the other so a single-colour theme is never torn into two.
    # The ``floor`` sat gate above drops near-zero-saturation pixels, so a white
    # hand is absent from S/V entirely. Sample it on its own terms: bright pixels
    # that move against the still background (the note bars) regardless of hue.
    # Comparing the saturation of that whole bright-note population against the
    # chromatic colour tells white from chromatic.
    achro_max = 0
    achro_vmin = 0
    chroma_sat = min((c.sat for c in kept), default=0.0)
    # Only look for a white hand when at most one chromatic hand was found - the
    # white-hand case is "one colour + one white". With two chromatic hands
    # already separated, bright low-saturation pixels are the hands/keys/glow, not
    # a third hand, so adding an achromatic cluster would steal their notes.
    n_chromatic = sum(1 for c in kept if not c.achromatic)
    if kept and chroma_sat >= 60 and n_chromatic < 2:
        allS, allV = [], []
        for hsv in frames_hsv_roll:
            s = hsv[..., 1].ravel(); v = hsv[..., 2].ravel()
            mm = v >= max(val_min, 120)
            if bg is not None:
                bv = bg[..., 2].ravel().astype(np.int16)
                mm &= np.abs(v.astype(np.int16) - bv) >= 35
            allS.append(s[mm]); allV.append(v[mm])
        S2 = np.concatenate(allS) if allS else np.array([], np.uint8)
        V2 = np.concatenate(allV) if allV else np.array([], np.uint8)
        if S2.size >= 400:
            # Split white from chromatic at the midpoint between a colourless bar
            # (saturation ~0) and the detected chromatic colour; derived from that
            # colour, so nothing is hard-coded.
            s_split = 0.5 * chroma_sat
            white_m = S2 < s_split
            white_share = float(white_m.sum()) / float(S2.size)
            # A real white hand is a substantial mass of bright pixels sitting well
            # below the chromatic colour's saturation; guards keep a single-colour
            # (chromatic-only) theme from being torn into two.
            if (white_share >= 0.20
                    and float(np.median(S2[white_m])) <= 40):
                achro_max = int(np.clip(s_split, 20, max(21, chroma_sat - 15)))
                white_val = float(np.median(V2[white_m]))
                # A white note is BRIGHT (its own core value); dim low-saturation
                # background/glow particles also clear the sat gate and would
                # otherwise register as a permanent note in every lane. Floor the
                # match near the white core so only genuine lit bars pass.
                achro_vmin = int(np.clip(0.75 * white_val, val_min, 245))
                kept.append(ColorCluster(
                    hue=0.0, sat=float(np.median(S2[white_m])),
                    val=white_val, weight=white_share, achromatic=True))

    return ThemeModel(clusters=kept, sat_min=final_sat_min, val_min=val_min,
                      achromatic_sat_max=achro_max, achromatic_val_min=achro_vmin)


def _otsu(x: np.ndarray) -> int:
    """Generic Otsu split (0..255) - the value maximising between-class variance,
    i.e. the valley between two populations (here: background vs note saturation)."""
    if x.size < 50:
        return 40
    hist = np.bincount(x.astype(np.int32).clip(0, 255), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    denom = omega * (total - omega)
    denom[denom == 0] = 1e-9
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    return int(np.argmax(sigma_b))


def _otsu_threshold(v: np.ndarray) -> int:
    """Otsu split of the value histogram (glow vs core). Biased toward the core
    side so faint glow tails stay excluded."""
    if v.size < 50:
        return 90
    hist = np.bincount(v.astype(np.int32).clip(0, 255), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (total - omega)
    denom[denom == 0] = 1e-9
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    thr = int(np.argmax(sigma_b))
    return int(np.clip(thr, 60, 180))


def _hue_peaks(hist: np.ndarray, min_sep: int, max_peaks: int) -> list[int]:
    order = np.argsort(hist)[::-1]
    peaks: list[int] = []
    for idx in order:
        if hist[idx] <= 0:
            break
        if all(min(abs(idx - p), 180 - abs(idx - p)) >= min_sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= max_peaks:
            break
    return peaks
