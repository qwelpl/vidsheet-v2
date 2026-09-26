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


@dataclass
class ThemeModel:
    clusters: list[ColorCluster]
    sat_min: int
    val_min: int
    rainbow: bool = False

    def classify(self, hue: float, sat: float, val: float) -> Optional[int]:
        """Return index of the closest colour cluster, or None if the pixel is
        not note-like (too desaturated / dark -> background or glow tail)."""
        if sat < self.sat_min or val < self.val_min:
            return None
        if self.rainbow:
            return 0  # a single rainbow class; hand comes from position
        best, best_d = None, 1e9
        for i, c in enumerate(self.clusters):
            dh = min(abs(hue - c.hue), 180 - abs(hue - c.hue))
            if dh < best_d:
                best_d, best = dh, i
        return best if best_d <= 22 else None

    def to_dict(self) -> dict:
        return {
            "sat_min": self.sat_min, "val_min": self.val_min, "rainbow": self.rainbow,
            "clusters": [
                {"hue": c.hue, "sat": c.sat, "val": c.val, "weight": c.weight,
                 "hand_hint": c.hand_hint}
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
    # vividly saturated — some visualisers use pale/pastel colours (e.g. a gold
    # left hand at saturation ~50) that a fixed sat_min=70 rejects outright,
    # collapsing that hand's notes to just the solid label. The dark background
    # sits near zero saturation, so a low floor still separates it, and the
    # saturation-squared hue weighting below keeps faint fringe from outvoting a
    # real colour. The detector's actual sat_min is derived from the palest
    # detected cluster at the end (never above the caller's cap).
    cap_sat_min = sat_min
    floor = 30
    hues = []
    sats = []
    vals = []
    for hsv in frames_hsv_roll:
        h = hsv[..., 0].ravel()
        s = hsv[..., 1].ravel()
        v = hsv[..., 2].ravel()
        m = (s >= floor) & (v >= val_min)
        hues.append(h[m])
        sats.append(s[m])
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
    val_min = int(np.clip(max(val_min, 0.55 * core_level), 90, 210))
    # Weight the hue histogram by saturation (real note colours are strongly
    # saturated, suppressing desaturated fringe, §8) AND normalise each frame's
    # contribution before summing, so a hue is scored by how CONSISTENTLY it
    # appears across frames rather than by raw pixel count. This stops a few
    # outlier frames — an intro/transition/logo in another colour — from
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

    # find up to 3 dominant, well-separated hue peaks
    peaks = _hue_peaks(hist, min_sep=12, max_peaks=3)
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
    # drop negligible clusters (compression fringe, particle tints) but always
    # keep at least the two dominant hand colours if present.
    kept = [c for i, c in enumerate(clusters) if i < 2 or c.weight >= 0.15]
    # Detector saturation floor from the palest kept colour, so a pastel hand is
    # admitted while the near-zero background stays out. Never above the cap.
    pale = min((c.sat for c in kept), default=cap_sat_min)
    final_sat_min = int(np.clip(0.55 * pale, 30, cap_sat_min))
    return ThemeModel(clusters=kept, sat_min=final_sat_min, val_min=val_min)


def _otsu(x: np.ndarray) -> int:
    """Generic Otsu split (0..255) — the value maximising between-class variance,
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
