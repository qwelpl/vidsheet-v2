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
    hues = []
    sats = []
    vals = []
    for hsv in frames_hsv_roll:
        h = hsv[..., 0].ravel()
        s = hsv[..., 1].ravel()
        v = hsv[..., 2].ravel()
        m = (s >= sat_min) & (v >= val_min)
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
    hist = np.bincount(H, minlength=180).astype(np.float64)
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 9), 0).ravel()

    # rainbow test: hue spread very wide and reasonably flat
    occupied = (hist > hist.max() * 0.15).sum()
    if occupied > 90:
        return ThemeModel(clusters=[ColorCluster(90, float(S.mean()), float(V.mean()), 1.0)],
                          sat_min=sat_min, val_min=val_min, rainbow=True)

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
    return ThemeModel(clusters=clusters, sat_min=sat_min, val_min=val_min)


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
