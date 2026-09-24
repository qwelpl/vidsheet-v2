"""Per-frame note observation via lane-column sampling (§7, §9, §15, §34, §50).

Rather than segmenting the whole frame and then splitting overlapping 2-D blobs,
we sample a narrow vertical column at every pitch lane's centre. Each frame
yields, per lane, the set of note *cores* currently occupying that column. This:

  * isolates neighbouring pitches even when their glow overlaps (§15),
  * separates the solid core from translucent glow via a saturation gate (§34),
  * exposes gaps between repeated notes as gaps between runs (§12),
  * is inherently lane-accurate, preventing octave/neighbour confusion (§7).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .keyboard import KeyboardGeometry
from .theme import ThemeModel


@dataclass
class LaneRun:
    """A contiguous note core seen in one lane's column in one frame."""
    y_top: float        # smaller y (trailing edge, further from keyboard)
    y_bottom: float     # larger y (leading edge, nearer the strike line)
    cluster: int        # theme colour cluster index (colour -> hand hint)
    val: float          # mean brightness of the core (velocity cue, §23)
    sat: float


@dataclass
class FrameObservation:
    index: int
    time: float
    lane_runs: dict[int, list[LaneRun]]  # midi -> runs, ordered top..bottom


class LaneSampler:
    """Extracts :class:`FrameObservation` objects from frames."""

    def __init__(self, geom: KeyboardGeometry, theme: ThemeModel,
                 column_frac: float = 0.5, min_run_px: int = 2):
        self.geom = geom
        self.theme = theme
        self.column_frac = column_frac
        self.min_run_px = min_run_px
        self.roll_top = 0
        self.strike_y = int(round(geom.strike_y))

    def observe(self, index: int, time: float, bgr: np.ndarray) -> FrameObservation:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        roll = hsv[self.roll_top: self.strike_y, :, :]
        rh = roll.shape[0]
        runs: dict[int, list[LaneRun]] = {}
        for midi, lane in self.geom.lanes.items():
            half = max(1.0, lane.half_width * self.column_frac)
            x0 = int(round(lane.center - half))
            x1 = int(round(lane.center + half)) + 1
            x0 = max(0, x0)
            x1 = min(roll.shape[1], x1)
            if x1 <= x0:
                continue
            col = roll[:, x0:x1, :]
            # average the narrow window to suppress compression noise (§33)
            hue = col[..., 0].astype(np.float32).mean(axis=1)
            sat = col[..., 1].astype(np.float32).mean(axis=1)
            val = col[..., 2].astype(np.float32).mean(axis=1)
            mask = self._core_mask(hue, sat, val)
            lane_runs = self._runs_from_mask(mask, hue, sat, val, rh)
            if lane_runs:
                runs[midi] = lane_runs
        return FrameObservation(index=index, time=time, lane_runs=runs)

    def _core_mask(self, hue, sat, val) -> np.ndarray:
        t = self.theme
        if t.rainbow:
            return (sat >= t.sat_min) & (val >= t.val_min)
        if t.sat_min == 0:  # colourless / value-keyed theme
            return val >= t.val_min
        m = np.zeros(hue.shape, dtype=bool)
        for i, c in enumerate(t.clusters):
            dh = np.minimum(np.abs(hue - c.hue), 180 - np.abs(hue - c.hue))
            m |= (dh <= 20) & (sat >= t.sat_min) & (val >= t.val_min)
        return m

    def _runs_from_mask(self, mask, hue, sat, val, rh) -> list[LaneRun]:
        runs = []
        y = 0
        n = len(mask)
        while y < n:
            if not mask[y]:
                y += 1
                continue
            j = y
            while j < n and mask[j]:
                j += 1
            if j - y >= self.min_run_px:
                seg_h = hue[y:j]
                seg_v = val[y:j]
                seg_s = sat[y:j]
                cluster = self._dominant_cluster(seg_h, seg_s, seg_v)
                runs.append(LaneRun(
                    y_top=float(y), y_bottom=float(j - 1),
                    cluster=cluster,
                    val=float(seg_v.mean()), sat=float(seg_s.mean()),
                ))
            y = j
        return runs

    def _dominant_cluster(self, hue, sat, val) -> int:
        t = self.theme
        if t.rainbow or t.sat_min == 0:
            return 0
        counts = np.zeros(len(t.clusters))
        for i, c in enumerate(t.clusters):
            dh = np.minimum(np.abs(hue - c.hue), 180 - np.abs(hue - c.hue))
            counts[i] = int((dh <= 20).sum())
        return int(counts.argmax())
