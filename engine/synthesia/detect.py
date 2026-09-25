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
                 column_frac: float = 1.0, min_run_px: int = 2,
                 min_fill: float = 0.35, center_fill: float = 0.72,
                 background: "np.ndarray | None" = None, motion_thr: int = 30,
                 motion_gate: bool = False, motion_alpha: float = 0.08):
        self.geom = geom
        self.theme = theme
        self.column_frac = column_frac
        self.min_run_px = min_run_px
        self.min_fill = min_fill
        self.center_fill = center_fill
        self.roll_top = 0
        self.strike_y = int(round(geom.strike_y))
        # Rolling-background motion gate (§32, §33, §35): note bars scroll, so at
        # any pixel they are only briefly present; an EMA background therefore
        # tracks the static content (keyboard, hit-line glow, composited footage)
        # and the moving bars stand out as the frame-vs-background difference.
        self.motion_thr = motion_thr
        self.motion_gate = motion_gate
        self.motion_alpha = motion_alpha
        self._bg = None  # EMA background over the roll region, float32
        self.bridge_px = max(3, int(round(geom.strike_y * 0.02)))
        # exclude a thin strip just above the strike line: many renderers draw a
        # coloured strike line / hit-flash there that would otherwise register as
        # a note in every lane on sparse clips.
        self.roll_bottom = max(1, self.strike_y - max(3, int(round(geom.strike_y * 0.008))))
        if background is not None:
            self._bg = background[self.roll_top: self.roll_bottom, :, :].astype(np.float32)

    def observe(self, index: int, time: float, bgr: np.ndarray) -> FrameObservation:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        roll = hsv[self.roll_top: self.roll_bottom, :, :]
        rh = roll.shape[0]
        moving = None
        if self.motion_gate:
            cur = bgr[self.roll_top: self.roll_bottom, :, :].astype(np.float32)
            if self._bg is None:
                self._bg = cur.copy()          # cold start: no motion this frame
            else:
                moving = np.abs(cur - self._bg).max(axis=2) >= self.motion_thr
                self._bg += self.motion_alpha * (cur - self._bg)   # EMA update
        runs: dict[int, list[LaneRun]] = {}
        for midi, lane in self.geom.lanes.items():
            half = max(1.5, lane.half_width * self.column_frac)
            x0 = max(0, int(round(lane.center - half)))
            x1 = min(roll.shape[1], int(round(lane.center + half)) + 1)
            if x1 - x0 < 2 or rh < 2:
                continue
            col = roll[:, x0:x1, :]
            hue = col[..., 0].astype(np.float32)
            sat = col[..., 1].astype(np.float32)
            val = col[..., 2].astype(np.float32)
            mask2d = self._core_mask(hue, sat, val)   # H x Wwin bool
            if moving is not None:
                mask2d &= moving[:, x0:x1]
            # Two gates decide a note is present in this lane, not a neighbour's
            # bleed: (a) the lane *centre* must be covered — a neighbouring
            # note's core edge never reaches the centre (§7); (b) enough of the
            # whole lane width is covered to reject single-pixel noise (§34).
            ci = int(round(lane.center)) - x0
            c0 = max(0, ci - 1); c1 = min(mask2d.shape[1], ci + 2)
            center = mask2d[:, c0:c1].mean(axis=1)
            fill = mask2d.mean(axis=1)
            row_active = (center >= self.center_fill) & (fill >= self.min_fill)
            # bridge thin gaps (seams, glow occlusion, compression) so a single
            # bar is not split; real repeated-note gaps are far larger (§12,§36)
            row_active = self._bridge(row_active, self.bridge_px)
            lane_runs = self._runs_from_mask(row_active, hue, sat, val, mask2d, rh)
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
        for c in t.clusters:
            dh = np.minimum(np.abs(hue - c.hue), 180 - np.abs(hue - c.hue))
            m |= (dh <= 20) & (sat >= t.sat_min) & (val >= t.val_min)
        return m

    def _runs_from_mask(self, mask, hue, sat, val, mask2d, rh) -> list[LaneRun]:
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
                m = mask2d[y:j]
                if m.any():
                    seg_h = hue[y:j][m]
                    seg_v = val[y:j][m]
                    seg_s = sat[y:j][m]
                else:
                    seg_h = hue[y:j].ravel(); seg_v = val[y:j].ravel(); seg_s = sat[y:j].ravel()
                cluster = self._dominant_cluster(seg_h)
                runs.append(LaneRun(
                    y_top=float(y), y_bottom=float(j - 1),
                    cluster=cluster,
                    val=float(seg_v.mean()), sat=float(seg_s.mean()),
                ))
            y = j
        return runs

    @staticmethod
    def _bridge(active: np.ndarray, gap: int) -> np.ndarray:
        if gap <= 0:
            return active
        out = active.copy()
        n = len(out)
        y = 0
        while y < n:
            if out[y]:
                y += 1
                continue
            j = y
            while j < n and not out[j]:
                j += 1
            # fill a short inactive gap that sits between two active runs
            if 0 < y and j < n and (j - y) <= gap:
                out[y:j] = True
            y = j
        return out

    def _dominant_cluster(self, hue) -> int:
        t = self.theme
        if t.rainbow or t.sat_min == 0:
            return 0
        counts = np.zeros(len(t.clusters))
        for i, c in enumerate(t.clusters):
            dh = np.minimum(np.abs(hue - c.hue), 180 - np.abs(hue - c.hue))
            counts[i] = int((dh <= 20).sum())
        return int(counts.argmax())
