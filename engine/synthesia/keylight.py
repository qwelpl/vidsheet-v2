"""Key-highlight note detection (§55, primary path for notes-over-video).

Many visualisers light up each key on the keyboard while it is held. The
keyboard is a *static* region at a fixed location, so — unlike the falling bars
that get composited over moving footage — the lit-key signal is clean even when
the roll is drawn over full-motion video. This detector reads which keys are
pressed each frame directly:

    key lit   -> note on   (exact pitch from geometry)
    key unlit -> note off  (true held duration — the key-release, §11/§17)

Attack times are refined against the audio onsets (§18); a lit run that spans
several audio attacks is a repeated note and is split accordingly (§12).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .keyboard import KeyboardGeometry
from .model import NoteEvent, Hand, is_black_key
from .theme import ThemeModel


@dataclass
class KeyLightSampler:
    geom: KeyboardGeometry
    theme: ThemeModel
    sat_min: int = 80
    val_min: int = 70
    fill_thr: float = 0.4

    def __post_init__(self):
        g = self.geom
        self.kbtop = int(round(g.keyboard_top))
        self.kbbot = int(round(g.keyboard_bottom))
        self.kbh = max(4, self.kbbot - self.kbtop)
        self.hues = [c.hue for c in self.theme.clusters]

    def _region(self, lane) -> tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of the key body to sample, avoiding the top glow
        strip and the key separators."""
        if is_black_key(lane.midi):
            y0 = self.kbtop + int(self.kbh * 0.16)
            y1 = self.kbtop + int(self.kbh * 0.48)
            xw = lane.half_width * 0.7
        else:
            y0 = self.kbtop + int(self.kbh * 0.58)   # below the black keys
            y1 = self.kbbot - int(self.kbh * 0.05)
            xw = lane.half_width * 0.68
        x0 = max(0, int(round(lane.center - xw)))
        x1 = min(self.geom.width, int(round(lane.center + xw)) + 1)
        return y0, y1, x0, x1

    def sample(self, bgr: np.ndarray) -> dict[int, int]:
        """Return {midi: cluster_index} for every lit key in the frame."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lit: dict[int, int] = {}
        for midi, lane in self.geom.lanes.items():
            y0, y1, x0, x1 = self._region(lane)
            if y1 <= y0 or x1 <= x0:
                continue
            reg = hsv[y0:y1, x0:x1]
            h = reg[..., 0].astype(np.float32)
            s = reg[..., 1].astype(np.float32)
            v = reg[..., 2].astype(np.float32)
            best_frac, best_c = 0.0, -1
            for ci, hue in enumerate(self.hues):
                dh = np.minimum(np.abs(h - hue), 180 - np.abs(h - hue))
                m = (dh <= 18) & (s >= self.sat_min) & (v >= self.val_min)
                frac = float(m.mean())
                if frac > best_frac:
                    best_frac, best_c = frac, ci
            if best_frac >= self.fill_thr:
                lit[midi] = best_c
        return lit


def applicable(geom: KeyboardGeometry, theme: ThemeModel,
               sample_frames: list[np.ndarray], min_hit_frac: float = 0.25) -> bool:
    """Decide whether key highlights are present and usable: a good share of the
    sampled frames must show at least one lit key."""
    if not theme.clusters or theme.sat_min == 0:
        return False
    s = KeyLightSampler(geom, theme)
    hits = sum(1 for f in sample_frames if s.sample(f))
    return hits >= max(2, int(min_hit_frac * len(sample_frames)))


@dataclass
class _LaneLight:
    times: list[float] = field(default_factory=list)
    lit: list[int] = field(default_factory=list)   # cluster index, -1 = off


class KeyLightCollector:
    def __init__(self, geom: KeyboardGeometry, theme: ThemeModel):
        self.sampler = KeyLightSampler(geom, theme)
        self.geom = geom
        self.hist: dict[int, _LaneLight] = {m: _LaneLight() for m in geom.lanes}

    def add(self, time: float, bgr: np.ndarray) -> None:
        lit = self.sampler.sample(bgr)
        for m, lane in self.hist.items():
            lane.times.append(time)
            lane.lit.append(lit.get(m, -1))


def extract_notes(hist: dict[int, _LaneLight], geom: KeyboardGeometry, fps: float,
                  audio_onsets: np.ndarray, min_frames: int = 2) -> list[NoteEvent]:
    frame_dt = 1.0 / fps
    onsets = np.sort(audio_onsets) if audio_onsets is not None else np.array([])
    notes: list[NoteEvent] = []
    for midi, lane in hist.items():
        notes.extend(_lane_notes(midi, lane, geom, frame_dt, onsets, min_frames))
    notes.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    return notes


def _lane_notes(midi, lane: _LaneLight, geom, frame_dt, onsets, min_frames):
    times = lane.times
    lit = lane.lit
    n = len(times)
    out: list[NoteEvent] = []
    i = 0
    # A held note must survive brief lit-detection flicker (compression, a hand
    # passing over the key). Tolerate a few consecutive unlit frames before
    # calling it a release, so a sustained note is not chopped into pieces.
    max_gap = max(3, int(round(0.06 / max(frame_dt, 1e-3))))  # ~60 ms
    while i < n:
        if lit[i] < 0:
            i += 1
            continue
        j = i
        last_lit = i
        gap = 0
        clusters = [lit[i]]
        j = i + 1
        while j < n:
            if lit[j] >= 0:
                clusters.append(lit[j]); last_lit = j; gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        run_len = last_lit - i + 1
        if run_len >= min_frames:
            t_on = times[i]
            t_off = times[min(last_lit + 1, n - 1)]  # release ~ next frame edge
            cluster = int(np.bincount(clusters).argmax()) if clusters else 0
            # Split into repeated notes ONLY where the key is re-struck: an audio
            # attack that coincides with a visible dip in the key's illumination
            # (§12). A steadily-held key with no dip is never split just because
            # some OTHER note attacked during it.
            splits = [t for t in onsets
                      if t_on + 0.05 < t < t_off - 0.03 and _restruck(lit, times, t)]
            bounds = [t_on] + splits + [t_off]
            for a, b in zip(bounds, bounds[1:]):
                if b - a < frame_dt * 0.5:
                    continue
                out.append(_mk(midi, a, b, cluster, geom, run_len, len(splits) > 0))
        i = last_lit + 1
    return out


def _restruck(lit: list[int], times: list[float], t: float) -> bool:
    """Is there a brief unlit dip in this key near time ``t`` (a re-strike)?"""
    # nearest frame index to t
    k = int(np.searchsorted(times, t))
    for idx in range(max(0, k - 2), min(len(lit), k + 3)):
        if lit[idx] < 0:
            return True
    return False


def _mk(midi, start, end, cluster, geom, run_len, was_split) -> NoteEvent:
    ne = NoteEvent(id=0, midi=midi, start=float(start), end=float(max(end, start + 1e-3)),
                   lane_x=geom.lanes[midi].center if midi in geom.lanes else None)
    ne.color_cluster = cluster
    ne.detection_confidence = float(np.clip(run_len / 4.0, 0.5, 1.0))
    ne.pitch_confidence = geom.confidence
    ne.timing_confidence = 0.7
    ne.duration_confidence = 0.8
    ne.velocity = 80
    if was_split:
        ne.flag("repeat_split_by_audio")
    return ne
