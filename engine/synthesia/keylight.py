"""Key-highlight note detection (§55, primary path for notes-over-video).

Many visualisers light up each key on the keyboard while it is held. The
keyboard is a *static* region at a fixed location, so - unlike the falling bars
that get composited over moving footage - the lit-key signal is clean even when
the roll is drawn over full-motion video. This detector reads which keys are
pressed each frame directly:

    key lit   -> note on   (exact pitch from geometry)
    key unlit -> note off  (true held duration - the key-release, §11/§17)

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


def _far_cover(mask: np.ndarray) -> float:
    """Coverage of ``mask`` measured over the key's FAR half - the end away from
    the strike line. A press fills the whole key, so its far half stays solid; a
    strike-line glow/smoke halo pools at the near edge and decays out before
    reaching the far half, so its coverage there collapses toward zero. Scoring
    the far half alone (rather than the whole patch, which a halo inflates, or the
    near half, which the strike-line edge darkens even on a real press) is what
    separates the two. Parameter-free: the split is the key's own midline, the
    near/far sense fixed by the keyboard geometry (strike line at the top)."""
    h = mask.shape[0]
    if h < 2:
        return float(mask.mean()) if mask.size else 0.0
    return float(mask[h // 2:].mean())


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
        # A pale hand colour (e.g. a lavender right hand at saturation ~56) never
        # clears a fixed sat_min of 80, so its lit keys read as unlit or get
        # misattributed to the vivid hand's cluster - collapsing hand assignment.
        # Follow the theme's own detected saturation floor instead, clamped so a
        # near-grey keybed shadow still can't masquerade as a lit note.
        self.sat_min = int(np.clip(self.theme.sat_min, 35, self.sat_min))
        # Translucent particle smoke / glow rises from played notes, shares the
        # hand HUE, and drifts over the keys. At the permissive ``sat_min`` (kept
        # low so a pale hand still registers, §55) that bleed floods the keys: a
        # colour plume over the treble reads every black key under it as
        # permanently pressed (the "fried" stuck-key clusters). But a real press
        # is an OPAQUE patch of note colour, which the diluted bleed is not, in two
        # independent ways - and a key is only counted lit where the colour is
        # opaque by at least one of them:
        #   (a) saturation near the note core. Taken as a fraction of the detected
        #       cluster saturation so it rejects diluted smoke yet still admits a
        #       genuinely pastel theme (whose cores are themselves low-saturation).
        #   (b) a spatially UNIFORM fill. A flat render paints the pressed key one
        #       solid colour (saturation variance ~0), even a pale one; smoke is
        #       turbulent (variance many times higher). This rescues pale, clean
        #       renders that (a) alone would reject, without re-admitting smoke.
        core_sat = min((c.sat for c in self.theme.clusters), default=float(self.sat_min))
        self.sat_solid = int(np.clip(0.9 * core_sat, self.sat_min, 160))
        self.uniform_tol = 6.0   # max in-patch saturation std for an opaque fill

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

    def sample_raw(self, bgr: np.ndarray) -> dict[int, tuple[int, float]]:
        """Return {midi: (best_cluster, fill_fraction)} for EVERY key, including
        sub-threshold coverage. Keeping the raw fraction (share of the key body
        matching the note colour) - not just a lit/unlit flag - is what lets us
        (a) separate a fast repeat that only *dims* between hits without clearing
        ``fill_thr`` (§12), and (b) place the attack sub-frame by interpolating
        where coverage crosses the lit threshold as the key lights up (§10, §18)."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        out: dict[int, tuple[int, float]] = {}
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
                # candidate note-coloured pixels at the permissive floor
                hue_m = (dh <= 18) & (s >= self.sat_min) & (v >= self.val_min)
                # Glow and particle smoke are a HALO that climbs from the strike
                # line: colour pools at the near (top) edge of the key body and
                # fades out before the far end, where a real press is still solidly
                # filled. So score each key by its FAR half only (see ``_far_cover``)
                # instead of the whole-patch mean, which a halo inflates - a
                # top-weighted bleed collapses to ~0 there while a genuine press,
                # uniform top-to-bottom, stays high. This is the press/halo geometry
                # itself, carrying no colour- or video-specific constant (§55).
                cover = _far_cover(hue_m)
                # opaque fraction: the matched pixels that are near-core saturated
                frac = _far_cover(hue_m & (s >= self.sat_solid))
                # ...or rescue a flat, solidly-filled low-saturation patch (a clean
                # pale render) that the saturation gate alone would miss, while a
                # turbulent smoke patch - never spatially uniform - stays rejected.
                if cover >= 0.5 and float(s[hue_m].std()) <= self.uniform_tol:
                    frac = cover
                if frac > best_frac:
                    best_frac, best_c = frac, ci
            out[midi] = (best_c, best_frac)
        return out

    def sample(self, bgr: np.ndarray) -> dict[int, tuple[int, float]]:
        """Lit keys only (coverage >= ``fill_thr``). Used to test applicability."""
        return {m: (c, f) for m, (c, f) in self.sample_raw(bgr).items()
                if f >= self.fill_thr}


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
    lit: list[int] = field(default_factory=list)     # cluster index, -1 = off
    fill: list[float] = field(default_factory=list)   # colour coverage 0..1


class KeyLightCollector:
    def __init__(self, geom: KeyboardGeometry, theme: ThemeModel):
        self.sampler = KeyLightSampler(geom, theme)
        self.geom = geom
        self.hist: dict[int, _LaneLight] = {m: _LaneLight() for m in geom.lanes}

    def add(self, time: float, bgr: np.ndarray) -> None:
        raw = self.sampler.sample_raw(bgr)
        thr = self.sampler.fill_thr
        for m, lane in self.hist.items():
            lane.times.append(time)
            c, f = raw.get(m, (-1, 0.0))
            lane.lit.append(c if f >= thr else -1)
            lane.fill.append(f)


def extract_notes(hist: dict[int, _LaneLight], geom: KeyboardGeometry, fps: float,
                  audio_onsets: np.ndarray, min_frames: int = 2,
                  fill_thr: float = 0.4, solid_fill: float = 0.6) -> list[NoteEvent]:
    frame_dt = 1.0 / fps
    onsets = np.sort(audio_onsets) if audio_onsets is not None else np.array([])
    notes: list[NoteEvent] = []
    for midi, lane in hist.items():
        notes.extend(_lane_notes(midi, lane, geom, frame_dt, onsets, min_frames,
                                 fill_thr, solid_fill))
    notes.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    return notes


def _subframe_onset(fill, times, i, thr) -> float:
    """Attack instant: interpolate where coverage crosses ``thr`` between the last
    dark frame and the first lit one, so the onset is placed to a fraction of a
    frame instead of snapping to the frame grid (the main source of tiny
    inter-hand timing skew)."""
    if i <= 0:
        return times[i]
    f0, f1 = fill[i - 1], fill[i]
    if f1 <= f0 or f1 < thr:
        return times[i]
    frac = min(max((thr - f0) / (f1 - f0), 0.0), 1.0)
    return times[i - 1] + frac * (times[i] - times[i - 1])


def _subframe_offset(fill, times, last_lit, n, thr) -> float:
    """Release instant: interpolate where coverage falls back through ``thr``
    between the last lit frame and the first dark one - avoids the full-frame
    overhang of snapping the release to ``times[last_lit+1]``."""
    if last_lit + 1 >= n:
        dt = (times[last_lit] - times[last_lit - 1]) if last_lit > 0 else 0.0
        return times[last_lit] + dt
    f0, f1 = fill[last_lit], fill[last_lit + 1]
    if f0 <= f1:
        return times[last_lit + 1]
    frac = min(max((f0 - thr) / (f0 - f1), 0.0), 1.0)
    return times[last_lit] + frac * (times[last_lit + 1] - times[last_lit])


def _lane_notes(midi, lane: _LaneLight, geom, frame_dt, onsets, min_frames,
                fill_thr=0.4, solid_fill=0.6):
    times = lane.times
    lit = lane.lit
    fill = lane.fill
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
        # A genuine key-light fills the whole key body (coverage ~0.9). Notes
        # composited over moving footage bleed colour onto the keys too - and when
        # the footage shares a hand's hue (e.g. a blue ambience / sparkle effect
        # behind a blue left-hand track) a lane briefly covers past ``fill_thr``
        # without ever lighting solidly, spawning phantom notes (here, hundreds of
        # short left-hand slivers in the middle of the right hand). Require the run
        # to reach a solid peak coverage so a partial, flickering bleed is rejected
        # while a real light-up is kept (§55).
        peak_fill = max(fill[i:last_lit + 1], default=0.0)
        if run_len >= min_frames and peak_fill >= solid_fill:
            t_on = _subframe_onset(fill, times, i, fill_thr)
            t_off = _subframe_offset(fill, times, last_lit, n, fill_thr)
            cluster = int(np.bincount(clusters).argmax()) if clusters else 0
            # Split into repeated notes at every re-strike of THIS key. The
            # evidence is the key's own illumination envelope: a re-hit makes the
            # colour coverage dip (the key darkens as it lifts) and recover, even
            # when the dip never fully clears ``fill_thr`` and even when the bass
            # attack is too quiet to register a global audio onset. A steadily
            # held key has a flat envelope and is never split just because some
            # OTHER note attacked during it (§12).
            # Repeats are split ONLY from this key's own illumination envelope.
            # Audio onsets are unpitched, so in polyphonic music a melody attack
            # over a held bass/chord note would wrongly split the held note into
            # phantom repeats - the dominant error on busy notes-over-video. The
            # envelope trough is the only evidence specific to this key.
            splits = _restrike_times(fill, times, i, last_lit, frame_dt, min_frames)
            splits = _dedupe(sorted(splits), frame_dt * max(min_frames, 2))
            bounds = [t_on] + splits + [t_off]
            for a, b in zip(bounds, bounds[1:]):
                if b - a < frame_dt * 0.5:
                    continue
                out.append(_mk(midi, a, b, cluster, geom, run_len, len(splits) > 0))
        i = last_lit + 1
    return out


def _restrike_times(fill: list[float], times: list[float], i: int, last_lit: int,
                    frame_dt: float, min_frames: int) -> list[float]:
    """Split points inside a lit run [i..last_lit] from the coverage envelope.

    A re-strike shows as a prominent trough: coverage climbs to a peak, dips to a
    local minimum, then recovers. We require the dip to reach at most ``dip_frac``
    of the surrounding peak and to recover to ``rise_frac`` of it, and consecutive
    splits to be at least ``min_sep`` frames apart, so ordinary flicker and a
    single sustained note never split."""
    dip_frac = 0.6      # trough must fall to <=60% of the local peak
    rise_frac = 0.85    # ...and climb back to >=85% of it to count as a new hit
    min_sep = max(min_frames, 2)
    seg = fill[i:last_lit + 1]
    m = len(seg)
    if m < 2 * min_sep:
        return []
    splits: list[float] = []
    peak = seg[0]
    last_split_k = 0
    k = 1
    while k < m - 1:
        if seg[k] >= peak:
            peak = seg[k]
            k += 1
            continue
        # descending below the running peak: walk to the local minimum
        j = k
        vmin, kmin = seg[k], k
        while j < m and seg[j] < peak:
            if seg[j] < vmin:
                vmin, kmin = seg[j], j
            j += 1
        recovered = j < m and seg[j] >= peak * rise_frac
        deep = vmin <= peak * dip_frac
        if recovered and deep and (kmin - last_split_k) >= min_sep:
            splits.append(times[i + kmin])
            last_split_k = kmin
            peak = seg[j]
        k = max(j, k + 1)
    return splits


def _restruck(fill: list[float], times: list[float], t: float,
              dip_frac: float = 0.7) -> bool:
    """Is there a genuine re-strike trough in this key near time ``t``?

    A real re-attack dips below BOTH the coverage just before it AND just after:
    the key was lit, dimmed as it lifted, then re-lit. Requiring a two-sided
    trough is what rejects a note's own onset ramp - a monotonic rise to full
    coverage has no preceding peak to dip below - so an audio attack that merely
    coincides with a note lighting up no longer carves a sliver off its front."""
    n = len(fill)
    k = int(np.searchsorted(times, t))
    lo, hi = max(0, k - 3), min(n, k + 4)
    if hi - lo < 3 or k <= lo or k >= hi - 1:
        return False
    left_peak = max(fill[lo:k + 1])
    right_peak = max(fill[k:hi])
    vmin = min(fill[max(0, k - 1):min(n, k + 2)])
    if left_peak <= 0 or right_peak <= 0:
        return False
    return vmin <= dip_frac * left_peak and vmin <= dip_frac * right_peak


def _dedupe(ts: list[float], min_gap: float) -> list[float]:
    out: list[float] = []
    for t in ts:
        if not out or t - out[-1] >= min_gap:
            out.append(t)
    return out


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
