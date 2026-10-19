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

# Shortest plausible same-key re-strike. A real repeated note tops out around a
# dozen hits per second; anything faster is not a performance but glow flicker on
# the key body (worst on black keys, which sit right under the top glow strip)
# carving a spurious trough that reads as a re-attack. Repeats detected closer
# than this - envelope troughs or tripwire pulses alike - are dropped (§12).
MIN_RESTRIKE_SEC = 0.11


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
        self.achromatic = [c.achromatic for c in self.theme.clusters]
        self.achro_sat_max = self.theme.achromatic_sat_max
        self.achro_val_min = self.theme.achromatic_val_min
        # A pale hand colour (e.g. a lavender right hand at saturation ~56) never
        # clears a fixed sat_min of 80, so its lit keys read as unlit or get
        # misattributed to the vivid hand's cluster - collapsing hand assignment.
        # Follow the theme's own detected saturation floor instead, clamped so a
        # near-grey keybed shadow still can't masquerade as a lit note.
        self.sat_min = int(np.clip(self.theme.sat_min, 35, self.sat_min))
        # Tripwire band in the roll, just above the strike line. Each falling bar
        # crosses it as a pulse, so repeated notes - which keep a key continuously
        # lit (glow bridges the gap) and so never dip in the key illumination -
        # still read as separate bar arrivals here, giving the re-strike timing the
        # lit-key envelope cannot (§12). The key body stays the detector of record
        # for which notes exist and their hand; this only subdivides repeats.
        strike = int(round(self.geom.strike_y))
        self.trip_y1 = max(1, strike - 8)
        self.trip_y0 = max(0, self.trip_y1 - max(16, int(round(0.045 * strike))))
        # A band well ABOVE the strike line, clear of the key-glow halo that
        # spreads a few pixels up from a lit key. Only a genuine falling bar
        # reaches here; a key lit purely by a neighbour's bloom has no bar, so this
        # band's coverage confirms a note is real and not adjacent-key bleed.
        self.roll_y1 = max(1, strike - int(round(0.14 * strike)))
        self.roll_y0 = max(0, strike - int(round(0.26 * strike)))
        self._chrom_hues = [c.hue for c in self.theme.clusters if not c.achromatic]

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

    def sample_raw(self, bgr: np.ndarray = None, hsv: np.ndarray = None
                   ) -> dict[int, tuple[int, float]]:
        """Return {midi: (best_cluster, fill_fraction)} for EVERY key, including
        sub-threshold coverage. Keeping the raw fraction (share of the key body
        matching the note colour) - not just a lit/unlit flag - is what lets us
        (a) separate a fast repeat that only *dims* between hits without clearing
        ``fill_thr`` (§12), and (b) place the attack sub-frame by interpolating
        where coverage crosses the lit threshold as the key lights up (§10, §18)."""
        if hsv is None:
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
            # Chromatic clusters: assign each lit pixel to its NEAREST hand hue,
            # then score each cluster by the share of the key it uniquely owns.
            # Scoring each cluster by every pixel within a fixed hue radius instead
            # double-counts pixels when two hands have close colours (a red hand at
            # 3 and an orange hand at 11 are only 8 apart, well inside the radius):
            # a key lit in either colour then matches both, the tie breaks to the
            # first cluster, and every key collapses onto one hand (§7, §16).
            chrom = [ci for ci, a in enumerate(self.achromatic) if not a]
            if chrom:
                lit = (s >= self.sat_min) & (v >= self.val_min)
                dists = np.stack([
                    np.minimum(np.abs(h - self.hues[ci]), 180 - np.abs(h - self.hues[ci]))
                    for ci in chrom
                ])
                nearest = np.argmin(dists, axis=0)
                within = dists.min(axis=0) <= 18
                for k, ci in enumerate(chrom):
                    frac = float((lit & within & (nearest == k)).mean())
                    if frac > best_frac:
                        best_frac, best_c = frac, ci
            for ci, a in enumerate(self.achromatic):
                if a:
                    # white/grey hand: lit key is bright and unsaturated, matched
                    # by low saturation (not a meaningless hue) and a high value
                    # floor so dim low-sat background does not read as lit.
                    m = (s < self.achro_sat_max) & (v >= self.achro_val_min)
                    frac = float(m.mean())
                    if frac > best_frac:
                        best_frac, best_c = frac, ci
            out[midi] = (best_c, best_frac)
        return out

    def sample(self, bgr: np.ndarray) -> dict[int, tuple[int, float]]:
        """Lit keys only (coverage >= ``fill_thr``). Used to test applicability."""
        return {m: (c, f) for m, (c, f) in self.sample_raw(bgr).items()
                if f >= self.fill_thr}

    def sample_tripwire(self, hsv: np.ndarray) -> dict[int, float]:
        """Per-lane note-colour coverage in the roll tripwire band just above the
        strike line. Hand is irrelevant here (any note colour counts); this only
        reports whether a bar is crossing, so repeats can be split."""
        out: dict[int, float] = {}
        if not self._chrom_hues or self.trip_y1 <= self.trip_y0:
            return out
        band = hsv[self.trip_y0:self.trip_y1, :, :]
        h = band[..., 0].astype(np.float32)
        s = band[..., 1].astype(np.float32)
        v = band[..., 2].astype(np.float32)
        lit = np.zeros(h.shape, dtype=bool)
        for hue in self._chrom_hues:
            dh = np.minimum(np.abs(h - hue), 180 - np.abs(h - hue))
            lit |= (dh <= 18) & (s >= self.sat_min) & (v >= self.val_min)
        for midi, lane in self.geom.lanes.items():
            xw = lane.half_width * 0.55
            x0 = max(0, int(round(lane.center - xw)))
            x1 = min(self.geom.width, int(round(lane.center + xw)) + 1)
            if x1 <= x0:
                out[midi] = 0.0
            else:
                out[midi] = float(lit[:, x0:x1].mean())
        return out

    def sample_roll(self, hsv: np.ndarray) -> dict[int, float]:
        """Per-lane note-colour coverage in the roll band well above the strike
        line, where only a falling bar reaches (not key-glow bloom). Confirms a lit
        key is backed by an actual note rather than a neighbour's bleed."""
        out: dict[int, float] = {}
        if not self._chrom_hues or self.roll_y1 <= self.roll_y0:
            return out
        band = hsv[self.roll_y0:self.roll_y1, :, :]
        h = band[..., 0].astype(np.float32)
        s = band[..., 1].astype(np.float32)
        v = band[..., 2].astype(np.float32)
        lit = np.zeros(h.shape, dtype=bool)
        for hue in self._chrom_hues:
            dh = np.minimum(np.abs(h - hue), 180 - np.abs(h - hue))
            lit |= (dh <= 18) & (s >= self.sat_min) & (v >= self.val_min)
        for midi, lane in self.geom.lanes.items():
            xw = lane.half_width * 0.55
            x0 = max(0, int(round(lane.center - xw)))
            x1 = min(self.geom.width, int(round(lane.center + xw)) + 1)
            out[midi] = 0.0 if x1 <= x0 else float(lit[:, x0:x1].mean())
        return out


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
    trip: list[float] = field(default_factory=list)   # roll tripwire coverage 0..1
    roll: list[float] = field(default_factory=list)   # falling-bar coverage 0..1


class KeyLightCollector:
    def __init__(self, geom: KeyboardGeometry, theme: ThemeModel):
        self.sampler = KeyLightSampler(geom, theme)
        self.geom = geom
        self.hist: dict[int, _LaneLight] = {m: _LaneLight() for m in geom.lanes}

    def add(self, time: float, bgr: np.ndarray) -> None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        raw = self.sampler.sample_raw(hsv=hsv)
        trip = self.sampler.sample_tripwire(hsv)
        roll = self.sampler.sample_roll(hsv)
        thr = self.sampler.fill_thr
        for m, lane in self.hist.items():
            lane.times.append(time)
            c, f = raw.get(m, (-1, 0.0))
            lane.lit.append(c if f >= thr else -1)
            lane.fill.append(f)
            lane.trip.append(trip.get(m, 0.0))
            lane.roll.append(roll.get(m, 0.0))


def extract_notes(hist: dict[int, _LaneLight], geom: KeyboardGeometry, fps: float,
                  audio_onsets: np.ndarray, min_frames: int = 2,
                  fill_thr: float = 0.4) -> list[NoteEvent]:
    frame_dt = 1.0 / fps
    onsets = np.sort(audio_onsets) if audio_onsets is not None else np.array([])
    notes: list[NoteEvent] = []
    for midi, lane in hist.items():
        notes.extend(_lane_notes(midi, lane, geom, frame_dt, onsets, min_frames,
                                 fill_thr))
    notes = _drop_glow_bleed(notes)
    notes = _drop_bloom_adjacent(notes, hist, geom)
    notes.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    return notes


# Strong bloom can light a key one or two semitones from a struck key to full
# brightness, so its coverage is indistinguishable from a real press (the fill
# test above only catches dim bleed). But bloom is a KEY-glow effect: the phantom
# key has no falling bar. A real note's bar sweeps the roll band well above the
# strike line in the ~second before it sounds; a bloom phantom's lane stays empty
# there. So drop a note that has a brighter same-time neighbour within two
# semitones AND no bar support of its own, while the neighbour has one - a genuine
# tone cluster keeps both (both bars present) and an isolated note keeps itself
# (no neighbour to bleed from). §7/§16.
_ROLL_SUPPORT = 0.30       # bar coverage that confirms a real falling bar
_ROLL_STRONG = 0.85        # a neighbour this well-barred is unambiguously real
_ROLL_REL = 0.80           # a minor-2nd note under this fraction of it is bloom
_ROLL_LEAD_LO = 0.05       # seconds before onset: window where the bar is aloft
_ROLL_LEAD_HI = 1.00


def _roll_support(lane: "_LaneLight", t: float) -> float:
    """Peak falling-bar coverage in this lane over [t-HI, t-LO] before onset t."""
    if not lane.roll:
        return 0.0
    ts = lane.times
    lo = t - _ROLL_LEAD_HI
    hi = t - _ROLL_LEAD_LO
    best = 0.0
    for k in range(len(ts)):
        if lo <= ts[k] <= hi and lane.roll[k] > best:
            best = lane.roll[k]
    return best


def _drop_bloom_adjacent(notes: list[NoteEvent], hist: dict[int, "_LaneLight"],
                         geom: KeyboardGeometry) -> list[NoteEvent]:
    order = sorted(notes, key=lambda n: n.start)
    support = {id(n): _roll_support(hist[n.midi], n.start)
               if n.midi in hist else 1.0 for n in order}
    drop: set[int] = set()
    for n in order:
        sn = support[id(n)]
        for o in order:
            dm = abs(o.midi - n.midi)
            if o is n or not (1 <= dm <= 2):
                continue
            overlap = min(n.end, o.end) - max(n.start, o.start)
            if overlap <= 0.5 * n.duration:
                continue
            so = support[id(o)]
            # (a) n has no bar at all while the neighbour clearly does: pure bloom.
            # (b) a minor-second filler inside a chord: the neighbour sits on a
            # solid bar and n's bar support is markedly weaker - a real cluster
            # would back both keys about equally, so only the bloom key is dropped.
            if (sn < _ROLL_SUPPORT and so >= _ROLL_SUPPORT) or (
                    dm <= 2 and so >= _ROLL_STRONG and sn < _ROLL_REL * so):
                drop.add(id(n))
                break
    return [n for n in order if id(n) not in drop]


# A real key press fills most of the key body; the bloom from a bright held key
# only half-fills the key one or two semitones away. So a note whose peak
# coverage is weak AND which overlaps a clearly brighter neighbour within a
# couple of semitones is glow bleed, not a played note, and is dropped. A genuine
# tone cluster survives because both keys light fully (neither is weak), and an
# isolated faint note survives because it has no brighter neighbour to bleed from.
_BLEED_FILL_MAX = 0.70     # peak coverage at/below which a note may be bleed
_BLEED_BRIGHTER = 0.20     # neighbour must out-fill it by at least this much


def _drop_glow_bleed(notes: list[NoteEvent]) -> list[NoteEvent]:
    order = sorted(notes, key=lambda n: n.start)
    kept: list[NoteEvent] = []
    for n in order:
        if n.peak_fill <= _BLEED_FILL_MAX:
            bleed = False
            for o in order:
                if o is n or not (1 <= abs(o.midi - n.midi) <= 2):
                    continue
                overlap = min(n.end, o.end) - max(n.start, o.start)
                if (overlap > 0.5 * n.duration
                        and o.peak_fill >= n.peak_fill + _BLEED_BRIGHTER):
                    bleed = True
                    break
            if bleed:
                continue
        kept.append(n)
    return kept


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
                fill_thr=0.4):
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
        dark = 0
        clusters = [lit[i]]
        j = i + 1
        while j < n:
            if lit[j] >= 0:
                clusters.append(lit[j]); last_lit = j; gap = 0; dark = 0
            else:
                gap += 1
                # Distinguish a real key RELEASE from brief lit-detection flicker.
                # A fast same-key repeat drops the key fully dark (coverage ~0) for
                # a frame or two between hits; a flicker only dims it. Break the run
                # on a full release so the repeat stays a separate note, but keep
                # bridging a dim flicker up to ``max_gap`` so a held note is not
                # chopped up (§12).
                if fill[j] < 0.15:
                    dark += 1
                    if dark >= 2:
                        break
                else:
                    dark = 0
                if gap > max_gap:
                    break
            j += 1
        run_len = last_lit - i + 1
        if run_len >= min_frames:
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
            # Repeats that keep the key continuously lit (glow bridges the gap)
            # leave no trough in the key illumination, so the envelope split above
            # misses them. The falling bars still separate in the roll: add a split
            # at each bar that crosses the tripwire band during this lit run (§12).
            if lane.trip:
                tw = _tripwire_restrikes(lane.trip, times, i, last_lit, fill_thr)
                # A real re-attack makes a sound, so keep a tripwire split only
                # when an audio attack lands near it. This rejects tripwire pulses
                # driven by moving footage behind the roll (embers, light streaks)
                # that have no corresponding note, without needing the illumination
                # to dip - which is exactly what glow-bridged repeats never do.
                if onsets is not None and len(onsets):
                    tw = [t for t in tw if _near(onsets, t, 0.06)]
                splits += tw
            splits = _dedupe(sorted(splits),
                             max(frame_dt * max(min_frames, 2), MIN_RESTRIKE_SEC))
            # Drop any split sitting within a re-strike interval of the run's own
            # ends: a tripwire pulse or envelope dip that close to the onset/offset
            # is not a second hit, just the attack or release of this one note, and
            # keeping it would split off a sliver note (§12).
            splits = [s for s in splits
                      if s - t_on >= MIN_RESTRIKE_SEC and t_off - s >= MIN_RESTRIKE_SEC]
            run_peak = max(fill[i:last_lit + 1], default=1.0)
            bounds = [t_on] + splits + [t_off]
            for a, b in zip(bounds, bounds[1:]):
                if b - a < frame_dt * 0.5:
                    continue
                out.append(_mk(midi, a, b, cluster, geom, run_len,
                               len(splits) > 0, run_peak))
        i = last_lit + 1
    # Coalesce any two consecutive hits on this key whose onsets fall closer than
    # a plausible re-strike. The split guards above cover dips inside one lit run;
    # this also catches the other origin - glow flicker that drops the key fully
    # dark for a frame or two, which the main loop reads as a release and a fresh
    # attack. Either way a sub-re-strike gap is one note, not two (§12).
    out.sort(key=lambda ne: ne.start)
    merged: list[NoteEvent] = []
    for ne in out:
        if merged and ne.start - merged[-1].start < MIN_RESTRIKE_SEC:
            prev = merged[-1]
            prev.end = max(prev.end, ne.end)
            continue
        merged.append(ne)
    return merged


def _restrike_times(fill: list[float], times: list[float], i: int, last_lit: int,
                    frame_dt: float, min_frames: int) -> list[float]:
    """Split points inside a lit run [i..last_lit] from the coverage envelope.

    A re-strike shows as a prominent trough: coverage dips to a local minimum,
    well below the surrounding coverage, then recovers. The dip must fall to at
    most ``dip_frac`` of the LOCAL coverage on each side and recover to at least
    ``rise_frac`` of the level before it, and consecutive splits must be at least
    ``min_sep`` frames apart, so ordinary flicker and a sustained note never
    split. Measuring against the local level on each side (not a running peak over
    the whole run) is essential: one frame a touch brighter than a long plateau
    would otherwise set the peak just above the plateau, and every later recovery
    would fall short of it, so a continuously-lit repeat's dips were all missed."""
    dip_frac = 0.6      # trough must fall to <=60% of the local coverage
    rise_frac = 0.85    # ...and climb back to >=85% of the pre-dip level
    min_sep = max(min_frames, 2, int(round(MIN_RESTRIKE_SEC / max(frame_dt, 1e-3))))
    win = max(3, min_sep + 1)
    seg = fill[i:last_lit + 1]
    m = len(seg)
    if m < 2 * min_sep:
        return []
    splits: list[float] = []
    # Seed from the onset (k=0), not -min_sep, so the first trough must clear
    # min_sep frames of the note's own attack. Otherwise a dip one or two frames
    # in carves a sub-frame sliver off the front of the note (§12).
    last_split_k = 0
    k = 1
    while k < m - 1:
        # a local minimum: not higher than the frame before, strictly below after
        if seg[k] <= seg[k - 1] and seg[k] < seg[k + 1]:
            pre = max(seg[max(0, k - win):k + 1])
            post = max(seg[k:min(m, k + win + 1)])
            base = min(pre, post)
            if (base > 0 and seg[k] <= dip_frac * base
                    and post >= rise_frac * pre
                    and (k - last_split_k) >= min_sep):
                splits.append(times[i + k])
                last_split_k = k
        k += 1
    return splits


def _near(onsets: np.ndarray, t: float, tol: float) -> bool:
    """Is there an audio attack within ``tol`` seconds of ``t``?"""
    j = int(np.searchsorted(onsets, t))
    return min((abs(onsets[k] - t) for k in (j - 1, j) if 0 <= k < len(onsets)),
               default=1e9) <= tol


def _tripwire_restrikes(trip: list[float], times: list[float], i: int,
                        last_lit: int, thr: float) -> list[float]:
    """Split points from the roll tripwire: a bar crossing it drives coverage up,
    the gap before the next bar drops it back down. Each rising edge that follows a
    genuine clearing of the band (coverage fell well below the lit threshold) is a
    fresh bar arrival. Requiring the clear first rejects tripwire flicker and a
    single bar's own ragged edge, so a sustained note is never split.

    The very first rising edge is the note's own onset (already placed from the key
    illumination) and is not returned - only re-strikes inside the run are."""
    hi = min(last_lit, len(trip) - 1)
    out: list[float] = []
    for k in range(i + 1, hi + 1):
        # a rising edge: the band was clear (gap between bars) and a new bar enters
        if trip[k] >= thr and trip[k - 1] < thr:
            out.append(times[k])
    return out


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


def _mk(midi, start, end, cluster, geom, run_len, was_split, peak_fill=1.0) -> NoteEvent:
    ne = NoteEvent(id=0, midi=midi, start=float(start), end=float(max(end, start + 1e-3)),
                   lane_x=geom.lanes[midi].center if midi in geom.lanes else None)
    ne.color_cluster = cluster
    ne.peak_fill = float(peak_fill)
    ne.detection_confidence = float(np.clip(run_len / 4.0, 0.5, 1.0))
    ne.pitch_confidence = geom.confidence
    ne.timing_confidence = 0.7
    ne.duration_confidence = 0.8
    ne.velocity = 80
    if was_split:
        # "restrike" is the flag cleanup_fragments / resolve_same_pitch_overlaps
        # honour to keep a genuine re-attack from being fused back into one note.
        ne.flag("restrike")
    return ne
