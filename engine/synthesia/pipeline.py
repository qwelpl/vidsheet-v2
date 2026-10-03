"""End-to-end reconstruction pipeline (§41, §42, §49, §62).

Multi-pass, streaming, deterministic. Passes:

  1. Geometry & theme - median-frame keyboard detection + colour clustering.
  2. Track - stream every frame, sample lanes, link tracks.
  3. Build notes - tracks -> notes with sub-frame timing.
  4. Musical analysis - hands, tempo, quantization.
  5. Verify - visual comparison + automatic error search.

Progress is reported through a callback with real stage/among-frame counts
(§42, never fake percentages).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import keyboard as kb
from .detect import LaneSampler, FrameObservation
from .events import HistoryCollector, extract_notes, estimate_fall_speed
from .exporters import notes_to_json
from .hands import assign_hands
from .model import VideoMeta, NoteEvent
from .theme import detect_theme, ThemeModel
from .tempo import estimate_tempo, quantize
from . import videoio
from .verify import visual_compare, search_errors, build_report, compare_ground_truth

import cv2

ProgressCb = Callable[[dict], None]


@dataclass
class Options:
    preset: str = "balanced"          # fast | balanced | maximum
    analysis_width: Optional[int] = None  # None = native
    verify_stride: int = 4            # keep every Nth observation for verify
    min_note_duration: float = 0.02
    geometry_manual: Optional[kb.KeyboardGeometry] = None
    detector: str = "auto"            # auto | bars | keylight

    @classmethod
    def for_preset(cls, preset: str) -> "Options":
        if preset == "fast":
            return cls(preset, analysis_width=854, verify_stride=8)
        if preset == "maximum":
            return cls(preset, analysis_width=None, verify_stride=2)
        return cls(preset, analysis_width=1280, verify_stride=4)


@dataclass
class AnalysisResult:
    meta: VideoMeta
    geometry: kb.KeyboardGeometry
    theme: ThemeModel
    notes: list[NoteEvent]
    tempo: object
    quantized: list[dict]
    report: object
    fall_speed: float
    diffs: list = field(default_factory=list)
    ground_truth: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "meta": self.meta.to_dict(),
            "geometry": self.geometry.to_dict(),
            "theme": self.theme.to_dict(),
            "notes": notes_to_json(self.notes),
            "tempo": self.tempo.to_dict(),
            "quantized": self.quantized,
            "report": self.report.to_dict(),
            "fall_speed": self.fall_speed,
            "ground_truth": self.ground_truth,
        }


def _scale_meta(meta: VideoMeta, width: Optional[int]) -> tuple[Optional[int], float]:
    if width is None or width >= meta.width:
        return None, 1.0
    return width, width / meta.width


def analyze(video_path: str, opts: Options,
            progress: Optional[ProgressCb] = None,
            truth: Optional[list[dict]] = None) -> AnalysisResult:
    prog = progress or (lambda d: None)
    meta = videoio.probe(video_path)
    prog({"stage": "prepare", "message":
          f"{meta.width}x{meta.height} @ {meta.fps:.3f}fps, {meta.duration:.1f}s",
          "progress": 0.0})

    scale_w, scale = _scale_meta(meta, opts.analysis_width)

    # --- Pass 1: geometry & theme -----------------------------------------
    prog({"stage": "keyboard", "message": "Detecting keyboard geometry", "progress": 0.02})
    sample_frames = _sample_frames(meta, scale_w, n=40)
    median = kb.temporal_median(sample_frames)
    if opts.geometry_manual is not None:
        geom = opts.geometry_manual
    else:
        geom = kb.build_geometry(median)
    prog({"stage": "keyboard",
          "message": f"Keyboard {geom_low_name(geom)}..{geom_high_name(geom)}, "
                     f"{len(geom.lanes)} lanes, conf {geom.confidence:.2f}",
          "progress": 0.06})

    roll_cut = max(1, int(geom.strike_y) - max(3, int(round(geom.strike_y * 0.008))))
    roll_hsv = [cv2.cvtColor(f[:roll_cut, :, :], cv2.COLOR_BGR2HSV)
                for f in sample_frames]
    theme = detect_theme(roll_hsv)
    prog({"stage": "theme",
          "message": f"{len(theme.clusters)} note colour(s)"
                     f"{' (rainbow)' if theme.rainbow else ''}",
          "progress": 0.08})

    # audio attacks (shared by both detectors for timing refinement, §18)
    try:
        from .audio import extract_onsets, snap_onsets
        audio_onsets = extract_onsets(video_path)
    except Exception:
        audio_onsets = None
        snap_onsets = None

    # Detector choice: key-highlight reading is far more robust when the notes
    # are composited over moving footage; falling-bar tracking is used for clean
    # renders. 'auto' picks key-highlight only when the keyboard clearly lights
    # up pressed keys.
    from . import keylight
    composited = _roll_is_composited(median, geom)
    use_keylight = opts.detector == "keylight" or (
        opts.detector == "auto" and composited
        and keylight.applicable(geom, theme, sample_frames))

    observations: list[FrameObservation] = []
    total = max(1, meta.frame_count)

    if use_keylight:
        prog({"stage": "theme",
              "message": "Key-highlight detection (notes-over-video mode)",
              "progress": 0.09})
        v = _quick_fall_speed(meta, geom, theme, scale_w)
        kcol = keylight.KeyLightCollector(geom, theme)
        # Track the falling bars in the SAME pass. The key light tells us which
        # key and for how long, but it fades in a few frames after the real
        # attack - and at a rate that differs with colour, skewing the two hands
        # apart. The bar's leading edge crossing the strike line is the exact,
        # colour-independent onset, so we use it to refine the timing below (§10).
        bar_sampler = LaneSampler(geom, theme)
        bar_coll = HistoryCollector(geom)
        n = 0
        for fr in videoio.stream_frames(meta, scale_width=scale_w):
            kcol.add(fr.time, fr.image)
            bar_coll.add(bar_sampler.observe(fr.index, fr.time, fr.image))
            n += 1
            if n % 150 == 0:
                down = sum(1 for l in kcol.hist.values() if l.lit and l.lit[-1] >= 0)
                prog({"stage": "track",
                      "message": f"Read {n} frames, {down} keys down",
                      "progress": 0.08 + 0.62 * min(1.0, n / total)})
        import numpy as _np
        notes = keylight.extract_notes(kcol.hist, geom, meta.fps,
                                       audio_onsets if audio_onsets is not None else _np.array([]),
                                       fill_thr=kcol.sampler.fill_thr)
        v_bars = estimate_fall_speed(bar_coll.hist, geom)
        bar_notes = extract_notes(bar_coll.hist, geom, meta.fps, v_bars)
        refined = _refine_onsets_from_bars(notes, bar_notes)
        prog({"stage": "reconstruct",
              "message": f"{len(notes)} notes (key-highlight; {refined} onsets from bars)",
              "progress": 0.80})
    else:
        sampler = LaneSampler(geom, theme)
        collector = HistoryCollector(geom)
        n = 0
        for fr in videoio.stream_frames(meta, scale_width=scale_w):
            obs = sampler.observe(fr.index, fr.time, fr.image)
            collector.add(obs)
            if n % opts.verify_stride == 0:
                observations.append(obs)
            n += 1
            if n % 120 == 0:
                active = sum(1 for h in collector.hist.values() if h.frames)
                prog({"stage": "track",
                      "message": f"Analysed {n} frames, {active} active lanes",
                      "progress": 0.08 + 0.62 * min(1.0, n / total)})
        v = estimate_fall_speed(collector.hist, geom)
        prog({"stage": "track", "message": f"Fall speed {v:.0f} px/s", "progress": 0.72})
        notes = extract_notes(collector.hist, geom, meta.fps, v)
        prog({"stage": "reconstruct", "message": f"{len(notes)} notes reconstructed",
              "progress": 0.80})

    # fuse same-pitch fragments / drop tiny slivers before musical analysis (§48)
    from .events import cleanup_fragments
    before = len(notes)
    notes = cleanup_fragments(notes)
    if before != len(notes):
        prog({"stage": "reconstruct",
              "message": f"Cleaned {before - len(notes)} fragment notes",
              "progress": 0.82})

    # --- Pass 4: musical analysis -----------------------------------------
    assign_hands(notes, theme)
    if snap_onsets and audio_onsets is not None and len(audio_onsets):
        refined = snap_onsets(notes, audio_onsets)
        from .audio import drop_unsupported_short
        dropped = drop_unsupported_short(notes, audio_onsets)
        msg = f"Audio verified timing on {refined} notes ({len(audio_onsets)} attacks)"
        if dropped:
            msg += f"; dropped {dropped} silent short artifacts"
        prog({"stage": "audio", "message": msg, "progress": 0.84})

    # align notes struck together (chords / both hands on a beat) so tiny
    # per-lane frame skew doesn't read as the hands being out of sync (§10)
    from .events import align_chords, resolve_same_pitch_overlaps
    aligned = align_chords(notes, audio_onsets)
    if aligned:
        prog({"stage": "audio", "message": f"Aligned {aligned} simultaneous onsets",
              "progress": 0.85})
    # a key can't sound twice at once: trim/drop any same-pitch overlaps that
    # timing nudges introduced, keeping genuine repeats distinct (§12)
    resolve_same_pitch_overlaps(notes)

    tempo = estimate_tempo(notes, meta.fps)
    quantized = quantize(notes, tempo)
    prog({"stage": "tempo", "message": f"~{tempo.bpm:.1f} BPM (conf {tempo.confidence:.2f})",
          "progress": 0.86})

    # --- Pass 5: verify + error search ------------------------------------
    search_errors(notes, meta.fps, opts.min_note_duration)
    diffs = visual_compare(notes, observations, geom, v)
    onset_unc = _onset_uncertainty_ms(meta.fps, v, geom)
    report = build_report(notes, diffs, geom, onset_unc,
                          ground_truth=truth is not None)
    prog({"stage": "verify", "message": report.notes, "progress": 0.96})

    gt = compare_ground_truth(notes, truth) if truth else None
    prog({"stage": "done", "message": "Reconstruction complete", "progress": 1.0})

    return AnalysisResult(meta=meta, geometry=geom, theme=theme, notes=notes,
                          tempo=tempo, quantized=quantized, report=report,
                          fall_speed=v, diffs=diffs, ground_truth=gt)


def _hitline_band_height(frames: list[np.ndarray], geom: kb.KeyboardGeometry,
                         theme: ThemeModel) -> int:
    """Height (px) of a persistent note-coloured band adjacent to the strike
    line, active across many lanes over time - a hit-line glow/reflection rather
    than real notes. Measured on the static-ish sample frames."""
    strike = int(geom.strike_y)
    look = min(strike, 40)
    counts = np.zeros(look, dtype=np.float64)
    lanes = list(geom.lanes.values())
    n = 0
    for f in frames:
        hsv = cv2.cvtColor(f[strike - look:strike, :, :], cv2.COLOR_BGR2HSV)
        for lane in lanes:
            x0 = max(0, int(lane.center - lane.half_width))
            x1 = min(hsv.shape[1], int(lane.center + lane.half_width) + 1)
            if x1 <= x0:
                continue
            col = hsv[:, x0:x1]
            hue = col[..., 0].astype(np.float32)
            sat = col[..., 1].astype(np.float32)
            val = col[..., 2].astype(np.float32)
            mask = np.zeros(hue.shape, bool)
            if theme.rainbow or theme.sat_min == 0:
                mask = (sat >= theme.sat_min) & (val >= theme.val_min)
            else:
                for c in theme.clusters:
                    dh = np.minimum(np.abs(hue - c.hue), 180 - np.abs(hue - c.hue))
                    mask |= (dh <= 20) & (sat >= theme.sat_min) & (val >= theme.val_min)
            counts[mask.mean(axis=1) >= 0.5] += 1
        n += 1
    if n == 0:
        return 0
    frac = counts / (n * max(1, len(lanes)))
    # walk up from the strike while a large fraction of lanes stay active
    h = 0
    for y in range(look - 1, -1, -1):
        if frac[y] > 0.3:
            h = look - y
        else:
            break
    return int(h)


def _refine_onsets_from_bars(notes, bar_notes, tol: float = 0.14) -> int:
    """Snap each key-light note's onset to its falling-bar strike-line crossing.

    The bar edge reaches the strike line at the true attack instant, with no
    colour-dependent fade lag, so it fixes both the absolute timing and the
    left/right-hand skew that the key-light fade introduces. Only the onset moves
    (within ``tol``); the key-light duration - a more reliable release signal -
    is preserved. A bar onset is claimed by at most one note of its pitch so two
    nearby notes don't collapse onto the same crossing."""
    import bisect
    import collections
    by: dict[int, list[float]] = collections.defaultdict(list)
    for b in bar_notes:
        by[b.midi].append(float(b.start))
    for m in by:
        by[m].sort()
    claimed: set = set()
    refined = 0
    for nte in sorted(notes, key=lambda x: x.start):
        cands = by.get(nte.midi)
        if not cands:
            continue
        i = bisect.bisect_left(cands, nte.start)
        best, bestd = None, tol
        for j in (i - 1, i):
            if 0 <= j < len(cands):
                d = abs(cands[j] - nte.start)
                key = (nte.midi, round(cands[j], 4))
                if d < bestd and key not in claimed:
                    bestd, best, bestkey = d, cands[j], key
        if best is None:
            continue
        claimed.add(bestkey)
        if abs(best - nte.start) > 1e-4:
            dur = nte.duration
            nte.start = float(best)
            nte.end = float(best) + dur
            nte.timing_confidence = min(1.0, max(nte.timing_confidence, 0.85))
            nte.flag("onset_from_bar")
            refined += 1
    return refined


def _roll_is_composited(median: np.ndarray, geom) -> bool:
    """Is the falling-note roll drawn over moving footage rather than a clean
    dark background? The temporal median averages notes away, leaving the static
    background; a clean render is near-black and edge-free, a video overlay is
    bright and textured. Composited roll -> prefer key-highlight detection."""
    strike = int(geom.strike_y)
    roll = median[: max(1, strike - 4), :, :]
    gray = cv2.cvtColor(roll, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    edges = float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean())
    return brightness > 40.0 or edges > 12.0


def _quick_fall_speed(meta: VideoMeta, geom, theme, scale_w) -> float:
    """Cheap fall-speed estimate for the reconstruction view when the primary
    detector is key-highlight (which doesn't track bars). Streams a few seconds
    of falling bars and fits their descent."""
    coll = HistoryCollector(geom)
    sampler = LaneSampler(geom, theme)
    start = meta.duration * 0.4
    n = 0
    for fr in videoio.stream_frames(meta, start=start, end=start + 5.0, scale_width=scale_w):
        coll.add(sampler.observe(fr.index, fr.time, fr.image))
        n += 1
        if n > 300:
            break
    v = estimate_fall_speed(coll.hist, geom)
    return v if v > 1 else geom.strike_y * 0.5


def _sample_frames(meta: VideoMeta, scale_w, n: int) -> list[np.ndarray]:
    frames = []
    stride = max(1, meta.frame_count // n)
    for fr in videoio.stream_frames(meta, scale_width=scale_w, step=stride):
        frames.append(fr.image.copy())
        if len(frames) >= n:
            break
    if not frames:  # extremely short clip
        for fr in videoio.stream_frames(meta, scale_width=scale_w):
            frames.append(fr.image.copy())
    return frames


def _onset_uncertainty_ms(fps: float, v: float, geom: kb.KeyboardGeometry) -> float:
    # sub-pixel crossing over one frame interval; ~half a pixel of edge noise
    px_noise = 0.5
    return (px_noise / max(v, 1e-3)) * 1000.0 + (1000.0 / fps) * 0.1


def geom_low_name(geom) -> str:
    from .model import midi_to_name
    return midi_to_name(geom.low_midi)


def geom_high_name(geom) -> str:
    from .model import midi_to_name
    return midi_to_name(geom.high_midi)
