"""End-to-end reconstruction pipeline (§41, §42, §49, §62).

Multi-pass, streaming, deterministic. Passes:

  1. Geometry & theme  — median-frame keyboard detection + colour clustering.
  2. Track             — stream every frame, sample lanes, link tracks.
  3. Build notes       — tracks -> notes with sub-frame timing.
  4. Musical analysis  — hands, tempo, quantization.
  5. Verify            — visual comparison + automatic error search.

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

    roll_hsv = [cv2.cvtColor(f[:int(geom.strike_y), :, :], cv2.COLOR_BGR2HSV)
                for f in sample_frames]
    theme = detect_theme(roll_hsv)
    prog({"stage": "theme",
          "message": f"{len(theme.clusters)} note colour(s)"
                     f"{' (rainbow)' if theme.rainbow else ''}",
          "progress": 0.08})

    # --- Pass 2: stream & collect per-lane history ------------------------
    sampler = LaneSampler(geom, theme)
    collector = HistoryCollector(geom)
    observations: list[FrameObservation] = []
    n = 0
    total = max(1, meta.frame_count)
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
    prog({"stage": "track", "message": f"Fall speed {v:.0f} px/s",
          "progress": 0.72})

    # --- Pass 3: extract note events --------------------------------------
    notes = extract_notes(collector.hist, geom, meta.fps, v)
    prog({"stage": "reconstruct", "message": f"{len(notes)} notes reconstructed",
          "progress": 0.80})

    # --- Pass 4: musical analysis -----------------------------------------
    assign_hands(notes, theme)
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
