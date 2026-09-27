"""Synthetic Synthesia video generator for ground-truth testing (§64, §65).

Because the source MIDI is known, transcription recovered from a generated video
can be scored exactly (pitch precision/recall, onset/offset error). These
renders are the regression fixtures and also power the app's built-in demo.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

from .model import is_black_key, midi_to_name


@dataclass
class SynthNote:
    midi: int
    start: float
    end: float
    hand: str = "right"       # 'left' | 'right'
    velocity: int = 90

    def to_truth(self) -> dict:
        return {"midi": self.midi, "start": self.start, "end": self.end,
                "hand": self.hand, "velocity": self.velocity,
                "name": midi_to_name(self.midi)}


@dataclass
class SynthConfig:
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    low_midi: int = 21
    high_midi: int = 108
    fall_speed: float = 320.0            # px per second
    keyboard_frac: float = 0.16          # keyboard height as fraction of frame
    left_colour: tuple = (232, 158, 46)  # BGR
    right_colour: tuple = (86, 196, 74)
    glow: bool = True
    particles: bool = True               # decoys the detector must reject (§35)


class SynthRenderer:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.kb_h = int(cfg.height * cfg.keyboard_frac)
        self.kb_top = cfg.height - self.kb_h
        self.whites = [m for m in range(cfg.low_midi, cfg.high_midi + 1)
                       if not is_black_key(m)]
        self.white_w = cfg.width / len(self.whites)
        self._centers: dict[int, float] = {}
        for i, m in enumerate(self.whites):
            self._centers[m] = (i + 0.5) * self.white_w
        for m in range(cfg.low_midi, cfg.high_midi + 1):
            if is_black_key(m):
                lo = self._centers.get(m - 1)
                hi = self._centers.get(m + 1)
                if lo is not None and hi is not None:
                    self._centers[m] = (lo + hi) / 2
                elif lo is not None:
                    self._centers[m] = lo + self.white_w / 2
                elif hi is not None:
                    self._centers[m] = hi - self.white_w / 2

    def lane(self, midi: int) -> tuple[float, float]:
        c = self._centers[midi]
        hw = self.white_w * (0.30 if is_black_key(midi) else 0.46)
        return c - hw, c + hw

    def render(self, notes: list[SynthNote], t: float) -> np.ndarray:
        cfg = self.cfg
        img = np.full((cfg.height, cfg.width, 3), 12, np.uint8)
        strike = self.kb_top
        for n in notes:
            lead = strike - cfg.fall_speed * (n.start - t)
            trail = strike - cfg.fall_speed * (n.end - t)
            y_bottom = min(lead, strike)
            y_top = trail
            if y_bottom < 0 or y_top > strike:
                continue
            x0, x1 = self.lane(n.midi)
            colour = cfg.left_colour if n.hand == "left" else cfg.right_colour
            b = np.array(colour, float) * (0.5 + 0.5 * n.velocity / 127.0)
            if cfg.glow:
                cv2.rectangle(img, (int(x0 - 4), int(max(0, y_top))),
                              (int(x1 + 4), int(y_bottom)),
                              tuple((b * 0.35).astype(int).tolist()), -1)
            cv2.rectangle(img, (int(x0), int(max(0, y_top))),
                          (int(x1), int(y_bottom)),
                          tuple(b.astype(int).tolist()), -1)
        if cfg.particles:
            self._particles(img, notes, t, strike)
        self._keyboard(img, notes, t, strike)
        return img

    def _keyboard(self, img, notes, t, strike):
        cfg = self.cfg
        cv2.rectangle(img, (0, strike), (cfg.width, cfg.height), (248, 248, 248), -1)
        # white key separators
        for i in range(1, len(self.whites)):
            x = int(i * self.white_w)
            cv2.line(img, (x, strike), (x, cfg.height), (60, 60, 60), 1)
        # black keys
        for m in range(cfg.low_midi, cfg.high_midi + 1):
            if is_black_key(m):
                x0, x1 = self._centers[m] - self.white_w * 0.30, self._centers[m] + self.white_w * 0.30
                cv2.rectangle(img, (int(x0), strike),
                              (int(x1), strike + int(self.kb_h * 0.62)), (18, 18, 18), -1)
        # key-press highlight for currently sounding notes (§55)
        for n in notes:
            if n.start <= t < n.end:
                x0, x1 = self.lane(n.midi)
                colour = cfg.left_colour if n.hand == "left" else cfg.right_colour
                cv2.rectangle(img, (int(x0), strike + 2), (int(x1), cfg.height - 2),
                              tuple(int(c * 0.7 + 80) for c in colour), -1)
        cv2.line(img, (0, strike), (cfg.width, strike), (0, 90, 210), 2)

    def _particles(self, img, notes, t, strike):
        # burst of shrinking dots just above the strike line for active notes;
        # these move/shrink/fade unlike real bars and must be rejected (§35).
        for n in notes:
            if 0 <= (t - n.start) < 0.25:
                cx = int(sum(self.lane(n.midi)) / 2)
                age = (t - n.start) / 0.25
                for k in range(6):
                    ang = k * 60 + t * 200
                    r = int(18 * age)
                    px = int(cx + r * math.cos(math.radians(ang)))
                    py = int(strike - 6 - r)
                    cv2.circle(img, (px, py), max(1, int(3 * (1 - age))),
                               (200, 230, 255), -1)


def generate_video(path: str, notes: list[SynthNote], cfg: SynthConfig,
                   with_audio: bool = False) -> list[dict]:
    """Render the video with ffmpeg and return the ground-truth note list."""
    r = SynthRenderer(cfg)
    duration = max((n.end for n in notes), default=1.0) + 1.0
    n_frames = int(math.ceil(duration * cfg.fps))
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n_frames):
        t = f / cfg.fps
        img = r.render(notes, t)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return [n.to_truth() for n in notes]


# ---------------------------------------------------------------------------
# Built-in test pieces
# ---------------------------------------------------------------------------

def demo_piece() -> list[SynthNote]:
    """A short two-hand excerpt exercising chords, repeats, a fast run and a
    sustained note - a compact stress test for the pipeline."""
    N = []
    # left-hand chord progression (root-position triads); a lead-in lets every
    # note's approach to the strike line be observed.
    prog = [(48, 52, 55), (45, 48, 52), (50, 53, 57), (43, 47, 50)]
    t = 1.0
    for chord in prog:
        for m in chord:
            N.append(SynthNote(m, t, t + 0.9, "left", 70))
        t += 1.0
    # right-hand melody with a repeated note and a fast run (offset by lead-in)
    melody = [(72, 0.0, 0.4), (72, 0.5, 0.9), (74, 1.0, 1.4), (76, 1.5, 2.4),
              (77, 2.5, 2.7), (79, 2.7, 2.9), (81, 2.9, 3.1), (83, 3.1, 3.3),
              (84, 3.3, 4.0)]
    for m, s, e in melody:
        N.append(SynthNote(m, s + 1.0, e + 1.0, "right", 100))
    # fast 16th run (right hand) incl. black keys
    base = 5.2
    for i, m in enumerate([72, 74, 76, 77, 79, 81, 83, 84]):
        N.append(SynthNote(m, base + i * 0.08, base + i * 0.08 + 0.07, "right", 95))
    # black-key passage (right hand) + a black-key chord (left) to exercise
    # accidentals and guard against over-suppressing real black keys
    chrom = [61, 63, 66, 68, 70, 73, 75, 78]  # C#4 D#4 F#4 G#4 A#4 C#5 D#5 F#5
    for i, m in enumerate(chrom):
        N.append(SynthNote(m, 6.0 + i * 0.25, 6.0 + i * 0.25 + 0.22, "right", 88))
    for m in (42, 46, 49):  # F#2 A#2 C#3 sustained triad (left, all black)
        N.append(SynthNote(m, 8.2, 9.1, "left", 66))
    return N


def scale_piece() -> list[SynthNote]:
    return [SynthNote(60 + i, i * 0.3, i * 0.3 + 0.28, "right", 90) for i in range(13)]
