"""Re-render the reconstruction as a Synthesia-style visualisation (§24, §29).

Given the reconstructed notes plus the detected geometry and fall speed, this
reproduces, for any timestamp, where each note bar *should* appear. It is used
both for the self-verification loop (render -> compare -> correct) and to drive
the difference-overlay view in the UI.
"""
from __future__ import annotations

import cv2
import numpy as np

from .keyboard import KeyboardGeometry
from .model import NoteEvent, Hand, is_black_key

_HAND_BGR = {
    Hand.LEFT: (232, 158, 46),    # blue-ish
    Hand.RIGHT: (86, 196, 74),    # green-ish
    Hand.UNKNOWN: (180, 180, 180),
}


def note_bar_at(note: NoteEvent, geom: KeyboardGeometry, v: float, t: float):
    """Predicted (y_top, y_bottom) of a note's bar in roll coords at time ``t``,
    or None if the bar is not within the visible roll at that instant."""
    strike = geom.strike_y
    lead = strike - v * (note.start - t)         # leading (bottom) edge
    trail = strike - v * (note.end - t)          # trailing (top) edge
    y_bottom = min(lead, strike)
    y_top = trail
    if y_bottom < 0 or y_top > strike:
        return None
    return max(0.0, y_top), min(strike, y_bottom)


def render_frame(geom: KeyboardGeometry, notes: list[NoteEvent], v: float,
                 t: float, glow: bool = True) -> np.ndarray:
    img = np.zeros((geom.height, geom.width, 3), np.uint8)
    for n in notes:
        span = note_bar_at(n, geom, v, t)
        if span is None:
            continue
        y_top, y_bottom = span
        lane = geom.lanes.get(n.midi)
        if lane is None:
            continue
        x0 = int(round(lane.x0)); x1 = int(round(lane.x1))
        colour = _HAND_BGR[n.hand]
        cv2.rectangle(img, (x0, int(y_top)), (x1, int(y_bottom)), colour, -1)
    _draw_keyboard(img, geom)
    return img


def _draw_keyboard(img, geom: KeyboardGeometry):
    top = int(geom.keyboard_top); bot = int(geom.keyboard_bottom)
    cv2.rectangle(img, (0, top), (geom.width, bot), (250, 250, 250), -1)
    for m, lane in geom.lanes.items():
        if is_black_key(m):
            x0 = int(round(lane.x0)); x1 = int(round(lane.x1))
            h = int((bot - top) * 0.62)
            cv2.rectangle(img, (x0, top), (x1, top + h), (20, 20, 20), -1)
    cv2.line(img, (0, top), (geom.width, top), (0, 90, 200), 1)


def predicted_presence(notes: list[NoteEvent], geom: KeyboardGeometry, v: float,
                       t: float) -> dict[int, tuple[float, float]]:
    """Which lanes have a bar at time ``t`` and where — used by verification."""
    out: dict[int, tuple[float, float]] = {}
    for n in notes:
        span = note_bar_at(n, geom, v, t)
        if span is not None:
            out[n.midi] = span
    return out
