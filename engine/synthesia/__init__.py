"""Synthesia reconstruction engine — a reverse compiler for piano-roll videos.

The video is treated as ground truth; the engine recovers the underlying
performance (notes, timing, hands, dynamics) with frame-level accuracy using
deterministic computer vision, exposing uncertainty rather than hiding it.
"""
from .pipeline import analyze, Options, AnalysisResult
from .model import NoteEvent, Hand, midi_to_name, name_to_midi

__all__ = ["analyze", "Options", "AnalysisResult", "NoteEvent", "Hand",
           "midi_to_name", "name_to_midi"]
__version__ = "0.1.0"
