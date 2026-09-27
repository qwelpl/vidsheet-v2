"""Left/right-hand assignment (§16).

Primary signal is the Synthesia note colour: each colour cluster is mapped to a
hand by the average pitch of the notes wearing it (the lower-centroid colour is
the left hand). When colour does not separate the hands (single-colour or
rainbow themes) a voice split is inferred from pitch context - but never with
the naive "low pitch = left" rule, because hand crossings occur; such notes get
a low ``hand_confidence`` so they surface for review.
"""
from __future__ import annotations

import numpy as np

from .model import NoteEvent, Hand
from .theme import ThemeModel


def assign_hands(notes: list[NoteEvent], theme: ThemeModel) -> None:
    if not notes:
        return
    n_clusters = len(theme.clusters)
    colour_separates = (n_clusters >= 2 and not theme.rainbow and theme.sat_min > 0)

    if colour_separates:
        _assign_by_colour(notes, n_clusters)
    else:
        _assign_by_context(notes)


def _assign_by_colour(notes: list[NoteEvent], n_clusters: int) -> None:
    # mean pitch per colour cluster
    pitches: dict[int, list[int]] = {}
    for n in notes:
        pitches.setdefault(n.color_cluster or 0, []).append(n.midi)
    means = {c: float(np.mean(v)) for c, v in pitches.items()}
    if len(means) < 2:
        _assign_by_context(notes)
        return
    ordered = sorted(means, key=lambda c: means[c])
    left_cluster = ordered[0]
    right_cluster = ordered[-1]
    # separation quality -> confidence
    sep = abs(means[right_cluster] - means[left_cluster])
    conf = float(np.clip(sep / 24.0, 0.5, 0.98))
    for n in notes:
        c = n.color_cluster or 0
        if c == left_cluster:
            n.hand = Hand.LEFT
        elif c == right_cluster:
            n.hand = Hand.RIGHT
        else:
            # extra cluster (e.g. 3-colour theme): nearest by pitch mean
            n.hand = Hand.LEFT if abs(means[c] - means[left_cluster]) < \
                abs(means[c] - means[right_cluster]) else Hand.RIGHT
        n.hand_confidence = conf


def _assign_by_context(notes: list[NoteEvent], window: float = 0.75) -> None:
    """Colour gives no hand info: split voices with an adaptive pitch boundary
    computed from a local time window. Crossing candidates are flagged."""
    times = np.array([n.start for n in notes])
    midis = np.array([n.midi for n in notes])
    for i, n in enumerate(notes):
        lo = np.searchsorted(times, n.start - window)
        hi = np.searchsorted(times, n.start + window)
        local = midis[lo:hi]
        if len(local) >= 4:
            boundary = float(np.median(local))
        else:
            boundary = 60.0  # middle C default
        n.hand = Hand.LEFT if n.midi < boundary else Hand.RIGHT
        margin = abs(n.midi - boundary)
        n.hand_confidence = float(np.clip(0.3 + margin / 18.0, 0.25, 0.7))
        if margin < 3:
            n.flag("hand_ambiguous")
