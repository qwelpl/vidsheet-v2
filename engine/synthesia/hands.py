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


# A hand can span at most about a major ninth at once; a simultaneous group wider
# than this cannot belong to one hand (§16).
_REACH = 14
# Notes whose onsets fall inside this window are treated as struck together.
_CHORD_WINDOW = 0.03


def _assign_by_context(notes: list[NoteEvent]) -> None:
    """Colour gives no hand info: track two hands as moving pitch centres and,
    at each instant the keys go down together, split the simultaneous notes into a
    lower (left) and upper (right) group so that NEITHER group spans more than a
    hand can reach. Among the reachable splits, pick the one that moves each hand
    least from where it just was, so the assignment stays continuous instead of
    teleporting across a crude pitch boundary (which is what produced impossible
    octave-plus "chords" in one hand). Unsplittable clusters are still assigned but
    flagged for review."""
    if not notes:
        return
    order = sorted(range(len(notes)), key=lambda i: notes[i].start)
    allm = [notes[i].midi for i in order]
    # start the hands apart, around the lower and upper quartiles of the material
    lx = float(np.percentile(allm, 25))
    rx = float(np.percentile(allm, 75))
    if rx - lx < 2:
        lx, rx = min(allm), max(allm)

    N = len(order)
    i = 0
    while i < N:
        t0 = notes[order[i]].start
        j = i
        while j < N and notes[order[j]].start - t0 <= _CHORD_WINDOW:
            j += 1
        cluster = sorted((order[k] for k in range(i, j)),
                         key=lambda idx: notes[idx].midi)
        pm = [notes[idx].midi for idx in cluster]
        m = len(cluster)

        # try every contiguous split; left = lowest k pitches, right = the rest.
        # A reachable split is always preferred; among reachable ones take the one
        # that moves the hands least. When none is reachable (a cluster too wide for
        # two hands - almost always a detection artifact or pedal sustain) pick the
        # split that makes the worse hand as narrow as possible, then flag it.
        best = None  # (infeasible?, primary, cost, k)
        for k in range(m + 1):
            L, R = pm[:k], pm[k:]
            span_l = (L[-1] - L[0]) if L else 0
            span_r = (R[-1] - R[0]) if R else 0
            infeasible = span_l > _REACH or span_r > _REACH
            mean_l = float(np.mean(L)) if L else lx
            mean_r = float(np.mean(R)) if R else rx
            cost = abs(mean_l - lx) + abs(mean_r - rx)
            primary = max(span_l, span_r) if infeasible else cost
            cand = (infeasible, primary, cost, k)
            if best is None or cand < best:
                best = cand
        infeasible, _, _, k = best
        L, R = cluster[:k], cluster[k:]

        conf = 0.3 if infeasible else (0.7 if (L and R) else 0.6)
        for idx in L:
            notes[idx].hand = Hand.LEFT
            notes[idx].hand_confidence = conf
            if infeasible:
                notes[idx].flag("hand_unreachable")
        for idx in R:
            notes[idx].hand = Hand.RIGHT
            notes[idx].hand_confidence = conf
            if infeasible:
                notes[idx].flag("hand_unreachable")

        if L:
            lx = 0.6 * lx + 0.4 * float(np.mean([notes[idx].midi for idx in L]))
        if R:
            rx = 0.6 * rx + 0.4 * float(np.mean([notes[idx].midi for idx in R]))
        i = j
