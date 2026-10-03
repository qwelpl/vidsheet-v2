"""Secondary audio verification (§18).

Audio is used only to *refine* and *cross-check* the visual reconstruction,
never to override strong visual evidence. Its most reliable contribution is
attack timing: a spectral-flux onset detector locates note attacks to a few
milliseconds, far tighter than noisy visual tracking over composited footage.
Visual onsets are snapped to a nearby audio attack when one exists, which
removes timing scatter without changing which notes or pitches were detected.
"""
from __future__ import annotations

import subprocess
import wave
from typing import Optional

import numpy as np

from .model import NoteEvent, VideoMeta


def extract_onsets(video_path: str, sr: int = 22050) -> np.ndarray:
    """Return audio attack times (seconds) via spectral flux. Empty if no audio."""
    tmp = video_path + ".mono.wav"
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video_path,
         "-ac", "1", "-ar", str(sr), "-y", tmp],
        capture_output=True)
    if res.returncode != 0:
        return np.array([])
    try:
        w = wave.open(tmp)
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        w.close()
    except Exception:
        return np.array([])
    if x.size < 4096:
        return np.array([])

    win, hop = 2048, 512
    wnd = np.hanning(win)
    n_frames = 1 + (len(x) - win) // hop
    prev = None
    flux = np.zeros(n_frames)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + win] * wnd
        mag = np.abs(np.fft.rfft(seg))
        if prev is not None:
            flux[i] = np.maximum(0.0, mag - prev).sum()
        prev = mag
    if flux.max() <= 0:
        return np.array([])
    flux /= flux.max()

    # adaptive peak picking with a local mean floor
    k = 8
    local = np.convolve(flux, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    thr = local + 0.06
    onsets = []
    last = -1.0
    for i in range(1, n_frames - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            t = i * hop / sr
            if t - last > 0.03:            # de-dupe within 30 ms
                onsets.append(t)
                last = t
    return np.array(onsets)


def drop_unsupported_short(notes: list[NoteEvent], onsets: np.ndarray,
                           max_dur: float = 0.07, tol: float = 0.055) -> int:
    """Remove short notes that produce no sound: a brief detection with no audio
    attack near its onset is a visual artifact (overlay text/emoji, watermark,
    reflection) rather than a played note (§18, §36). Longer notes and anything
    with audio support are untouched. Returns the number removed. Notes are
    edited in place; caller re-ids."""
    if onsets.size == 0:
        return 0
    onsets = np.sort(onsets)
    keep = []
    removed = 0
    for n in notes:
        # a re-strike is a visually-confirmed re-attack (leading-edge notch); the
        # audio onset detector blurs rapid same-key repeats into one attack, so do
        # not require separate audio support for these or fast repeats vanish.
        if n.duration <= max_dur and "restrike" not in n.issues:
            i = int(np.searchsorted(onsets, n.start))
            near = min((abs(onsets[j] - n.start) for j in (i - 1, i)
                        if 0 <= j < onsets.size), default=1e9)
            if near > tol:
                removed += 1
                continue
        keep.append(n)
    notes[:] = keep
    for i, n in enumerate(notes):
        n.id = i + 1
    return removed


def snap_onsets(notes: list[NoteEvent], onsets: np.ndarray,
                tol: Optional[float] = None) -> int:
    """Snap each note's onset to the nearest audio attack within ``tol`` seconds,
    preserving duration. Returns how many notes were refined (§10, §18).

    The visual onset (key-light threshold crossing or bar edge) jitters by a
    couple of frames, so a fixed 60 ms window left most notes unsnapped and
    smeared chords - the main cause of the two hands sounding out of time. The
    audio attack is accurate, so widen the window to catch that jitter; it is
    derived from the song's own note spacing (clamped 60-110 ms) and kept below
    half the median inter-onset interval so a note never snaps to a neighbour."""
    if onsets.size == 0 or not notes:
        return 0
    onsets = np.sort(onsets)
    if tol is None:
        if onsets.size >= 3:
            median_ioi = float(np.median(np.diff(onsets)))
            tol = float(np.clip(0.45 * median_ioi, 0.06, 0.11))
        else:
            tol = 0.09
    refined = 0
    # A key cannot sound twice at the same instant, so at most one note of a given
    # pitch may snap to any one attack. The audio detector blurs a rapid same-key
    # repeat into a single attack; without this guard two visually-distinct
    # re-strikes both snap to it, collide, and one is later dropped - the repeat
    # is lost. Earliest note claims the attack; the rest keep their detected time.
    claimed: set = set()
    for n in sorted(notes, key=lambda x: x.start):
        idx = int(np.searchsorted(onsets, n.start))
        best, bestd = None, tol
        for j in (idx - 1, idx):
            if 0 <= j < onsets.size:
                d = abs(onsets[j] - n.start)
                if d < bestd:
                    bestd, best = d, float(onsets[j])
        if best is None:
            continue
        key = (n.midi, round(best, 4))
        if key in claimed:
            continue          # a same-pitch note already took this attack
        claimed.add(key)
        if abs(best - n.start) > 1e-4:
            dur = n.duration
            n.start = best
            n.end = best + dur
            n.timing_confidence = min(1.0, max(n.timing_confidence, 0.9))
            if n.verification.value == "unverified":
                from .model import VerificationStatus
                n.verification = VerificationStatus.AUDIO_CONFIRMED
            refined += 1
    notes.sort(key=lambda x: (x.start, x.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    return refined
