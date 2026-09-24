"""Video I/O: metadata probing, streaming frame extraction, source download.

Frames are streamed from ffmpeg through a pipe so a two-hour 1080p60 video is
never fully resident in memory. Presentation timestamps are read from the
container rather than assuming perfectly constant frame spacing (§4).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .model import VideoMeta


def _require(bin_name: str) -> str:
    path = shutil.which(bin_name)
    if not path:
        raise RuntimeError(
            f"'{bin_name}' not found on PATH. Install ffmpeg (brew install ffmpeg)."
        )
    return path


def probe(path: str) -> VideoMeta:
    """Read width/height/fps/duration/frame count using ffprobe."""
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration,filename",
        "-of", "json", path,
    ]
    out = subprocess.check_output(cmd)
    info = json.loads(out)
    stream = info["streams"][0]
    fmt = info.get("format", {})

    def parse_rate(r: str) -> float:
        if not r or r == "0/0":
            return 0.0
        num, _, den = r.partition("/")
        den = float(den or 1)
        return float(num) / den if den else 0.0

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    if fps <= 0:
        fps = 30.0
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    nb = stream.get("nb_frames")
    frame_count = int(nb) if nb and nb.isdigit() else int(round(duration * fps))
    variable = abs(parse_rate(stream.get("r_frame_rate")) - fps) > 0.5

    return VideoMeta(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(fps),
        duration=float(duration),
        frame_count=int(frame_count),
        variable_fps=variable,
    )


@dataclass
class Frame:
    index: int
    time: float          # presentation timestamp in seconds
    image: np.ndarray    # H x W x 3, BGR uint8


def stream_frames(
    meta: VideoMeta,
    start: float = 0.0,
    end: Optional[float] = None,
    scale_width: Optional[int] = None,
    step: int = 1,
) -> Iterator[Frame]:
    """Yield frames as BGR uint8 arrays.

    ``scale_width`` optionally downscales (analysis stays native by default,
    §4). ``step`` skips frames for fast preview passes. Timestamps are derived
    from the requested start plus frame index / fps; for VFR sources callers
    should prefer :func:`stream_frames_with_pts`.
    """
    ffmpeg = _require("ffmpeg")
    w = scale_width or meta.width
    if scale_width:
        h = int(round(meta.height * scale_width / meta.width))
        h -= h % 2
    else:
        h = meta.height

    vf = []
    if scale_width:
        vf.append(f"scale={w}:{h}")
    if step > 1:
        vf.append(f"select=not(mod(n\\,{step}))")

    cmd = [ffmpeg, "-nostdin", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", meta.path]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - start):.6f}"]
    if vf:
        cmd += ["-vf", ",".join(vf), "-vsync", "0"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            t = start + (idx * step) / meta.fps
            yield Frame(index=idx * step, time=t, image=img)
            idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def extract_audio(meta: VideoMeta, out_path: str, sr: int = 22050) -> Optional[str]:
    """Extract mono PCM wav for secondary audio verification. Returns path or
    None if the source has no audio track."""
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg, "-nostdin", "-loglevel", "error", "-i", meta.path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "-y", out_path,
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        return None
    return out_path


def download(url: str, out_dir: str) -> tuple[str, str]:
    """Download best-quality source with yt-dlp. Prioritises resolution over
    bandwidth and avoids recompression (§3). Returns (path, title)."""
    ytdlp = _require("yt-dlp")
    import os
    tmpl = os.path.join(out_dir, "%(id)s.%(ext)s")
    # bv*+ba => best video + best audio, prefer height then fps, no re-encode.
    cmd = [
        ytdlp, "-f", "bv*[height<=2160]+ba/b", "--merge-output-format", "mp4",
        "-S", "res,fps,vcodec", "--no-playlist", "-o", tmpl,
        "--print", "after_move:filepath", "--print", "after_move:title",
        "--no-simulate", url,
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    path = out[0]
    title = out[1] if len(out) > 1 else ""
    return path, title
