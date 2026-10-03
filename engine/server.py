"""FastAPI backend for the reconstruction workspace.

Jobs run in background threads and stream real progress stages (§42). The UI
polls job status, then loads the full reconstruction (notes + geometry + theme +
report) and the source video for synchronised playback and comparison.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from synthesia import analyze, Options
from synthesia import videoio
from synthesia.exporters import (write_midi, write_csv, write_musicxml,
                                 write_json)

app = FastAPI(title="Synthesia Reconstruction Engine")
# Comma-separated list of allowed web origins, e.g. the Vercel domain.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)

WORK = os.path.join(tempfile.gettempdir(), "synthesia_jobs")
os.makedirs(WORK, exist_ok=True)


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued|running|done|error
    stages: list[dict] = field(default_factory=list)
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[dict] = None
    video_path: Optional[str] = None
    preset: str = "balanced"
    title: str = ""
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id, "status": self.status, "progress": self.progress,
            "message": self.message, "stages": self.stages[-30:],
            "error": self.error, "title": self.title,
            "has_result": self.result is not None,
        }


JOBS: dict[str, Job] = {}


def _job_dir(job_id: str) -> str:
    d = os.path.join(WORK, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _run_job(job: Job, source: str, is_url: bool):
    try:
        job.status = "running"
        d = _job_dir(job.id)
        if is_url:
            job.message = "Downloading source"
            path, title = videoio.download(source, d)
            job.video_path = path
            job.title = title or "YouTube source"
        else:
            job.video_path = source
            job.title = job.title or os.path.basename(source)

        def progress(ev: dict):
            job.stages.append(ev)
            job.progress = ev.get("progress", job.progress)
            job.message = ev.get("message", job.message)

        opts = Options.for_preset(job.preset)
        result = analyze(job.video_path, opts, progress)
        # persist exports
        write_midi(os.path.join(d, "reconstruction.mid"), result.notes, result.tempo)
        write_csv(os.path.join(d, "notes.csv"), result.notes)
        write_musicxml(os.path.join(d, "score.musicxml"), result.notes, result.tempo)
        payload = result.to_dict()
        write_json(os.path.join(d, "project.json"), payload)
        job.result = payload
        job.status = "done"
        job.progress = 1.0
        job.message = "Complete"
    except Exception as e:  # surface real failure reasons (§69)
        job.status = "error"
        job.error = f"{e}"
        job.stages.append({"stage": "error", "message": str(e), "progress": job.progress})
        traceback.print_exc()


def _start(job: Job, source: str, is_url: bool):
    JOBS[job.id] = job
    t = threading.Thread(target=_run_job, args=(job, source, is_url), daemon=True)
    t.start()


@app.post("/api/analyze/youtube")
def analyze_youtube(url: str = Form(...), preset: str = Form("balanced")):
    job = Job(id=uuid.uuid4().hex[:12], preset=preset)
    _start(job, url, is_url=True)
    return job.public()


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...), preset: str = Form("balanced")):
    job = Job(id=uuid.uuid4().hex[:12], preset=preset, title=file.filename or "upload")
    d = _job_dir(job.id)
    dest = os.path.join(d, "source" + os.path.splitext(file.filename or ".mp4")[1])
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _start(job, dest, is_url=False)
    return job.public()


@app.post("/api/demo")
def analyze_demo(preset: str = Form("maximum")):
    from synthesia.synth import generate_video, demo_piece, SynthConfig
    job = Job(id=uuid.uuid4().hex[:12], preset=preset, title="Synthetic demo (known ground truth)")
    d = _job_dir(job.id)
    video = os.path.join(d, "demo.mp4")
    truth = generate_video(video, demo_piece(), SynthConfig())

    def run():
        try:
            job.status = "running"

            def progress(ev):
                job.stages.append(ev); job.progress = ev.get("progress", job.progress)
                job.message = ev.get("message", job.message)
            job.video_path = video
            result = analyze(video, Options.for_preset(job.preset), progress, truth=truth)
            write_midi(os.path.join(d, "reconstruction.mid"), result.notes, result.tempo)
            write_csv(os.path.join(d, "notes.csv"), result.notes)
            write_musicxml(os.path.join(d, "score.musicxml"), result.notes, result.tempo)
            payload = result.to_dict()
            write_json(os.path.join(d, "project.json"), payload)
            job.result = payload
            job.status = "done"; job.progress = 1.0; job.message = "Complete"
        except Exception as e:
            job.status = "error"; job.error = str(e); traceback.print_exc()

    JOBS[job.id] = job
    threading.Thread(target=run, daemon=True).start()
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.public()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.result is None:
        raise HTTPException(404, "result not ready")
    return JSONResponse(job.result)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.video_path or not os.path.exists(job.video_path):
        raise HTTPException(404, "video not found")
    return FileResponse(job.video_path)


_MSCORE = shutil.which("mscore") or shutil.which("musescore") or shutil.which("MuseScore")


def _render_pdf(d: str) -> str:
    """Engrave score.musicxml to a sheet-music PDF via MuseScore. Rendered fresh
    so it reflects the latest (possibly UI-edited) score."""
    src = os.path.join(d, "score.musicxml")
    if not os.path.exists(src):
        raise HTTPException(404, "score not available")
    if not _MSCORE:
        raise HTTPException(501, "PDF export needs MuseScore (mscore) on PATH")
    pdf = os.path.join(d, "score.pdf")
    try:
        r = subprocess.run([_MSCORE, src, "-o", pdf], capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "PDF render timed out")
    if not os.path.exists(pdf):
        raise HTTPException(500, f"PDF render failed: {r.stderr[-300:]}")
    return pdf


@app.get("/api/jobs/{job_id}/export/{fmt}")
def job_export(job_id: str, fmt: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    d = _job_dir(job_id)
    if fmt == "pdf":
        return FileResponse(_render_pdf(d), filename="sheet-music.pdf",
                            media_type="application/pdf")
    files = {"midi": "reconstruction.mid", "csv": "notes.csv",
             "musicxml": "score.musicxml", "json": "project.json"}
    fname = files.get(fmt)
    if not fname or not os.path.exists(os.path.join(d, fname)):
        raise HTTPException(404, "export not available")
    return FileResponse(os.path.join(d, fname), filename=fname)


@app.post("/api/jobs/{job_id}/notes")
async def update_notes(job_id: str, request: Request):
    """Persist UI-edited notes and regenerate the export files so a subsequent
    download reflects the corrections (pitch, timing, velocity, hand, deletes)."""
    job = JOBS.get(job_id)
    if not job or job.result is None:
        raise HTTPException(404, "job not found")
    body = await request.json()
    raw = body.get("notes", body if isinstance(body, list) else None)
    if not isinstance(raw, list):
        raise HTTPException(400, "expected a notes array")
    from synthesia.model import NoteEvent
    from synthesia.tempo import TempoAnalysis
    notes = [NoteEvent.from_dict(n) for n in raw]
    notes.sort(key=lambda n: (n.start, n.midi))
    for i, n in enumerate(notes):
        n.id = i + 1
    t = job.result.get("tempo", {}) or {}
    tempo = TempoAnalysis(
        bpm=float(t.get("bpm", 120.0)),
        beat_period=float(t.get("beat_period", 0.5)),
        beat_phase=float(t.get("beat_phase", 0.0)),
        time_signature=tuple(t.get("time_signature", [4, 4])),
        segments=[], confidence=float(t.get("confidence", 0.5)),
    )
    d = _job_dir(job_id)
    write_midi(os.path.join(d, "reconstruction.mid"), notes, tempo)
    write_csv(os.path.join(d, "notes.csv"), notes)
    write_musicxml(os.path.join(d, "score.musicxml"), notes, tempo)
    job.result["notes"] = [n.to_dict() for n in notes]
    write_json(os.path.join(d, "project.json"), job.result)
    return {"ok": True, "notes": len(notes)}


# Accept HEAD as well as GET: uptime monitors (UptimeRobot et al.) probe with
# HEAD by default, and a GET-only route answers 405 - which reads as "down" even
# though the service is up. Both verbs keep the Render instance from idling.
@app.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "jobs": len(JOBS)}
