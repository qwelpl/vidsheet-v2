# Reprise - Synthesia Reconstruction Engine

Reprise is a forensic reverse-compiler for Synthesia-style piano-roll videos. It
treats the **video as ground truth** and recovers the underlying performance - 
every note, exact pitch/octave, sub-frame onset and release, chords, repeats,
hands, dynamics, tempo - using deterministic computer vision. Every reconstructed
note carries its own evidence and confidence; nothing is silently guessed.

The pipeline is a chain of measurable, testable stages:

```
video frames → keyboard geometry → pitch-lane mapping → per-lane note detection
→ temporal occupancy events → sub-frame strike/release timing → hands / tempo
→ MIDI / MusicXML reconstruction → render-and-compare self-verification
```

On a clean synthetic render with known ground truth the reconstruction is
note-for-note exact (F1 = 1.0, onset error ≈ 7 ms at 60 fps, duration error ≈ 1 ms).

## Architecture

- **`engine/`** - Python analysis engine (OpenCV) + FastAPI job server.
  - `synthesia/keyboard.py` - keyboard detection & exact pitch-lane geometry from
    the black-key 2/3 groups (§6, §7), never an even 88-way split.
  - `synthesia/theme.py` - colour-theme discovery; separates solid note cores from
    translucent glow by value (§8, §34).
  - `synthesia/detect.py` - per-lane column sampling with a centre-coverage gate
    that rejects a neighbour's bleed (§7, §15).
  - `synthesia/events.py` - strike-line occupancy events + sub-frame timing via
    approach/trailing-edge line fits (§9–12, §53).
  - `synthesia/hands.py`, `tempo.py` - hand assignment, tempo & quantization.
  - `synthesia/renderer.py`, `verify.py` - render the reconstruction back and
    compare against the source; metrics and automatic error search (§24, §25, §48).
  - `synthesia/synth.py` - synthetic ground-truth video generator (§65).
  - `synthesia/exporters.py` - MIDI (high PPQ), MusicXML, CSV, JSON (§22, §58).
- **`web/`** - Next.js workspace UI: original vs reconstruction comparison, canvas
  piano roll, note inspector with corrections, confidence colouring, frame
  inspector, transport, exports.

## Running

Requires `ffmpeg`, `ffprobe`, `yt-dlp` on PATH, Python 3.12+, Node 20+.

```bash
# 1. engine + API
cd engine
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m uvicorn server:app --port 8000

# 2. web UI (separate shell)
cd web
npm install
npm run dev            # http://localhost:3000  (or next free port)
```

Open the UI, paste a YouTube URL / upload a video / click **Run demo**, then
**Analyze**.

### Deploying

The web UI deploys to Vercel from `web/`. The engine cannot run on Vercel: jobs
live in process memory, run in background threads and need `ffmpeg`/`yt-dlp`.
Host it as a single long-running container instead (Render, Fly.io, Railway, a VPS):

```bash
cd engine
docker build -t reprise-engine .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://<your-app>.vercel.app reprise-engine
```

Keep it to one instance/worker. `render.yaml` deploys it to Render as
`reprise-engine.onrender.com`, which production web builds use by default.
To use another host, set `NEXT_PUBLIC_API=https://<engine-host>` in the Vercel
project and redeploy (the value is baked in at build time).

### CLI (no server)

```bash
cd engine
./.venv/bin/python cli.py demo --preset maximum --out out      # synthetic ground-truth run
./.venv/bin/python cli.py analyze path/to/video.mp4 --preset maximum --out out
```

## Tests

```bash
cd engine && ./.venv/bin/python -m pytest -q
```

The suite generates synthetic clips from known MIDI and asserts pitch precision/
recall, onset/duration error, repeated-note separation, and geometry detection - 
the regression guard against silent accuracy drops (§65, §66).

## Accuracy honesty

Reprise never claims literal 100% accuracy unless compared against known
ground-truth note data. It reports measured statistics - visual agreement,
per-field confidence, notes needing review - and flags every uncertain event for
human correction rather than hiding it.
