"use client";
import { useEffect, useRef, useCallback } from "react";
import { Note, Project } from "@/lib/api";
import { HAND_COLOR, HAND_COLOR_DIM, confidenceTone, isBlack } from "@/lib/render";

interface Props {
  project: Project;
  notes: Note[];
  time: number;
  pxPerSec: number;
  scrollX: number;              // seconds at left edge
  selectedId: number | null;
  uncertainOnly: boolean;
  showBeats: boolean;
  onSelect: (n: Note | null) => void;
  onSeek: (t: number) => void;
  onScroll: (s: number) => void;
  onZoom: (p: number) => void;
}

const GUTTER = 46;
const RULER = 22;

export default function PianoRoll(p: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; scroll: number } | null>(null);

  const { project, notes, selectedId, uncertainOnly } = p;
  const geom = project.geometry;
  const lo = geom.low_midi, hi = geom.high_midi;
  const span = hi - lo + 1;

  const draw = useCallback(() => {
    const cv = ref.current, wrap = box.current;
    if (!cv || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + "px"; cv.style.height = H + "px";
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const gridW = W - GUTTER;
    const gridH = H - RULER;
    const rowH = gridH / span;
    const midiToY = (m: number) => RULER + (hi - m) * rowH;
    const timeToX = (t: number) => GUTTER + (t - p.scrollX) * p.pxPerSec;
    const xToTime = (x: number) => (x - GUTTER) / p.pxPerSec + p.scrollX;

    // pitch lanes background (black-key rows tinted)
    for (let m = lo; m <= hi; m++) {
      const y = midiToY(m);
      ctx.fillStyle = isBlack(m) ? "#131519" : "#171a21";
      ctx.fillRect(GUTTER, y, gridW, rowH);
    }
    // octave lines
    ctx.strokeStyle = "#20242e";
    for (let m = lo; m <= hi; m++) {
      if (m % 12 === 0) {
        const y = midiToY(m) + rowH;
        ctx.beginPath(); ctx.moveTo(GUTTER, y); ctx.lineTo(W, y); ctx.stroke();
      }
    }

    // beat / measure grid
    if (p.showBeats) {
      const { beat_phase, beat_period, time_signature } = project.tempo;
      const tsNum = time_signature[0] || 4;
      const t0 = p.scrollX, t1 = p.scrollX + gridW / p.pxPerSec;
      let b = Math.floor((t0 - beat_phase) / beat_period);
      for (let t = beat_phase + b * beat_period; t < t1; t += beat_period, b++) {
        const x = timeToX(t);
        if (x < GUTTER) continue;
        const measure = ((b % tsNum) + tsNum) % tsNum === 0;
        ctx.strokeStyle = measure ? "#2c313d" : "#20242c";
        ctx.beginPath(); ctx.moveTo(x, RULER); ctx.lineTo(x, H); ctx.stroke();
      }
    }

    // notes
    for (const n of notes) {
      if (uncertainOnly && confidenceTone(n) === "high") continue;
      const x = timeToX(n.start);
      const w = Math.max(2, (n.end - n.start) * p.pxPerSec);
      if (x + w < GUTTER || x > W) continue;
      const y = midiToY(n.midi) + 0.5;
      const h = Math.max(2, rowH - 1);
      const tone = confidenceTone(n);
      const sel = n.id === selectedId;
      ctx.fillStyle = sel ? "#ffffff" : (rowH < 5 ? HAND_COLOR[n.hand] : HAND_COLOR_DIM[n.hand]);
      ctx.fillRect(Math.max(GUTTER, x), y, w, h);
      if (rowH >= 4) {
        ctx.fillStyle = HAND_COLOR[n.hand];
        ctx.fillRect(Math.max(GUTTER, x), y, w, Math.min(h, Math.max(2, h * 0.5)));
      }
      if (tone !== "high") {
        ctx.strokeStyle = tone === "low" ? "#e0533a" : "#e0a52a";
        ctx.lineWidth = 1;
        ctx.strokeRect(Math.max(GUTTER, x) + 0.5, y + 0.5, Math.max(1, w - 1), h - 1);
      }
      if (sel) {
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
        ctx.strokeRect(Math.max(GUTTER, x) - 0.5, y - 0.5, w + 1, h + 1);
      }
    }

    // gutter (pitch labels)
    ctx.fillStyle = "#12141a";
    ctx.fillRect(0, 0, GUTTER, H);
    ctx.strokeStyle = "#20242e"; ctx.beginPath();
    ctx.moveTo(GUTTER, 0); ctx.lineTo(GUTTER, H); ctx.stroke();
    ctx.font = "10px ui-monospace, monospace";
    for (let m = lo; m <= hi; m++) {
      if (m % 12 === 0) {
        const y = midiToY(m);
        ctx.fillStyle = "#6b7180";
        ctx.fillText(`C${m / 12 - 1}`, 6, y + rowH - 1);
      }
    }

    // ruler
    ctx.fillStyle = "#12141a";
    ctx.fillRect(GUTTER, 0, gridW, RULER);
    ctx.strokeStyle = "#20242e"; ctx.beginPath();
    ctx.moveTo(GUTTER, RULER); ctx.lineTo(W, RULER); ctx.stroke();
    ctx.fillStyle = "#6b7180"; ctx.font = "10px ui-monospace, monospace";
    const step = niceStep(p.pxPerSec);
    const t0 = Math.ceil(p.scrollX / step) * step;
    for (let t = t0; timeToX(t) < W; t += step) {
      const x = timeToX(t);
      ctx.strokeStyle = "#252a34";
      ctx.beginPath(); ctx.moveTo(x, RULER - 5); ctx.lineTo(x, RULER); ctx.stroke();
      ctx.fillText(t.toFixed(step < 1 ? 1 : 0) + "s", x + 3, 13);
    }

    // playhead
    const px = timeToX(p.time);
    if (px >= GUTTER) {
      ctx.strokeStyle = "#e8ecf5"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
      ctx.fillStyle = "#e8ecf5";
      ctx.beginPath(); ctx.moveTo(px - 4, 0); ctx.lineTo(px + 4, 0); ctx.lineTo(px, 5); ctx.fill();
    }
  }, [project, notes, selectedId, uncertainOnly, p.time, p.pxPerSec, p.scrollX, p.showBeats, lo, hi, span]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const r = () => draw();
    window.addEventListener("resize", r);
    return () => window.removeEventListener("resize", r);
  }, [draw]);

  function hit(e: React.MouseEvent) {
    const rect = ref.current!.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const gridH = rect.height - RULER;
    const rowH = gridH / span;
    if (y < RULER) { p.onSeek((x - GUTTER) / p.pxPerSec + p.scrollX); return; }
    const t = (x - GUTTER) / p.pxPerSec + p.scrollX;
    const midi = hi - Math.floor((y - RULER) / rowH);
    let best: Note | null = null;
    for (const n of notes) {
      if (n.midi === midi && t >= n.start - 0.02 && t <= n.end + 0.02) {
        best = n; break;
      }
    }
    p.onSelect(best);
  }

  return (
    <div ref={box} style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}>
      <canvas
        ref={ref}
        style={{ display: "block", cursor: drag.current ? "grabbing" : "crosshair" }}
        onMouseDown={(e) => {
          if (e.button === 1 || e.altKey) drag.current = { x: e.clientX, scroll: p.scrollX };
          else hit(e);
        }}
        onMouseMove={(e) => {
          if (drag.current) {
            const dx = e.clientX - drag.current.x;
            p.onScroll(Math.max(0, drag.current.scroll - dx / p.pxPerSec));
          }
        }}
        onMouseUp={() => (drag.current = null)}
        onMouseLeave={() => (drag.current = null)}
        onWheel={(e) => {
          if (e.ctrlKey || e.metaKey) {
            p.onZoom(Math.min(1200, Math.max(6, p.pxPerSec * (e.deltaY < 0 ? 1.15 : 0.87))));
          } else {
            p.onScroll(Math.max(0, p.scrollX + e.deltaX / p.pxPerSec));
          }
        }}
      />
    </div>
  );
}

function niceStep(pps: number): number {
  const target = 70 / pps;
  const steps = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60];
  return steps.find((s) => s >= target) || 120;
}
