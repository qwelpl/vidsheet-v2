"use client";
import { useEffect, useRef, useCallback } from "react";
import { Project, Note } from "@/lib/api";
import { HAND_COLOR, noteBarAt, laneByMidi, isBlack } from "@/lib/render";

interface Props {
  project: Project;
  notes: Note[];
  time: number;
  transparent?: boolean;   // overlay mode — no background/keyboard
  opacity?: number;
  selectedId?: number | null;
}

// Renders the reconstruction in the ORIGINAL video's geometry coordinates so it
// lines up pixel-for-pixel when overlaid on the source (§29).
export default function ReconView({ project, notes, time, transparent, opacity = 1, selectedId }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const geom = project.geometry;
  const v = project.fall_speed;
  const lanes = laneByMidi(geom);

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
    const sx = W / geom.width, sy = H / geom.height;

    if (!transparent) {
      ctx.fillStyle = "#0a0b0e";
      ctx.fillRect(0, 0, W, H);
    }
    ctx.globalAlpha = opacity;

    // falling note bars
    for (const n of notes) {
      const span = noteBarAt(n, geom, v, time);
      if (!span) continue;
      const lane = lanes.get(n.midi);
      if (!lane) continue;
      const x0 = (lane.center - lane.half_width) * sx;
      const x1 = (lane.center + lane.half_width) * sx;
      const yt = span[0] * sy, yb = span[1] * sy;
      const c = HAND_COLOR[n.hand];
      if (!transparent) {
        ctx.fillStyle = c + "22";
        ctx.fillRect(x0 - 2, yt, x1 - x0 + 4, yb - yt);
      }
      ctx.fillStyle = n.id === selectedId ? "#ffffff" : c;
      ctx.fillRect(x0, yt, Math.max(1, x1 - x0), Math.max(1, yb - yt));
      if (transparent) {
        ctx.strokeStyle = c; ctx.lineWidth = 1;
        ctx.strokeRect(x0, yt, Math.max(1, x1 - x0), Math.max(1, yb - yt));
      }
    }

    // keyboard + strike line
    const strike = geom.strike_y * sy;
    if (!transparent) {
      ctx.fillStyle = "#f2f2f4";
      ctx.fillRect(0, strike, W, H - strike);
      for (const l of geom.lanes) {
        if (isBlack(l.midi)) {
          const x0 = (l.center - l.half_width) * sx;
          const x1 = (l.center + l.half_width) * sx;
          ctx.fillStyle = "#141414";
          ctx.fillRect(x0, strike, x1 - x0, (geom.keyboard_bottom - geom.keyboard_top) * sy * 0.62);
        }
      }
      // key-press highlight for sounding notes (§55)
      for (const n of notes) {
        if (n.start <= time && time < n.end) {
          const l = lanes.get(n.midi);
          if (!l) continue;
          ctx.fillStyle = HAND_COLOR[n.hand] + "cc";
          ctx.fillRect((l.center - l.half_width) * sx, strike + 1,
            l.half_width * 2 * sx, (H - strike) - 2);
        }
      }
    }
    ctx.strokeStyle = "#e0a52a"; ctx.lineWidth = 1;
    ctx.globalAlpha = opacity * 0.9;
    ctx.beginPath(); ctx.moveTo(0, strike); ctx.lineTo(W, strike); ctx.stroke();
    ctx.globalAlpha = 1;
  }, [project, notes, time, transparent, opacity, selectedId, geom, v]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const r = () => draw();
    window.addEventListener("resize", r);
    return () => window.removeEventListener("resize", r);
  }, [draw]);

  return (
    <div ref={box} style={{ position: "absolute", inset: 0 }}>
      <canvas ref={ref} style={{ display: "block" }} />
    </div>
  );
}
