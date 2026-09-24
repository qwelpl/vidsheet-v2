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

    // Letterbox to the source aspect ratio exactly like the <video> element's
    // object-fit: contain, so a single uniform scale keeps the reconstruction
    // undistorted and pixel-aligned for overlay/difference (§29).
    const s = Math.min(W / geom.width, H / geom.height);
    const ox = (W - geom.width * s) / 2;
    const oy = (H - geom.height * s) / 2;
    const X = (px: number) => ox + px * s;
    const Y = (py: number) => oy + py * s;

    if (!transparent) {
      ctx.fillStyle = "#0a0b0e";
      ctx.fillRect(0, 0, W, H);
    }
    ctx.globalAlpha = opacity;

    const strike = Y(geom.strike_y);
    const kbBottom = Y(geom.keyboard_bottom);
    const kbH = kbBottom - strike;

    // falling note bars (clipped to the roll area above the strike line)
    for (const n of notes) {
      const span = noteBarAt(n, geom, v, time);
      if (!span) continue;
      const lane = lanes.get(n.midi);
      if (!lane) continue;
      const x0 = X(lane.center - lane.half_width);
      const x1 = X(lane.center + lane.half_width);
      const yt = Y(span[0]), yb = Y(span[1]);
      const c = HAND_COLOR[n.hand];
      const w = Math.max(1, x1 - x0), h = Math.max(1, yb - yt);
      if (!transparent) {
        ctx.fillStyle = c + "20";
        ctx.fillRect(x0 - 2, yt, w + 4, h);
      }
      ctx.fillStyle = n.id === selectedId ? "#ffffff" : c;
      roundRect(ctx, x0, yt, w, h, Math.min(3, w / 3));
      ctx.fill();
      // brighter leading edge
      ctx.fillStyle = "#ffffff40";
      ctx.fillRect(x0, yb - Math.min(3, h), w, Math.min(3, h));
      if (transparent) {
        ctx.strokeStyle = n.id === selectedId ? "#fff" : c;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x0 + 0.5, yt + 0.5, w - 1, h - 1);
      }
    }

    if (!transparent) {
      // keyboard bed
      ctx.fillStyle = "#eceef2";
      ctx.fillRect(X(0), strike, geom.width * s, kbH);
      // white-key separators
      ctx.strokeStyle = "#c3c7d0"; ctx.lineWidth = 1;
      for (const l of geom.lanes) {
        if (!isBlack(l.midi)) {
          const xl = X(l.center - l.half_width);
          ctx.beginPath(); ctx.moveTo(xl, strike); ctx.lineTo(xl, kbBottom); ctx.stroke();
        }
      }
      // sounding white keys highlighted (§55)
      for (const n of notes) {
        if (n.start <= time && time < n.end && !isBlack(n.midi)) {
          const l = lanes.get(n.midi); if (!l) continue;
          ctx.fillStyle = HAND_COLOR[n.hand] + "55";
          ctx.fillRect(X(l.center - l.half_width), strike + 1, l.half_width * 2 * s, kbH - 1);
        }
      }
      // black keys on top (shorter), rounded bottoms
      const blackH = kbH * 0.62;
      for (const l of geom.lanes) {
        if (!isBlack(l.midi)) continue;
        const x0 = X(l.center - l.half_width), w = l.half_width * 2 * s;
        const sounding = notes.some((n) => n.midi === l.midi && n.start <= time && time < n.end);
        ctx.fillStyle = sounding ? HAND_COLOR[notes.find((n) => n.midi === l.midi)!.hand] : "#161820";
        roundRect(ctx, x0, strike, w, blackH, Math.min(2.5, w / 3), true);
        ctx.fill();
      }
      // top rim of keyboard
      ctx.fillStyle = "#d0d3da";
      ctx.fillRect(X(0), strike, geom.width * s, Math.max(1, kbH * 0.03));
    }

    // strike line
    ctx.strokeStyle = "#e0a52a"; ctx.lineWidth = 1.5;
    ctx.globalAlpha = opacity * 0.9;
    ctx.beginPath(); ctx.moveTo(X(0), strike); ctx.lineTo(X(geom.width), strike); ctx.stroke();
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

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number,
  w: number, h: number, r: number, bottomOnly = false) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  if (bottomOnly) {
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.closePath();
  } else {
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
}
