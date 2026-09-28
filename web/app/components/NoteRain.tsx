"use client";
import { useEffect, useRef } from "react";

// Ambient background: colored piano-roll bars falling like a live Synthesia
// scroll, plus a soft strike-line glow at the bottom. Purely decorative.
interface Bar {
  x: number;      // lane centre (fraction of width)
  w: number;      // width px
  y: number;      // top y px
  h: number;      // height px
  vy: number;     // px/sec
  color: string;
  alpha: number;
}

const BLUE = "76, 139, 245";   // --accent
const GREEN = "56, 193, 114";  // --accent-2

export default function NoteRain() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const LANES = 26;
    const bars: Bar[] = [];
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function resize() {
      W = canvas!.clientWidth;
      H = canvas!.clientHeight;
      canvas!.width = Math.floor(W * dpr);
      canvas!.height = Math.floor(H * dpr);
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    function spawn() {
      const lane = Math.floor(Math.random() * LANES);
      const laneW = W / LANES;
      const w = laneW * (0.34 + Math.random() * 0.16);
      const h = 40 + Math.random() * 240;
      bars.push({
        x: (lane + 0.5) * laneW,
        w,
        y: -h,
        h,
        vy: 46 + Math.random() * 70,
        color: Math.random() < 0.5 ? BLUE : GREEN,
        alpha: 0.10 + Math.random() * 0.22,
      });
    }
    // seed a screenful
    for (let i = 0; i < 34; i++) {
      spawn();
      bars[bars.length - 1].y = Math.random() * H;
    }

    function roundRect(x: number, y: number, w: number, h: number, r: number) {
      r = Math.min(r, w / 2, h / 2);
      ctx!.beginPath();
      ctx!.moveTo(x + r, y);
      ctx!.arcTo(x + w, y, x + w, y + h, r);
      ctx!.arcTo(x + w, y + h, x, y + h, r);
      ctx!.arcTo(x, y + h, x, y, r);
      ctx!.arcTo(x, y, x + w, y, r);
      ctx!.closePath();
    }

    function frame(now: number) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      ctx!.clearRect(0, 0, W, H);

      // strike-line glow near the bottom
      const gy = H - 90;
      const grad = ctx!.createLinearGradient(0, gy - 60, 0, H);
      grad.addColorStop(0, "rgba(76,139,245,0)");
      grad.addColorStop(1, "rgba(76,139,245,0.05)");
      ctx!.fillStyle = grad;
      ctx!.fillRect(0, gy - 60, W, H - (gy - 60));

      if (!reduce) {
        acc += dt;
        while (acc > 0.16) { spawn(); acc -= 0.16; }
      }

      for (let i = bars.length - 1; i >= 0; i--) {
        const b = bars[i];
        if (!reduce) b.y += b.vy * dt;
        // fade as it nears / passes the strike line
        const dist = gy - (b.y + b.h);
        let a = b.alpha;
        if (dist < 120) a = b.alpha * Math.max(0, dist / 120) + 0.02;
        ctx!.fillStyle = `rgba(${b.color}, ${a})`;
        roundRect(b.x - b.w / 2, b.y, b.w, b.h, 5);
        ctx!.fill();
        if (b.y > H + 40) bars.splice(i, 1);
      }
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      aria-hidden
      style={{
        position: "fixed", inset: 0, width: "100%", height: "100%",
        zIndex: 0, pointerEvents: "none",
        maskImage: "radial-gradient(120% 90% at 50% 30%, #000 40%, transparent 100%)",
        WebkitMaskImage: "radial-gradient(120% 90% at 50% 30%, #000 40%, transparent 100%)",
      }}
    />
  );
}
