"use client";
import { useRef, useState } from "react";
import { startYoutube, startUpload, startDemo, JobStatus } from "@/lib/api";

const PRESETS = [
  { id: "fast", label: "Fast", desc: "Downscaled, quick preview" },
  { id: "balanced", label: "Balanced", desc: "1280px, full frames" },
  { id: "maximum", label: "Maximum Accuracy", desc: "Native res, every frame, all passes" },
];

export default function Landing({ onJob }: { onJob: (j: JobStatus) => void }) {
  const [url, setUrl] = useState("");
  const [preset, setPreset] = useState("balanced");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function guard(fn: () => Promise<JobStatus>) {
    setBusy(true);
    setErr(null);
    try {
      onJob(await fn());
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  return (
    <div className="fade-in" style={{ height: "100vh", display: "grid", placeItems: "center" }}>
      <div style={{ width: 620, maxWidth: "92vw" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 4 }}>
          <h1 style={{ fontSize: 22, fontWeight: 600, letterSpacing: "-0.02em", margin: 0 }}>
            Reprise
          </h1>
          <span style={{ color: "var(--text-faint)", fontSize: 12 }}>
            Synthesia reconstruction engine
          </span>
        </div>
        <p style={{ color: "var(--text-dim)", marginTop: 0, marginBottom: 22, fontSize: 13, lineHeight: 1.5 }}>
          Reverse-compile a piano-roll video into an exact performance. Visual note
          detection is the source of truth; every reconstructed note carries its own
          evidence and confidence.
        </p>

        <div className="panel" style={{ borderRadius: 10, padding: 18 }}>
          <div className="label" style={{ marginBottom: 8 }}>YouTube or direct video URL</div>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              className="input mono"
              placeholder="https://www.youtube.com/watch?v=…"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && url && guard(() => startYoutube(url, preset))}
            />
            <button
              className="btn btn-primary"
              disabled={!url || busy}
              onClick={() => guard(() => startYoutube(url, preset))}
            >
              Analyze
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "16px 0" }}>
            <div className="divider" style={{ flex: 1 }} />
            <span style={{ color: "var(--text-faint)", fontSize: 11 }}>or</span>
            <div className="divider" style={{ flex: 1 }} />
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" disabled={busy} onClick={() => fileRef.current?.click()} style={{ flex: 1, justifyContent: "center" }}>
              Upload video (MP4 / MOV / WebM / MKV)
            </button>
            <button className="btn" disabled={busy} onClick={() => guard(() => startDemo("maximum"))}>
              Run demo
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) guard(() => startUpload(f, preset));
              }}
            />
          </div>

          <div className="label" style={{ margin: "18px 0 8px" }}>Analysis quality</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => setPreset(p.id)}
                className="panel-2"
                style={{
                  textAlign: "left", padding: "9px 11px", borderRadius: 7, cursor: "pointer",
                  border: `1px solid ${preset === p.id ? "var(--accent)" : "var(--border)"}`,
                  color: "var(--text)",
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 500 }}>{p.label}</div>
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 2 }}>{p.desc}</div>
              </button>
            ))}
          </div>

          {busy && (
            <div style={{ marginTop: 14, color: "var(--text-dim)", fontSize: 12 }}>
              Starting job…
            </div>
          )}
          {err && (
            <div style={{ marginTop: 14, color: "var(--danger)", fontSize: 12 }} className="mono">
              {err}
            </div>
          )}
        </div>

        <div style={{ marginTop: 14, color: "var(--text-faint)", fontSize: 11, lineHeight: 1.6 }}>
          The demo renders a synthetic clip from a known score, then reconstructs it —
          so its accuracy is verified against ground truth, not estimated.
        </div>
      </div>
    </div>
  );
}
