"use client";
import { useRef, useState } from "react";
import { startYoutube, startUpload, startDemo, JobStatus } from "@/lib/api";
import NoteRain from "./NoteRain";
import ConfirmDialog from "./ConfirmDialog";

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
  // the queue action waiting on the rights-confirmation dialog
  const [pending, setPending] = useState<(() => Promise<JobStatus>) | null>(null);

  // Ask for rights confirmation before every queue. Store the action; the dialog
  // runs it on agreement.
  function requestQueue(fn: () => Promise<JobStatus>) {
    setPending(() => fn);
  }

  async function runPending() {
    const fn = pending;
    setPending(null);
    if (!fn) return;
    await run(fn);
  }

  // Run a queue action directly (no rights dialog). Used by the demo, whose
  // content we own.
  async function run(fn: () => Promise<JobStatus>) {
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
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", position: "relative", overflow: "hidden" }}>
      <NoteRain />
      <div className="vs-rise" style={{ width: 620, maxWidth: "92vw", position: "relative", zIndex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11, marginBottom: 10 }}>
          <h1 style={{ fontSize: 34, fontWeight: 800, letterSpacing: "-0.03em", margin: 0, color: "var(--text)" }}>
            Vidsheet
          </h1>
          <span style={{ color: "var(--text-faint)", fontSize: 12, alignSelf: "flex-end", paddingBottom: 6 }}>
            piano video to sheet music
          </span>
        </div>

        <div className="panel vs-panel" style={{ borderRadius: 12, padding: 20 }}>
          <div style={{ marginBottom: 12, fontSize: 12, color: "var(--text-dim)",
            background: "var(--panel-2)", border: "1px solid var(--border-soft)",
            borderRadius: 7, padding: "8px 11px" }}>
            Only <strong style={{ color: "var(--text)" }}>Synthesia-style</strong> piano-roll videos
            (falling colored note bars over a keyboard) are supported. Hands are split by bar color,
            so the left and right hands must use <strong style={{ color: "var(--text)" }}>different
            colors</strong>. If both hands share one color, every note lands on a single staff.
          </div>
          <div className="label" style={{ marginBottom: 8 }}>YouTube or direct video URL</div>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              className="input mono"
              placeholder="https://www.youtube.com/watch?v=…"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && url && !busy && requestQueue(() => startYoutube(url, preset))}
            />
            <button
              className="btn btn-primary"
              disabled={!url || busy}
              onClick={() => requestQueue(() => startYoutube(url, preset))}
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
            <button className="btn" disabled={busy} onClick={() => run(() => startDemo("maximum"))}>
              Run demo
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) requestQueue(() => startUpload(f, preset));
                e.target.value = "";
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

        <div style={{ textAlign: "center", marginTop: 14, fontSize: 11, color: "var(--text-faint)" }}>
          <a href="/terms" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
            Terms of Service
          </a>
        </div>
      </div>

      <ConfirmDialog open={pending !== null} onConfirm={runPending} onCancel={() => setPending(null)} />
    </div>
  );
}
