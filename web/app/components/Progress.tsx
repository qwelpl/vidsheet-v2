"use client";
import { JobStatus } from "@/lib/api";

const STAGES = [
  ["prepare", "Preparing video"],
  ["keyboard", "Detecting keyboard"],
  ["theme", "Reading note theme"],
  ["track", "Tracking falling notes"],
  ["reconstruct", "Reconstructing events"],
  ["tempo", "Analysing tempo"],
  ["verify", "Verifying reconstruction"],
  ["done", "Complete"],
];

export default function Progress({ job, onCancel }: { job: JobStatus; onCancel: () => void }) {
  const current = job.stages[job.stages.length - 1]?.stage || "prepare";
  const reachedIdx = STAGES.findIndex((s) => s[0] === current);

  return (
    <div className="fade-in" style={{ height: "100vh", display: "grid", placeItems: "center" }}>
      <div style={{ width: 560, maxWidth: "92vw" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{job.title || "Analyzing"}</div>
          <button className="btn btn-ghost" onClick={onCancel}>Cancel</button>
        </div>

        <div className="panel" style={{ borderRadius: 10, padding: 18 }}>
          <div style={{ height: 6, background: "#0b0c10", borderRadius: 3, overflow: "hidden" }}>
            <div
              style={{
                height: "100%", width: `${Math.round(job.progress * 100)}%`,
                background: job.status === "error" ? "var(--danger)" : "var(--accent)",
                transition: "width .3s ease",
              }}
            />
          </div>
          <div className="mono" style={{ marginTop: 10, fontSize: 12, color: "var(--text-dim)" }}>
            {Math.round(job.progress * 100)}% · {job.message}
          </div>

          <div style={{ marginTop: 16, display: "grid", gap: 6 }}>
            {STAGES.map(([id, label], i) => {
              const done = i < reachedIdx || job.status === "done";
              const active = i === reachedIdx && job.status !== "done";
              return (
                <div key={id} style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 12 }}>
                  <span
                    style={{
                      width: 8, height: 8, borderRadius: 4,
                      background: done ? "var(--accent-2)" : active ? "var(--accent)" : "#2a2e3a",
                    }}
                  />
                  <span style={{ color: done ? "var(--text)" : active ? "var(--text)" : "var(--text-faint)" }}>
                    {label}
                  </span>
                </div>
              );
            })}
          </div>

          {job.status === "error" && (
            <div className="mono" style={{ marginTop: 14, color: "var(--danger)", fontSize: 12, lineHeight: 1.5 }}>
              {job.error}
            </div>
          )}
        </div>

        <div style={{ marginTop: 12, maxHeight: 120, overflow: "auto" }} className="mono">
          {job.stages.slice(-8).map((s, i) => (
            <div key={i} style={{ fontSize: 11, color: "var(--text-faint)", padding: "1px 0" }}>
              {s.stage} - {s.message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
