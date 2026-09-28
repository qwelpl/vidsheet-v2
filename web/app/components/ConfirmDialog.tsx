"use client";
import { useEffect, useState } from "react";

// Shown every time a user queues a piece: they must reaffirm they have the
// rights before the job starts.
export default function ConfirmDialog({
  open, onConfirm, onCancel,
}: {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (open) setChecked(false);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel(); }}
      style={{ position: "fixed", inset: 0, zIndex: 100, display: "grid", placeItems: "center",
        background: "rgba(0,0,0,.6)", backdropFilter: "blur(3px)", padding: 20 }}
    >
      <div className="panel vs-panel" style={{ width: 460, maxWidth: "100%", borderRadius: 12, padding: 22 }}>
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>Before we process this</div>
        <div style={{ fontSize: 12.5, color: "var(--text-dim)", lineHeight: 1.55, marginBottom: 14 }}>
          You are responsible for the content you submit. Do not upload copyrighted material
          you are not authorized to use.
        </div>

        <label style={{ display: "flex", gap: 10, alignItems: "flex-start",
          fontSize: 12.5, color: "var(--text)", cursor: "pointer", lineHeight: 1.5,
          background: "var(--panel-2)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: "11px 12px" }}>
          <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)}
            style={{ marginTop: 2, accentColor: "var(--accent)", width: 16, height: 16, flexShrink: 0 }} />
          <span>
            I confirm that I own this content, it is in the public domain, or I have all
            necessary rights and permissions to upload, process, and create sheet music from it.
          </span>
        </label>

        <div style={{ fontSize: 11, color: "var(--text-faint)", lineHeight: 1.55, margin: "12px 2px 0" }}>
          Compliance with copyright is your responsibility. By continuing you agree to the{" "}
          <a href="/terms" target="_blank" rel="noopener noreferrer"
            style={{ color: "var(--accent)" }}>Terms of Service</a>.
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 18 }}>
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn btn-primary" disabled={!checked}
            onClick={() => { if (checked) onConfirm(); }}>
            Agree &amp; start
          </button>
        </div>
      </div>
    </div>
  );
}
