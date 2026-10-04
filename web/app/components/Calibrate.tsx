"use client";
import { useEffect, useState } from "react";
import {
  JobStatus, PrepGeometry, ManualGeometry,
  getJob, frameUrl, analyzePrepared,
} from "@/lib/api";

const WHITE_KEYS = [
  "A0", "B0",
  "C1", "D1", "E1", "F1", "G1", "A1", "B1",
  "C2", "D2", "E2", "F2", "G2", "A2", "B2",
  "C3", "D3", "E3", "F3", "G3", "A3", "B3",
  "C4", "D4", "E4", "F4", "G4", "A4", "B4",
  "C5", "D5", "E5", "F5", "G5", "A5", "B5",
  "C6", "D6", "E6", "F6", "G6", "A6", "B6",
  "C7", "D7", "E7", "F7", "G7", "A7", "B7",
  "C8",
];

const DISP_W = 680;

export default function Calibrate({
  prep, onJob, onCancel,
}: {
  prep: JobStatus;
  onJob: (j: JobStatus) => void;
  onCancel: () => void;
}) {
  const [job, setJob] = useState<JobStatus>(prep);
  const [geo, setGeo] = useState<PrepGeometry | null>(prep.geometry ?? null);
  const [detector, setDetector] = useState<"auto" | "bars" | "keylight">("auto");
  const [override, setOverride] = useState(false);
  const [left, setLeft] = useState(0);
  const [right, setRight] = useState(0);
  const [strike, setStrike] = useState(0);
  const [bottom, setBottom] = useState(0);
  const [firstKey, setFirstKey] = useState("A0");
  const [lastKey, setLastKey] = useState("C8");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // poll the prepare job until the frame + geometry guess are ready
  useEffect(() => {
    if (job.status === "done" || job.status === "error") return;
    const t = setInterval(async () => {
      try {
        const s = await getJob(job.id);
        setJob(s);
        if (s.status === "done" && s.geometry) setGeo(s.geometry);
      } catch { /* transient */ }
    }, 700);
    return () => clearInterval(t);
  }, [job.id, job.status]);

  // seed the handles from the auto guess once it arrives
  useEffect(() => {
    if (!geo) return;
    setLeft(geo.left_edge);
    setRight(geo.right_edge);
    setStrike(Math.round(geo.strike_y));
    setBottom(Math.round(geo.keyboard_bottom));
    if (WHITE_KEYS.includes(geo.low_name)) setFirstKey(geo.low_name);
    if (WHITE_KEYS.includes(geo.high_name)) setLastKey(geo.high_name);
  }, [geo]);

  async function go() {
    setBusy(true);
    setErr(null);
    try {
      const manual: ManualGeometry | null = override && geo ? {
        frame_width: geo.frame_width, frame_height: geo.frame_height,
        left, right, strike_y: strike, bottom,
        first_key: firstKey, last_key: lastKey,
      } : null;
      onJob(await analyzePrepared(job.id, detector, manual));
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  const scale = geo ? DISP_W / geo.frame_width : 1;
  const dispH = geo ? Math.round(geo.frame_height * scale) : 360;

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      <div className="vs-rise" style={{ width: DISP_W + 40, maxWidth: "95vw" }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 4px", color: "var(--text)" }}>
          Calibrate
        </h2>
        <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 12 }}>
          Check the detected keyboard, pick a detector, and override the keyboard if it’s wrong.
        </div>

        <div className="panel" style={{ borderRadius: 12, padding: 16 }}>
          {!geo ? (
            <div style={{ height: dispH, display: "grid", placeItems: "center", color: "var(--text-dim)" }}>
              {job.status === "error" ? (
                <span className="mono" style={{ color: "var(--danger)" }}>{job.error}</span>
              ) : "Fetching a frame…"}
            </div>
          ) : (
            <>
              <div style={{ position: "relative", width: DISP_W, height: dispH, margin: "0 auto",
                borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={frameUrl(job.id)} alt="calibration frame"
                  style={{ width: DISP_W, height: dispH, display: "block" }} />
                <div style={{ position: "absolute", left: left * scale, top: 0, width: 2, height: dispH, background: "#38e4ff" }} />
                <div style={{ position: "absolute", left: right * scale, top: 0, width: 2, height: dispH, background: "#38e4ff" }} />
                <div style={{ position: "absolute", left: 0, top: strike * scale, width: DISP_W, height: 2, background: "#ff4d4d" }} />
                <div style={{ position: "absolute", left: 0, top: bottom * scale, width: DISP_W, height: 2, background: "#ffb020" }} />
              </div>

              <div style={{ display: "flex", gap: 8, margin: "14px 0 6px", alignItems: "center" }}>
                <span className="label">Detector</span>
                {(["auto", "bars", "keylight"] as const).map((d) => (
                  <button key={d} onClick={() => setDetector(d)} className="panel-2"
                    style={{ padding: "6px 12px", borderRadius: 7, cursor: "pointer", color: "var(--text)",
                      border: `1px solid ${detector === d ? "var(--accent)" : "var(--border)"}`, fontSize: 12 }}>
                    {d === "auto" ? "Auto" : d === "bars" ? "Falling bars" : "Key highlight"}
                  </button>
                ))}
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginBottom: 10 }}>
                If the video shows real hands over the keyboard, choose <strong>Falling bars</strong>.
              </div>

              <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, color: "var(--text)", marginBottom: 8 }}>
                <input type="checkbox" checked={override} onChange={(e) => setOverride(e.target.checked)} />
                Override keyboard (auto detected {geo.low_name}–{geo.high_name}, {geo.confidence < 0.6 ? "low" : "ok"} confidence)
              </label>

              {override && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 16px", fontSize: 11, color: "var(--text-dim)" }}>
                  <Slider label="Left edge" v={left} set={setLeft} max={geo.frame_width} />
                  <Slider label="Right edge" v={right} set={setRight} max={geo.frame_width} />
                  <Slider label="Strike line" v={strike} set={setStrike} max={geo.frame_height} />
                  <Slider label="Bottom" v={bottom} set={setBottom} max={geo.frame_height} />
                  <KeySelect label="First key" v={firstKey} set={setFirstKey} />
                  <KeySelect label="Last key" v={lastKey} set={setLastKey} />
                </div>
              )}
            </>
          )}

          <div style={{ display: "flex", gap: 8, marginTop: 16, justifyContent: "flex-end" }}>
            <button className="btn" onClick={onCancel} disabled={busy}>Back</button>
            <button className="btn btn-primary" onClick={go} disabled={!geo || busy}>
              {busy ? "Starting…" : "Analyze"}
            </button>
          </div>
          {err && <div className="mono" style={{ marginTop: 10, color: "var(--danger)", fontSize: 12 }}>{err}</div>}
        </div>
      </div>
    </div>
  );
}

function Slider({ label, v, set, max }: { label: string; v: number; set: (n: number) => void; max: number }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span>{label}: {Math.round(v)}</span>
      <input type="range" min={0} max={max} value={v} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
}

function KeySelect({ label, v, set }: { label: string; v: string; set: (s: string) => void }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span>{label}</span>
      <select className="input mono" value={v} onChange={(e) => set(e.target.value)} style={{ padding: "4px 6px" }}>
        {WHITE_KEYS.map((k) => <option key={k} value={k}>{k}</option>)}
      </select>
    </label>
  );
}
