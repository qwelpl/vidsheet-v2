"use client";
import { Note, Project } from "@/lib/api";
import { HAND_COLOR, confidenceTone } from "@/lib/render";

interface Props {
  project: Project;
  note: Note | null;
  onEdit: (id: number, patch: Partial<Note>) => void;
  onDelete: (id: number) => void;
}

function Bar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const col = value >= 0.8 ? "#35b36b" : value >= 0.55 ? "#e0a52a" : "#e0533a";
  return (
    <div style={{ marginBottom: 7 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-dim)" }}>
        <span>{label}</span><span className="mono">{pct}%</span>
      </div>
      <div style={{ height: 4, background: "#0b0c10", borderRadius: 2, marginTop: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: col, borderRadius: 2 }} />
      </div>
    </div>
  );
}

export default function Inspector({ project, note, onEdit, onDelete }: Props) {
  return (
    <div style={{ height: "100%", overflow: "auto", padding: 12 }}>
      {note ? (
        <div className="fade-in">
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: HAND_COLOR[note.hand] }} />
            <span style={{ fontSize: 18, fontWeight: 600 }} className="mono">{note.name}</span>
            <span className="chip">MIDI {note.midi}</span>
            {confidenceTone(note) !== "high" && (
              <span className="chip" style={{ color: confidenceTone(note) === "low" ? "#e0533a" : "#e0a52a" }}>
                review
              </span>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 12 }} className="mono">
            <Field label="Start" value={note.start.toFixed(3) + "s"} />
            <Field label="End" value={note.end.toFixed(3) + "s"} />
            <Field label="Duration" value={(note.duration * 1000).toFixed(0) + " ms"} />
            <Field label="Velocity" value={String(note.velocity) + (note.velocity_observed ? "" : " *")} />
            <Field label="Hand" value={note.hand} />
            <Field label="Frames" value={String(note.source_frames.length)} />
          </div>

          <div className="label" style={{ marginBottom: 6 }}>Confidence</div>
          <Bar label="Detection" value={note.detection_confidence} />
          <Bar label="Pitch" value={note.pitch_confidence} />
          <Bar label="Timing" value={note.timing_confidence} />
          <Bar label="Duration" value={note.duration_confidence} />
          <Bar label="Hand" value={note.hand_confidence} />

          {note.issues.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div className="label" style={{ marginBottom: 4 }}>Flags</div>
              {note.issues.map((i) => (
                <span key={i} className="chip" style={{ margin: "0 4px 4px 0", color: "#e0a52a" }}>{i}</span>
              ))}
            </div>
          )}

          <div className="divider" style={{ margin: "12px 0" }} />
          <div className="label" style={{ marginBottom: 6 }}>Corrections</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button className="btn" onClick={() => onEdit(note.id, { hand: note.hand === "left" ? "right" : "left", manually_corrected: true })}>
              Flip hand
            </button>
            <button className="btn" onClick={() => onEdit(note.id, { start: +(note.start - 0.01).toFixed(4), manually_corrected: true })}>
              −10ms start
            </button>
            <button className="btn" onClick={() => onEdit(note.id, { start: +(note.start + 0.01).toFixed(4), manually_corrected: true })}>
              +10ms start
            </button>
            <button className="btn" onClick={() => onEdit(note.id, { end: +(note.end - 0.01).toFixed(4), manually_corrected: true })}>
              −10ms end
            </button>
            <button className="btn" onClick={() => onEdit(note.id, { end: +(note.end + 0.01).toFixed(4), manually_corrected: true })}>
              +10ms end
            </button>
          </div>
          <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8 }}>
            <span className="label">Velocity</span>
            <input
              type="range" min={1} max={127} value={note.velocity}
              onChange={(e) => onEdit(note.id, { velocity: +e.target.value, manually_corrected: true })}
              style={{ flex: 1 }}
            />
          </div>
          <button className="btn" style={{ marginTop: 10, color: "#e0533a", width: "100%", justifyContent: "center" }}
            onClick={() => onDelete(note.id)}>
            Delete note
          </button>
          {note.manually_corrected && (
            <div style={{ marginTop: 8, fontSize: 11, color: "var(--accent)" }}>● manually corrected</div>
          )}
        </div>
      ) : (
        <ReportPanel project={project} />
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel-2" style={{ padding: "6px 8px", borderRadius: 5 }}>
      <div className="label" style={{ fontSize: 9 }}>{label}</div>
      <div style={{ fontSize: 12, marginTop: 1 }}>{value}</div>
    </div>
  );
}

function ReportPanel({ project }: { project: Project }) {
  const r = project.report;
  const gt = project.ground_truth;
  const stat = (label: string, value: string | number, col?: string) => (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 12 }}>
      <span style={{ color: "var(--text-dim)" }}>{label}</span>
      <span className="mono" style={{ color: col || "var(--text)" }}>{value}</span>
    </div>
  );
  return (
    <div className="fade-in">
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Reconstruction report</div>
      <div className="panel-2" style={{ padding: 10, borderRadius: 6, marginBottom: 10 }}>
        {stat("Notes detected", r.notes_total)}
        {stat("High confidence", r.high_confidence, "#35b36b")}
        {stat("Medium confidence", r.medium_confidence, "#e0a52a")}
        {stat("Low confidence", r.low_confidence, r.low_confidence ? "#e0533a" : undefined)}
        {stat("Needs review", r.needs_review, r.needs_review ? "#e0a52a" : undefined)}
        {stat("Possible missing", r.possible_missing)}
        {stat("Possible duplicates", r.possible_duplicates)}
        {stat("Onset uncertainty", `±${r.mean_onset_uncertainty_ms.toFixed(2)} ms`)}
        {stat("Keyboard", `${project.geometry.low_name}–${project.geometry.high_name}`)}
        {stat("Tempo", `${project.tempo.bpm.toFixed(1)} BPM`)}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.5 }}>{r.notes}</div>

      {gt && (
        <div className="panel-2" style={{ padding: 10, borderRadius: 6, marginTop: 10 }}>
          <div className="label" style={{ marginBottom: 6 }}>Ground-truth comparison</div>
          {stat("Precision", gt.precision.toFixed(3), "#35b36b")}
          {stat("Recall", gt.recall.toFixed(3), "#35b36b")}
          {stat("F1", gt.f1.toFixed(3), "#35b36b")}
          {stat("Onset error", gt.mean_onset_error_ms != null ? `${gt.mean_onset_error_ms} ms` : "—")}
          {stat("Duration error", gt.mean_duration_error_ms != null ? `${gt.mean_duration_error_ms} ms` : "—")}
          {stat("False positive / negative", `${gt.false_positive} / ${gt.false_negative}`)}
        </div>
      )}
      <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 10, lineHeight: 1.5 }}>
        Confidence reflects visual evidence strength, not verified accuracy. Select any
        note to inspect and correct it.
      </div>
    </div>
  );
}
