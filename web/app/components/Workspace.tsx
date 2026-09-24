"use client";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Project, Note, videoUrl, exportUrl } from "@/lib/api";
import { fmtTime, confidenceTone } from "@/lib/render";
import PianoRoll from "./PianoRoll";
import ReconView from "./ReconView";
import Inspector from "./Inspector";

type Mode = "original" | "reconstruction" | "split" | "overlay" | "difference";

export default function Workspace({ jobId, project, onExit }: { jobId: string; project: Project; onExit: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState<Mode>("split");
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [selected, setSelected] = useState<Note | null>(null);
  const [uncertainOnly, setUncertainOnly] = useState(false);
  const [showBeats, setShowBeats] = useState(true);
  const [frameInspect, setFrameInspect] = useState(false);
  const [pxPerSec, setPxPerSec] = useState(90);
  const [scrollX, setScrollX] = useState(0);
  const [exportOpen, setExportOpen] = useState(false);

  // editable notes with undo/redo (§26, §60)
  const [notes, setNotes] = useState<Note[]>(project.notes);
  const undo = useRef<Note[][]>([]);
  const redo = useRef<Note[][]>([]);
  useEffect(() => { setNotes(project.notes); undo.current = []; redo.current = []; }, [project]);

  const fps = project.meta.fps || 30;
  const duration = project.meta.duration;

  const push = useCallback((next: Note[]) => {
    undo.current.push(notes); redo.current = []; setNotes(next);
  }, [notes]);

  const editNote = useCallback((id: number, patch: Partial<Note>) => {
    push(notes.map((n) => (n.id === id ? { ...n, ...patch, duration: patch.end != null || patch.start != null ? (patch.end ?? n.end) - (patch.start ?? n.start) : n.duration } : n)));
    setSelected((s) => (s && s.id === id ? { ...s, ...patch } : s));
  }, [notes, push]);

  const deleteNote = useCallback((id: number) => {
    push(notes.filter((n) => n.id !== id));
    setSelected(null);
  }, [notes, push]);

  const doUndo = useCallback(() => {
    const prev = undo.current.pop();
    if (prev) { redo.current.push(notes); setNotes(prev); }
  }, [notes]);
  const doRedo = useCallback(() => {
    const nxt = redo.current.pop();
    if (nxt) { undo.current.push(notes); setNotes(nxt); }
  }, [notes]);

  // playback clock
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const v = video.current;
      if (v) setTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => { if (video.current) video.current.playbackRate = speed; }, [speed]);

  const togglePlay = useCallback(() => {
    const v = video.current; if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); } else { v.pause(); setPlaying(false); }
  }, []);

  const seek = useCallback((t: number) => {
    const v = video.current; if (!v) return;
    v.currentTime = Math.max(0, Math.min(duration, t));
    setTime(v.currentTime);
  }, [duration]);

  const step = useCallback((frames: number) => {
    const v = video.current; if (!v) return;
    v.pause(); setPlaying(false);
    v.currentTime = Math.max(0, Math.min(duration, v.currentTime + frames / fps));
  }, [duration, fps]);

  // keyboard shortcuts (§26)
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      else if (e.key === "ArrowRight") step(e.shiftKey ? 10 : 1);
      else if (e.key === "ArrowLeft") step(e.shiftKey ? -10 : -1);
      else if (e.key === "u") setUncertainOnly((x) => !x);
      else if ((e.metaKey || e.ctrlKey) && e.key === "z") { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); }
      else if (e.key === "Delete" && selected) deleteNote(selected.id);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [togglePlay, step, doUndo, doRedo, selected, deleteNote]);

  // keep playhead in view
  useEffect(() => {
    const win = 900 / pxPerSec;
    if (time < scrollX || time > scrollX + win) setScrollX(Math.max(0, time - win * 0.3));
  }, [time, pxPerSec, scrollX]);

  const uncertainCount = useMemo(() => notes.filter((n) => confidenceTone(n) !== "high").length, [notes]);
  const src = videoUrl(jobId);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* top bar */}
      <div className="panel" style={{ borderLeft: 0, borderRight: 0, borderTop: 0, padding: "7px 12px", display: "flex", alignItems: "center", gap: 12 }}>
        <button className="btn btn-ghost" onClick={onExit}>← New</button>
        <div style={{ fontWeight: 600 }}>{project.meta.title || "Reconstruction"}</div>
        <span className="chip">{project.meta.width}×{project.meta.height}</span>
        <span className="chip">{fps.toFixed(project.meta.variable_fps ? 2 : 0)} fps</span>
        <span className="chip">{project.geometry.low_name}–{project.geometry.high_name}</span>
        <span className="chip">{notes.length} notes</span>
        {uncertainCount > 0 && <span className="chip" style={{ color: "#e0a52a" }}>{uncertainCount} to review</span>}
        {project.ground_truth && (
          <span className="chip" style={{ color: "#35b36b" }}>F1 {project.ground_truth.f1.toFixed(3)}</span>
        )}
        <div style={{ flex: 1 }} />
        <div className="seg">
          {(["original", "reconstruction", "split", "overlay", "difference"] as Mode[]).map((m) => (
            <button key={m} className={mode === m ? "active" : ""} onClick={() => setMode(m)}>{m}</button>
          ))}
        </div>
        <div style={{ position: "relative" }}>
          <button className="btn btn-primary" onClick={() => setExportOpen((o) => !o)}>Export ▾</button>
          {exportOpen && (
            <div className="panel" style={{ position: "absolute", right: 0, top: 34, borderRadius: 8, padding: 6, zIndex: 20, minWidth: 150 }}>
              {["midi", "musicxml", "csv", "json"].map((f) => (
                <a key={f} href={exportUrl(jobId, f)} className="btn btn-ghost" style={{ display: "flex", width: "100%", justifyContent: "flex-start" }}>
                  {f.toUpperCase()}
                </a>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* main viewers */}
      <div style={{ flex: "1 1 55%", display: "flex", minHeight: 0, borderBottom: "1px solid var(--border)" }}>
        <div style={{ flex: 1, display: "flex", minWidth: 0 }}>
          {/* left / video pane */}
          {mode !== "reconstruction" && (
            <div style={{ position: "relative", flex: 1, background: "#000", minWidth: 0 }}>
              <video
                ref={video}
                src={src}
                crossOrigin="anonymous"
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain",
                  display: mode === "difference" || mode === "overlay" || mode === "split" || mode === "original" ? "block" : "none" }}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                playsInline
              />
              {(mode === "overlay" || mode === "difference") && (
                <div style={{ position: "absolute", inset: 0, mixBlendMode: mode === "difference" ? "difference" : "normal" }}>
                  <ReconView project={project} notes={notes} time={time} transparent opacity={mode === "difference" ? 1 : overlayOpacity} selectedId={selected?.id ?? null} />
                </div>
              )}
              {frameInspect && (
                <div style={{ position: "absolute", inset: 0 }}>
                  <ReconView project={project} notes={notes} time={time} transparent opacity={0.85} selectedId={selected?.id ?? null} />
                </div>
              )}
              <PaneLabel text={mode === "difference" ? "difference" : mode === "overlay" ? "overlay" : "original"} />
              {frameInspect && (
                <div className="mono" style={{ position: "absolute", left: 8, bottom: 8, fontSize: 11, background: "#000a", padding: "4px 8px", borderRadius: 5, color: "#cfd3dc" }}>
                  frame {Math.round(time * fps)} · {fmtTime(time)} · strike y={project.geometry.strike_y.toFixed(0)} · v={project.fall_speed.toFixed(0)}px/s
                </div>
              )}
            </div>
          )}
          {/* right / reconstruction pane */}
          {(mode === "split" || mode === "reconstruction") && (
            <div style={{ position: "relative", flex: 1, background: "#0a0b0e", borderLeft: mode === "split" ? "1px solid var(--border)" : 0, minWidth: 0 }}>
              <ReconView project={project} notes={notes} time={time} selectedId={selected?.id ?? null} />
              <PaneLabel text="reconstruction" />
            </div>
          )}
        </div>

        {/* sidebar */}
        <div className="panel" style={{ width: 300, borderTop: 0, borderBottom: 0, borderRight: 0, flexShrink: 0 }}>
          <Inspector project={project} note={selected} onEdit={editNote} onDelete={deleteNote} />
        </div>
      </div>

      {/* transport */}
      <div className="panel" style={{ borderLeft: 0, borderRight: 0, borderTop: 0, padding: "6px 12px", display: "flex", alignItems: "center", gap: 8 }}>
        <button className="btn" onClick={() => step(-10)} title="−10 frames (Shift+←)">⏮</button>
        <button className="btn" onClick={() => step(-1)} title="−1 frame (←)">◀|</button>
        <button className="btn btn-primary" onClick={togglePlay} style={{ minWidth: 40, justifyContent: "center" }}>{playing ? "❚❚" : "▶"}</button>
        <button className="btn" onClick={() => step(1)} title="+1 frame (→)">|▶</button>
        <button className="btn" onClick={() => step(10)} title="+10 frames (Shift+→)">⏭</button>
        <span className="mono" style={{ fontSize: 12, color: "var(--text-dim)", minWidth: 150 }}>
          {fmtTime(time)} / {fmtTime(duration)}
        </span>
        <input type="range" min={0} max={duration} step={0.001} value={time}
          onChange={(e) => seek(+e.target.value)} style={{ flex: 1 }} />
        <span className="label">Speed</span>
        <div className="seg">
          {[0.25, 0.5, 1, 2].map((s) => (
            <button key={s} className={speed === s ? "active" : ""} onClick={() => setSpeed(s)}>{s}×</button>
          ))}
        </div>
        {(mode === "overlay") && (
          <>
            <span className="label">Opacity</span>
            <input type="range" min={0} max={1} step={0.05} value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(+e.target.value)} style={{ width: 80 }} />
          </>
        )}
        <button className={"btn" + (uncertainOnly ? " btn-primary" : "")} onClick={() => setUncertainOnly((x) => !x)} title="Show only uncertain (U)">
          Uncertain{uncertainOnly ? " ●" : ""}
        </button>
        <button className={"btn" + (frameInspect ? " btn-primary" : "")} onClick={() => setFrameInspect((x) => !x)}>Frame inspector</button>
        <button className={"btn" + (showBeats ? " btn-primary" : "")} onClick={() => setShowBeats((x) => !x)}>Grid</button>
        <button className="btn" onClick={doUndo} title="Undo (⌘Z)">Undo</button>
        <button className="btn" onClick={doRedo} title="Redo (⇧⌘Z)">Redo</button>
      </div>

      {/* piano roll */}
      <div style={{ flex: "1 1 45%", minHeight: 0 }}>
        <PianoRoll
          project={project}
          notes={uncertainOnly ? notes.filter((n) => confidenceTone(n) !== "high") : notes}
          time={time}
          pxPerSec={pxPerSec}
          scrollX={scrollX}
          selectedId={selected?.id ?? null}
          uncertainOnly={false}
          showBeats={showBeats}
          onSelect={setSelected}
          onSeek={seek}
          onScroll={setScrollX}
          onZoom={setPxPerSec}
        />
      </div>
    </div>
  );
}

function PaneLabel({ text }: { text: string }) {
  return (
    <div className="mono" style={{ position: "absolute", left: 8, top: 8, fontSize: 10, textTransform: "uppercase",
      letterSpacing: ".08em", background: "#000a", padding: "3px 7px", borderRadius: 4, color: "#9aa0ad" }}>
      {text}
    </div>
  );
}
