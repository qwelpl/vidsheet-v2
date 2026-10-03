// Backend client + reconstruction data types.

// Always the Render-hosted engine (see render.yaml) - including local dev, so a
// stray local engine on :8000 can never serve requests. Override only to point
// at another remote engine; a localhost value is ignored to keep runs off this
// machine.
const ENGINE = "https://reprise-engine.onrender.com";
const override = process.env.NEXT_PUBLIC_API;
export const API =
  override && !/^https?:\/\/(localhost|127\.0\.0\.1)/i.test(override)
    ? override
    : ENGINE;

export type Hand = "left" | "right" | "unknown";

export interface Note {
  id: number;
  midi: number;
  name: string;
  start: number;
  end: number;
  duration: number;
  hand: Hand;
  velocity: number;
  detection_confidence: number;
  pitch_confidence: number;
  timing_confidence: number;
  duration_confidence: number;
  hand_confidence: number;
  velocity_observed: boolean;
  overall_confidence: number;
  color_cluster: number | null;
  lane_x: number | null;
  issues: string[];
  verification: string;
  manually_corrected: boolean;
  source_frames: number[];
}

export interface Lane {
  midi: number;
  name: string;
  center: number;
  half_width: number;
  is_black: boolean;
}

export interface Geometry {
  width: number;
  height: number;
  keyboard_top: number;
  keyboard_bottom: number;
  strike_y: number;
  low_midi: number;
  high_midi: number;
  low_name: string;
  high_name: string;
  note_direction: string;
  confidence: number;
  manual: boolean;
  lanes: Lane[];
}

export interface Report {
  notes_total: number;
  high_confidence: number;
  medium_confidence: number;
  low_confidence: number;
  needs_review: number;
  possible_missing: number;
  possible_duplicates: number;
  pitch_conflicts: number;
  timing_conflicts: number;
  mean_onset_uncertainty_ms: number;
  keyboard_low_midi: number;
  keyboard_high_midi: number;
  visual_verified: boolean;
  ground_truth_compared: boolean;
  notes: string;
}

export interface Meta {
  width: number;
  height: number;
  fps: number;
  duration: number;
  frame_count: number;
  title: string;
  variable_fps: boolean;
}

export interface Tempo {
  bpm: number;
  beat_period: number;
  beat_phase: number;
  time_signature: number[];
  confidence: number;
  segments: { start: number; bpm: number; beats: number[] }[];
}

export interface Project {
  meta: Meta;
  geometry: Geometry;
  theme: { clusters: { hue: number }[]; rainbow: boolean; sat_min: number; val_min: number };
  notes: Note[];
  tempo: Tempo;
  quantized: { id: number; midi: number; beat: number; end_beat: number; error_ms: number; ambiguous: boolean }[];
  report: Report;
  fall_speed: number;
  ground_truth: null | {
    precision: number; recall: number; f1: number;
    true_positive: number; false_positive: number; false_negative: number;
    mean_onset_error_ms: number | null; mean_duration_error_ms: number | null;
  };
}

export interface JobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  stages: { stage: string; message: string; progress: number }[];
  error: string | null;
  title: string;
  has_result: boolean;
}

async function post(path: string, body: FormData): Promise<JobStatus> {
  const r = await fetch(`${API}${path}`, { method: "POST", body });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function startYoutube(url: string, preset: string) {
  const f = new FormData();
  f.set("url", url);
  f.set("preset", preset);
  return post("/api/analyze/youtube", f);
}

export function startUpload(file: File, preset: string) {
  const f = new FormData();
  f.set("file", file);
  f.set("preset", preset);
  return post("/api/analyze/upload", f);
}

export function startDemo(preset = "maximum") {
  const f = new FormData();
  f.set("preset", preset);
  return post("/api/demo", f);
}

export async function getJob(id: string): Promise<JobStatus> {
  const r = await fetch(`${API}/api/jobs/${id}`);
  if (!r.ok) throw new Error("job fetch failed");
  return r.json();
}

export async function getResult(id: string): Promise<Project> {
  const r = await fetch(`${API}/api/jobs/${id}/result`);
  if (!r.ok) throw new Error("result fetch failed");
  return r.json();
}

export const videoUrl = (id: string) => `${API}/api/jobs/${id}/video`;
export const exportUrl = (id: string, fmt: string) => `${API}/api/jobs/${id}/export/${fmt}`;

// Persist UI-edited notes so the server regenerates the export files (MIDI etc.)
// to match the corrections before the user downloads them.
export async function syncNotes(id: string, notes: Note[]): Promise<void> {
  const r = await fetch(`${API}/api/jobs/${id}/notes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  });
  if (!r.ok) throw new Error(await r.text());
}
