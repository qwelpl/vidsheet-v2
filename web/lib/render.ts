import { Note, Geometry } from "./api";

export const HAND_COLOR: Record<string, string> = {
  left: "#2e6df6",
  right: "#35b36b",
  unknown: "#8b90a0",
};

export const HAND_COLOR_DIM: Record<string, string> = {
  left: "#1c3f8f",
  right: "#1f6b41",
  unknown: "#4a4e5a",
};

// Confidence -> border treatment (§27): high = none, medium = amber, low = red.
export function confidenceTone(n: Note): "high" | "medium" | "low" {
  if (n.overall_confidence >= 0.8 && n.issues.length === 0) return "high";
  if (n.overall_confidence >= 0.55) return "medium";
  return "low";
}

export function isBlack(midi: number): boolean {
  return [1, 3, 6, 8, 10].includes(((midi % 12) + 12) % 12);
}

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// MIDI number -> scientific pitch name, e.g. 60 -> "C4". Mirrors the engine's
// midi_to_name so edited notes keep the same naming convention.
export function midiName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  return NOTE_NAMES_SHARP[((midi % 12) + 12) % 12] + octave;
}

// Predicted vertical span (roll pixels) of a note bar at time t, mirroring the
// engine renderer so the reconstruction view matches the original geometry.
export function noteBarAt(
  n: Note,
  geom: Geometry,
  v: number,
  t: number
): [number, number] | null {
  const strike = geom.strike_y;
  const lead = strike - v * (n.start - t);
  const trail = strike - v * (n.end - t);
  const yBottom = Math.min(lead, strike);
  const yTop = trail;
  if (yBottom < 0 || yTop > strike) return null;
  return [Math.max(0, yTop), Math.min(strike, yBottom)];
}

export function laneByMidi(geom: Geometry) {
  const m = new Map<number, Geometry["lanes"][number]>();
  for (const l of geom.lanes) m.set(l.midi, l);
  return m;
}

export function fmtTime(s: number): string {
  if (!isFinite(s)) return "0:00.000";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 1000);
  return `${m}:${sec.toString().padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
}
