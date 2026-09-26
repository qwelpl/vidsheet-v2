"use client";
import { useEffect, useLayoutEffect, useRef } from "react";
import { isBlack, midiName } from "@/lib/render";

interface Props {
  value: number;               // currently selected MIDI
  lowMidi: number;             // keyboard range from detected geometry
  highMidi: number;
  onPick: (midi: number) => void;
  onClose: () => void;
}

const WHITE_W = 22;
const WHITE_H = 120;
const BLACK_W = 14;
const BLACK_H = 76;

// A click-to-pick piano. Renders the detected keyboard range with real white/
// black-key layout, highlights the current pitch, and reports the picked MIDI.
export default function PianoPopup({ value, lowMidi, highMidi, onPick, onClose }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const current = useRef<HTMLButtonElement>(null);

  // dismiss on outside click / Escape
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [onClose]);

  // scroll the current key into view
  useLayoutEffect(() => {
    const s = scroller.current, c = current.current;
    if (s && c) s.scrollLeft = c.offsetLeft - s.clientWidth / 2 + WHITE_W / 2;
  }, []);

  // lay out keys: white keys advance x; black keys straddle the previous seam
  const whites: number[] = [];
  const blacks: { midi: number; x: number }[] = [];
  let wx = 0;
  for (let m = lowMidi; m <= highMidi; m++) {
    if (isBlack(m)) {
      blacks.push({ midi: m, x: wx - BLACK_W / 2 });
    } else {
      whites.push(m);
      wx += WHITE_W;
    }
  }
  const width = wx;

  const keyBtn = (m: number, black: boolean, x: number) => {
    const sel = m === value;
    return (
      <button
        key={m}
        ref={sel ? current : undefined}
        title={midiName(m)}
        onClick={() => onPick(m)}
        style={{
          position: "absolute",
          left: x,
          top: 0,
          width: black ? BLACK_W : WHITE_W - 1,
          height: black ? BLACK_H : WHITE_H,
          border: "1px solid #000",
          borderTop: 0,
          borderBottomLeftRadius: 3,
          borderBottomRightRadius: 3,
          background: sel
            ? "var(--accent)"
            : black
            ? "#1a1c22"
            : "#eef0f4",
          color: black ? "#cfd3dc" : "#20232a",
          zIndex: black ? 2 : 1,
          cursor: "pointer",
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "center",
          paddingBottom: 3,
          fontSize: 8,
          fontFamily: "var(--mono, monospace)",
          lineHeight: 1,
        }}
      >
        {/* label only C's (white) and the selected key, to keep it readable */}
        {sel ? midiName(m) : !black && m % 12 === 0 ? midiName(m) : ""}
      </button>
    );
  };

  return (
    <div
      ref={box}
      className="panel fade-in"
      style={{
        position: "absolute",
        left: 0,
        top: "calc(100% + 6px)",
        zIndex: 40,
        padding: 8,
        borderRadius: 8,
        width: 340,
        boxShadow: "0 8px 30px #000a",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <span className="label">Pick pitch</span>
        <span className="chip mono">{midiName(value)} · MIDI {value}</span>
      </div>
      <div ref={scroller} style={{ overflowX: "auto", overflowY: "hidden", paddingBottom: 4 }}>
        <div style={{ position: "relative", width, height: WHITE_H }}>
          {whites.map((m, i) => keyBtn(m, false, i * WHITE_W))}
          {blacks.map((b) => keyBtn(b.midi, true, b.x))}
        </div>
      </div>
    </div>
  );
}
