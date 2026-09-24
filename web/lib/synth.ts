// Lightweight Web Audio piano for auditioning the reconstruction (§46).
// Voices are triggered from the transport clock so playback stays in sync with
// the video and honours the playback rate.

function midiToFreq(m: number): number {
  return 440 * Math.pow(2, (m - 69) / 12);
}

export class PianoSynth {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private active = new Set<{ stop: (t: number) => void }>();

  private ensure() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.9;
      this.master.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  // A short, piano-ish struck tone: two detuned partials + lowpass + ADSR.
  noteOn(midi: number, velocity: number, durationSec: number) {
    const ctx = this.ensure();
    const now = ctx.currentTime;
    const f = midiToFreq(midi);
    const vel = Math.max(0.08, Math.min(1, velocity / 127));
    const dur = Math.max(0.05, durationSec);

    const g = ctx.createGain();
    const filt = ctx.createBiquadFilter();
    filt.type = "lowpass";
    filt.frequency.value = 1800 + vel * 4200;
    filt.Q.value = 0.6;
    g.connect(filt);
    filt.connect(this.master!);

    const oscs: OscillatorNode[] = [];
    [[1, "triangle"], [2, "sine"], [1.001, "sawtooth"]].forEach(([mult, type], i) => {
      const o = ctx.createOscillator();
      o.type = type as OscillatorType;
      o.frequency.value = f * (mult as number);
      const og = ctx.createGain();
      og.gain.value = i === 0 ? 1 : i === 1 ? 0.35 : 0.12;
      o.connect(og);
      og.connect(g);
      o.start(now);
      oscs.push(o);
    });

    const peak = 0.28 * vel;
    const sustain = peak * 0.35;
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(peak, now + 0.006);   // attack
    g.gain.exponentialRampToValueAtTime(Math.max(0.0002, sustain), now + 0.12); // decay
    const rel = now + dur;
    g.gain.setValueAtTime(Math.max(0.0002, sustain), rel);
    g.gain.exponentialRampToValueAtTime(0.0001, rel + 0.18);  // release

    const voice = {
      stop: (t: number) => {
        try {
          g.gain.cancelScheduledValues(t);
          g.gain.setTargetAtTime(0.0001, t, 0.03);
          oscs.forEach((o) => o.stop(t + 0.2));
        } catch {}
      },
    };
    oscs.forEach((o) => (o.onended = () => this.active.delete(voice)));
    oscs.forEach((o) => o.stop(rel + 0.4));
    this.active.add(voice);
  }

  allOff() {
    const t = this.ctx?.currentTime ?? 0;
    this.active.forEach((v) => v.stop(t));
    this.active.clear();
  }

  resume() {
    this.ensure();
  }
}
