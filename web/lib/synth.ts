// Piano playback for auditioning the reconstruction (§46).
//
// Primary: a real *sampled* grand (smplr's SplendidGrandPiano — recorded
// multi-velocity samples), so it actually sounds like a piano. The samples load
// from a CDN on first use; until they are ready (or if offline) a self-contained
// synthesised voice is used so playback always works.

import { SplendidGrandPiano } from "smplr";

function midiToFreq(m: number): number {
  return 440 * Math.pow(2, (m - 69) / 12);
}

export class PianoSynth {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private comp: DynamicsCompressorNode | null = null;
  private wave: PeriodicWave | null = null;
  private noise: AudioBuffer | null = null;
  private active = new Set<{ stop: (t: number) => void }>();

  private piano: SplendidGrandPiano | null = null;
  private sampledReady = false;

  private ensure() {
    if (!this.ctx) {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.ctx = ctx;
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18; comp.knee.value = 24; comp.ratio.value = 3;
      comp.attack.value = 0.003; comp.release.value = 0.25;
      const master = ctx.createGain();
      master.gain.value = 0.55;
      comp.connect(master); master.connect(ctx.destination);
      this.comp = comp; this.master = master;

      const partials = [0, 1, 0.62, 0.44, 0.30, 0.20, 0.14, 0.10, 0.072,
        0.052, 0.038, 0.027, 0.02, 0.014, 0.01, 0.007];
      const real = new Float32Array(partials.length);
      const imag = new Float32Array(partials.length);
      for (let i = 0; i < partials.length; i++) imag[i] = partials[i];
      this.wave = ctx.createPeriodicWave(real, imag, { disableNormalization: false });

      const n = Math.floor(ctx.sampleRate * 0.4);
      const buf = ctx.createBuffer(1, n, ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      this.noise = buf;

      // kick off sampled-piano load; fall back to synth until it resolves
      try {
        this.piano = new SplendidGrandPiano(ctx, { volume: 110 });
        this.piano.load.then(() => { this.sampledReady = true; }).catch(() => {});
      } catch {
        this.piano = null;
      }
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  /** True once the recorded samples are playable. */
  get usingSamples(): boolean {
    return this.sampledReady;
  }

  noteOn(midi: number, velocity: number, durationSec: number) {
    const ctx = this.ensure();
    const dur = Math.max(0.06, durationSec);
    if (this.sampledReady && this.piano) {
      try {
        this.piano.start({
          note: midi,
          velocity: Math.max(1, Math.min(127, velocity)),
          time: ctx.currentTime,
          duration: dur,
        });
        return;
      } catch {
        /* fall through to synth */
      }
    }
    this.synthNote(ctx, midi, velocity, dur);
  }

  // --- self-contained fallback voice ---------------------------------------
  private synthNote(ctx: AudioContext, midi: number, velocity: number, dur: number) {
    const now = ctx.currentTime;
    const f = midiToFreq(midi);
    const vel = Math.max(0.06, Math.min(1, velocity / 127));
    const tail = Math.min(6, 1.1 + Math.pow(2, (60 - midi) / 12) * 1.3);

    const out = ctx.createGain();
    out.connect(this.comp!);
    const filt = ctx.createBiquadFilter();
    filt.type = "lowpass"; filt.Q.value = 0.4;
    filt.frequency.setValueAtTime(1200 + vel * 6500 + f * 3, now);
    filt.frequency.exponentialRampToValueAtTime(Math.max(400, f * 4 + 500), now + Math.min(tail, 1.6));
    filt.connect(out);

    const oscs: OscillatorNode[] = [];
    for (const cents of [-2.5, 2.5]) {
      const o = ctx.createOscillator();
      o.setPeriodicWave(this.wave!);
      o.frequency.value = f; o.detune.value = cents;
      const og = ctx.createGain(); og.gain.value = 0.5;
      o.connect(og); og.connect(filt); o.start(now);
      oscs.push(o);
    }
    const nsrc = ctx.createBufferSource();
    nsrc.buffer = this.noise;
    const nfilt = ctx.createBiquadFilter();
    nfilt.type = "bandpass"; nfilt.frequency.value = Math.min(6000, f * 5 + 1200); nfilt.Q.value = 0.7;
    const ng = ctx.createGain();
    ng.gain.setValueAtTime(0.0001, now);
    ng.gain.exponentialRampToValueAtTime(0.18 * vel, now + 0.002);
    ng.gain.exponentialRampToValueAtTime(0.0001, now + 0.06);
    nsrc.connect(nfilt); nfilt.connect(ng); ng.connect(out);
    nsrc.start(now); nsrc.stop(now + 0.12);

    const peak = 0.5 * (0.4 + 0.6 * vel);
    const g = out.gain;
    g.setValueAtTime(0.0001, now);
    g.exponentialRampToValueAtTime(peak, now + 0.004);
    g.exponentialRampToValueAtTime(Math.max(0.0004, peak * 0.28), now + 0.14);
    const rel = now + dur;
    g.setTargetAtTime(0.0001, rel, Math.max(0.08, tail * 0.18));

    const voice = {
      stop: (t: number) => {
        try {
          g.cancelScheduledValues(t); g.setTargetAtTime(0.0001, t, 0.06);
          oscs.forEach((o) => o.stop(t + 0.4));
        } catch {}
      },
    };
    oscs.forEach((o) => (o.onended = () => this.active.delete(voice)));
    oscs.forEach((o) => o.stop(rel + tail * 0.6 + 0.15));
    this.active.add(voice);
  }

  allOff() {
    if (this.piano) { try { this.piano.stop(); } catch {} }
    const t = this.ctx?.currentTime ?? 0;
    this.active.forEach((v) => v.stop(t));
    this.active.clear();
  }

  resume() {
    this.ensure();
  }
}
