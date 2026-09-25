// Web Audio piano for auditioning the reconstruction (§46).
// A physically-inspired voice: a harmonic-rich periodic wave (piano-like partial
// rolloff) with a slightly detuned second string for beating, a short filtered
// hammer-noise transient at the attack, a two-stage amplitude decay (fast head +
// slow tail), and a low-pass that closes over the note's life to mimic string
// damping. Far closer to a real piano than plain oscillators, and dependency-free.

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

  private ensure() {
    if (!this.ctx) {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.ctx = ctx;
      // gentle bus compression + headroom so chords don't clip
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18;
      comp.knee.value = 24;
      comp.ratio.value = 3;
      comp.attack.value = 0.003;
      comp.release.value = 0.25;
      const master = ctx.createGain();
      master.gain.value = 0.55;
      comp.connect(master);
      master.connect(ctx.destination);
      this.comp = comp;
      this.master = master;

      // piano-like partial amplitudes (odd/even mix, gentle rolloff)
      const partials = [0, 1, 0.62, 0.44, 0.30, 0.20, 0.14, 0.10, 0.072,
        0.052, 0.038, 0.027, 0.02, 0.014, 0.01, 0.007];
      const real = new Float32Array(partials.length);
      const imag = new Float32Array(partials.length);
      for (let i = 0; i < partials.length; i++) imag[i] = partials[i];
      this.wave = ctx.createPeriodicWave(real, imag, { disableNormalization: false });

      // 0.4 s of white noise for the hammer transient
      const n = Math.floor(ctx.sampleRate * 0.4);
      const buf = ctx.createBuffer(1, n, ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      this.noise = buf;
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  noteOn(midi: number, velocity: number, durationSec: number) {
    const ctx = this.ensure();
    const now = ctx.currentTime;
    const f = midiToFreq(midi);
    const vel = Math.max(0.06, Math.min(1, velocity / 127));
    const dur = Math.max(0.06, durationSec);
    // lower notes ring longer; higher notes decay fast (real piano behaviour)
    const tail = Math.min(6, 1.1 + Math.pow(2, (60 - midi) / 12) * 1.3);

    const out = ctx.createGain();      // master amplitude envelope
    out.connect(this.comp!);

    // brightness filter that closes as the note rings out
    const filt = ctx.createBiquadFilter();
    filt.type = "lowpass";
    filt.Q.value = 0.4;
    const bright = 1200 + vel * 6500 + f * 3;
    filt.frequency.setValueAtTime(bright, now);
    filt.frequency.exponentialRampToValueAtTime(Math.max(400, f * 4 + 500), now + Math.min(tail, 1.6));
    filt.connect(out);

    // two slightly detuned "strings" for natural beating/chorus
    const oscs: OscillatorNode[] = [];
    for (const cents of [-2.5, 2.5]) {
      const o = ctx.createOscillator();
      o.setPeriodicWave(this.wave!);
      o.frequency.value = f;
      o.detune.value = cents;
      const og = ctx.createGain();
      og.gain.value = 0.5;
      o.connect(og);
      og.connect(filt);
      o.start(now);
      oscs.push(o);
    }

    // hammer-noise transient (short, band-limited, velocity-scaled)
    const nsrc = ctx.createBufferSource();
    nsrc.buffer = this.noise;
    const nfilt = ctx.createBiquadFilter();
    nfilt.type = "bandpass";
    nfilt.frequency.value = Math.min(6000, f * 5 + 1200);
    nfilt.Q.value = 0.7;
    const ng = ctx.createGain();
    ng.gain.setValueAtTime(0.0001, now);
    ng.gain.exponentialRampToValueAtTime(0.18 * vel, now + 0.002);
    ng.gain.exponentialRampToValueAtTime(0.0001, now + 0.06);
    nsrc.connect(nfilt); nfilt.connect(ng); ng.connect(out);
    nsrc.start(now); nsrc.stop(now + 0.12);

    // amplitude: fast attack, quick initial decay to a sustain shelf, long tail
    const peak = 0.5 * (0.4 + 0.6 * vel);
    const sustain = peak * 0.28;
    const g = out.gain;
    g.setValueAtTime(0.0001, now);
    g.exponentialRampToValueAtTime(peak, now + 0.004);
    g.exponentialRampToValueAtTime(Math.max(0.0004, sustain), now + 0.14);
    // hold roughly for the note, then release into the ringing tail
    const rel = now + dur;
    g.setTargetAtTime(0.0001, rel, Math.max(0.08, tail * 0.18));

    const stopAt = rel + tail * 0.6 + 0.15;
    const voice = {
      stop: (t: number) => {
        try {
          g.cancelScheduledValues(t);
          g.setTargetAtTime(0.0001, t, 0.06);
          oscs.forEach((o) => o.stop(t + 0.4));
        } catch {}
      },
    };
    oscs.forEach((o) => (o.onended = () => this.active.delete(voice)));
    oscs.forEach((o) => o.stop(stopAt));
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
