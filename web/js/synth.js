// Sintetizador polifónico con Web Audio: una "ranura" por mano, cada una con su acorde sonando.

const PRESETS = {
  pad: {
    osc: [['sawtooth', 1, -7, 0.5], ['sawtooth', 1, 7, 0.5], ['sine', 0.5, 0, 0.4]],
    // ataque corto para que el acorde se escuche apenas cambia el gesto
    cutoff: 1400, q: 0.7, attack: 0.03, decay: null, sustain: 1, release: 0.5,
  },
  organo: {
    osc: [['sine', 1, 0, 0.6], ['sine', 2, 0, 0.3], ['sine', 3, 0, 0.15], ['sine', 4, 0, 0.08]],
    cutoff: 6000, q: 0.5, attack: 0.02, decay: null, sustain: 1, release: 0.25,
  },
  pulso: {
    osc: [['triangle', 1, 0, 0.8], ['square', 2, 0, 0.08]],
    cutoff: 2500, q: 1, attack: 0.005, decay: 0.35, sustain: 0, release: 0.3,
  },
  chip: {
    osc: [['square', 1, 0, 0.35], ['square', 2, 3, 0.1]],
    cutoff: 4000, q: 0.5, attack: 0.01, decay: null, sustain: 1, release: 0.1,
  },
};

const NOTE_LEVEL = 0.16;
const mtof = (m) => 440 * 2 ** ((m - 69) / 12);

function impulseResponse(ctx, seconds = 2.6, decay = 2.4) {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len) ** decay;
  }
  return buf;
}

export class Synth {
  constructor() {
    this.ctx = null;
    this.slots = new Map();
    this.instrument = 'pad';
  }

  async start() {
    if (this.ctx) return;
    const ctx = (this.ctx = new AudioContext({ latencyHint: 'interactive' }));
    this.master = ctx.createGain();
    this.master.gain.value = 0.7;
    const comp = ctx.createDynamicsCompressor();
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.master.connect(comp).connect(this.analyser).connect(ctx.destination);

    // bus -> seco + reverb
    this.bus = ctx.createGain();
    this.wet = ctx.createGain();
    this.wet.gain.value = 0.35;
    const reverb = ctx.createConvolver();
    reverb.buffer = impulseResponse(ctx);
    this.bus.connect(this.master);
    this.bus.connect(reverb).connect(this.wet).connect(this.master);
    await ctx.resume();
  }

  setVolume(v) { if (this.master) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05); }
  setReverb(v) { if (this.wet) this.wet.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05); }
  setInstrument(name) { this.instrument = name; }

  /** Hace sonar `notes` (MIDI) en la ranura; si ya sonaban esas mismas notas no hace nada. */
  play(slot, notes) {
    if (!this.ctx) return;
    const key = `${this.instrument}:${notes.join(',')}`;
    if (this.slots.get(slot)?.key === key) return;
    this.release(slot);
    if (!notes.length) return;
    this.slots.set(slot, { key, voices: notes.map((n) => this._voice(n)) });
  }

  release(slot) {
    this.slots.get(slot)?.voices.forEach((v) => v.stop());
    this.slots.delete(slot);
  }

  releaseAll() { [...this.slots.keys()].forEach((s) => this.release(s)); }

  _voice(midi) {
    const { ctx } = this;
    const p = PRESETS[this.instrument];
    const t = ctx.currentTime;
    const f = mtof(midi);

    const out = ctx.createGain();
    out.gain.setValueAtTime(0, t);
    out.gain.linearRampToValueAtTime(NOTE_LEVEL, t + p.attack);
    if (p.decay) out.gain.setTargetAtTime(NOTE_LEVEL * p.sustain, t + p.attack, p.decay);
    out.connect(this.bus);

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = p.cutoff;
    filter.Q.value = p.q;
    filter.connect(out);

    const oscs = p.osc.map(([type, mult, detune, gain]) => {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = f * mult;
      o.detune.value = detune;
      const g = ctx.createGain();
      g.gain.value = gain;
      o.connect(g).connect(filter);
      o.start(t);
      return o;
    });

    return {
      stop() {
        const now = ctx.currentTime;
        out.gain.cancelScheduledValues(now);
        out.gain.setValueAtTime(out.gain.value, now);
        out.gain.setTargetAtTime(0, now, p.release / 4);
        oscs.forEach((o) => o.stop(now + p.release * 1.5));
        setTimeout(() => out.disconnect(), (p.release * 1.5 + 0.1) * 1000);
      },
    };
  }
}
