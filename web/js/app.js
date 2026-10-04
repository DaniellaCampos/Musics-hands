import { normalizeLandmarks, countFingers, classifyGesture } from './fingers.js';
import { Synth } from './synth.js';

const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const HAND_TASK_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Acordes diatónicos de Do mayor; el orden coincide con las clases del dataset (do, re, mi, fa, sol, la, si)
const CHORDS = {
  acorde_do_1dedo: { short: 'Do', name: 'Do mayor', notes: [60, 64, 67], gesture: '1 dedo' },
  acorde_re_2dedos: { short: 'Re m', name: 'Re menor', notes: [62, 65, 69], gesture: '2 dedos' },
  acorde_mi_3dedos: { short: 'Mi m', name: 'Mi menor', notes: [64, 67, 71], gesture: 'Te quiero' },
  acorde_fa_4dedos: { short: 'Fa', name: 'Fa mayor', notes: [65, 69, 72], gesture: '4 dedos' },
  acorde_sol_5dedos: { short: 'Sol', name: 'Sol mayor', notes: [67, 71, 74], gesture: '5 dedos' },
  acorde_la_rock: { short: 'La m', name: 'La menor', notes: [69, 72, 76], gesture: 'Rock' },
  acorde_si_pinch: { short: 'Si°', name: 'Si disminuido', notes: [71, 74, 77], gesture: 'Pinch' },
};
const CHORD_KEYS = Object.keys(CHORDS);
// Modo escala: total de dedos (1–10) -> grado de Do mayor
const SCALE = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76];
const NOTE_NAMES = ['Do', 'Do♯', 'Re', 'Re♯', 'Mi', 'Fa', 'Fa♯', 'Sol', 'Sol♯', 'La', 'La♯', 'Si'];
const noteName = (m) => `${NOTE_NAMES[m % 12]}${Math.floor(m / 12) - 1}`;

const STABLE_FRAMES = 2; // frames iguales seguidos para cambiar de acorde (evita parpadeos)
const LOST_FRAMES = 8; // frames sin la mano para soltar su sonido
const PIANO_RANGE = [48, 83];
const TIPS = [4, 8, 12, 16, 20];

const $ = (sel) => document.querySelector(sel);
const video = $('#video');
const overlay = $('#overlay');
const octx = overlay.getContext('2d');
const synth = new Synth();

const state = {
  mode: 'gesture',
  running: false,
  landmarker: null,
  connections: [],
  lastVideoTime: -1,
  detMs: 0,
  frames: 0,
  fpsT: performance.now(),
};
const slots = { izq: newSlot(), der: newSlot(), total: newSlot() };
function newSlot() { return { key: null, candidate: null, n: 0, missing: 0, notes: [], label: null }; }

const css = getComputedStyle(document.documentElement);
const COLORS = { izq: css.getPropertyValue('--izq').trim(), der: css.getPropertyValue('--der').trim() };

// ---------------------------------------------------------------- arranque

$('#start').addEventListener('click', start);

async function start() {
  const btn = $('#start');
  const note = $('#gate-note');
  btn.disabled = true;
  note.classList.remove('error');
  try {
    await synth.start(); // debe ocurrir dentro del clic (política de autoplay)
    synth.setVolume(+$('#volume').value);
    synth.setReverb(+$('#reverb').value);
    led('#led-audio', 'on');
    drawScope();

    note.textContent = 'Cargando el reconocimiento de gestos…';
    const { FilesetResolver, HandLandmarker } = await import(`${MP_URL}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: HAND_TASK_URL, delegate },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      state.landmarker = await HandLandmarker.createFromOptions(fileset, options('GPU'));
      $('#delegate').textContent = 'GPU';
    } catch {
      state.landmarker = await HandLandmarker.createFromOptions(fileset, options('CPU'));
      $('#delegate').textContent = 'CPU';
    }
    state.connections = HandLandmarker.HAND_CONNECTIONS;

    note.textContent = 'Esperando permiso de la cámara…';
    await startCamera();
    led('#led-cam', 'on');

    $('#gate').hidden = true;
    state.running = true;
    scheduleNext();
  } catch (e) {
    console.error(e);
    btn.disabled = false;
    note.classList.add('error');
    note.textContent = e.name === 'NotAllowedError'
      ? 'Permiso de cámara denegado. Habilítalo en la barra de direcciones y vuelve a intentar.'
      : `No se pudo iniciar: ${e.message}. ¿Hay conexión a internet? (el detector se descarga la primera vez)`;
    led('#led-cam', 'err');
  }
}

// ---------------------------------------------------------------- cámara

const CAMERA_KEY = 'musics-hands.camera';
// Cámaras de teléfono enlazadas a Windows (Enlace Móvil, DroidCam, etc.) o virtuales
const PHONE_RE = /phone|tel[eé]fono|m[oó]vil|android|iphone|galaxy|pixel|link|droidcam|iriun|camo|virtual/i;

const savedCamera = () => { try { return localStorage.getItem(CAMERA_KEY); } catch { return null; } };
const saveCamera = (id) => { try { localStorage.setItem(CAMERA_KEY, id); } catch { /* sin almacenamiento */ } };

async function openCamera(deviceId) {
  video.srcObject?.getTracks().forEach((t) => t.stop());
  // MediaPipe reduce la imagen a 192–224 px: más resolución solo cuesta tiempo
  const constraints = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } };
  if (deviceId) constraints.deviceId = { exact: deviceId };
  const stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  state.lastVideoTime = -1;
  return stream.getVideoTracks()[0].getSettings().deviceId;
}

async function listCameras() {
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
}

async function startCamera() {
  const saved = savedCamera();
  let id;
  try {
    id = await openCamera(saved);
  } catch (e) {
    if (!saved) throw e;
    id = await openCamera(null); // la cámara guardada ya no está conectada
  }
  // Los nombres de las cámaras solo se conocen después de dar permiso
  const cams = await listCameras();
  const current = cams.find((c) => c.deviceId === id);
  if (!saved && current && PHONE_RE.test(current.label)) {
    const builtIn = cams.find((c) => !PHONE_RE.test(c.label));
    if (builtIn) id = await openCamera(builtIn.deviceId);
  }
  fillCameraSelect(cams, id);
}

function fillCameraSelect(cams, activeId) {
  const select = $('#camera');
  select.innerHTML = '';
  cams.forEach((c, i) => select.add(new Option(c.label || `Cámara ${i + 1}`, c.deviceId, false, c.deviceId === activeId)));
  select.disabled = cams.length < 2;
}

$('#camera').addEventListener('change', async (e) => {
  const id = e.target.value;
  try {
    await openCamera(id);
    saveCamera(id);
    led('#led-cam', 'on');
  } catch (err) {
    console.error(err);
    led('#led-cam', 'err');
  }
});

navigator.mediaDevices?.addEventListener('devicechange', async () => {
  if (!state.running) return;
  const active = video.srcObject?.getVideoTracks()[0]?.getSettings().deviceId;
  fillCameraSelect(await listCameras(), active);
});

// ---------------------------------------------------------------- bucle principal

// requestVideoFrameCallback se dispara justo cuando llega un cuadro nuevo de la cámara
// (rAF puede esperar hasta un refresco de pantalla extra)
const scheduleNext = 'requestVideoFrameCallback' in HTMLVideoElement.prototype
  ? () => video.requestVideoFrameCallback(loop)
  : () => requestAnimationFrame(loop);

function loop() {
  if (!state.running) return;
  scheduleNext();
  if (video.readyState < 2 || video.currentTime === state.lastVideoTime) return;
  state.lastVideoTime = video.currentTime;

  const t0 = performance.now();
  const result = state.landmarker.detectForVideo(video, t0);
  state.detMs = state.detMs * 0.9 + (performance.now() - t0) * 0.1;
  const hands = parseHands(result);
  updateMusic(hands); // primero el audio, después lo visual
  drawOverlay(hands);
  renderHands(hands);
  tickFps();
}

function parseHands(result) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const hands = result.landmarks.map((lms, i) => {
    const pts = normalizeLandmarks(lms, w, h);
    const counted = countFingers(pts);
    const label = result.handedness[i]?.[0]?.categoryName;
    return {
      lms,
      count: counted.count,
      fingers: counted.fingers,
      gesture: classifyGesture(counted),
      // MediaPipe asume imagen espejada; aquí el video entra sin espejar, así que se invierte
      side: label === 'Left' ? 'der' : 'izq',
      screenX: 1 - lms[0].x,
    };
  });
  // Si ambas se etiquetan igual, se decide por la posición en pantalla
  if (hands.length === 2 && hands[0].side === hands[1].side) {
    hands.sort((a, b) => a.screenX - b.screenX);
    hands[0].side = 'izq';
    hands[1].side = 'der';
  }
  return hands;
}

// ---------------------------------------------------------------- música

function chordForHand(hand) {
  return hand.gesture === null ? undefined : CHORD_KEYS[hand.gesture];
}

function updateMusic(hands) {
  if (state.mode === 'gesture') {
    for (const side of ['izq', 'der']) {
      const hand = hands.find((h) => h.side === side);
      if (!hand) { markMissing(side); continue; }
      if (hand.count === 0) { propose(side, 'mute', () => ({ notes: [], label: null })); continue; }
      const key = chordForHand(hand);
      if (!key) { slots[side].missing = 0; continue; }
      const shift = side === 'izq' ? -12 : 0;
      propose(side, key, () => ({ notes: CHORDS[key].notes.map((n) => n + shift), label: CHORDS[key].short }));
    }
  } else {
    if (!hands.length) { markMissing('total'); return; }
    const total = hands.reduce((a, h) => a + h.count, 0);
    propose('total', `t${total}`, () => {
      if (!total) return { notes: [], label: null };
      const n = SCALE[total - 1];
      return { notes: [n, n - 12], label: `${total} · ${noteName(n)}` };
    });
  }
}

function propose(slotName, key, make) {
  const s = slots[slotName];
  s.missing = 0;
  if (key === s.key) { s.candidate = null; s.n = 0; return; }
  if (key === s.candidate) s.n++;
  else { s.candidate = key; s.n = 1; }
  if (s.n < STABLE_FRAMES) return;
  const { notes, label } = make();
  Object.assign(s, { key, candidate: null, n: 0, notes, label });
  synth.play(slotName, notes);
  onChange();
}

function markMissing(slotName) {
  const s = slots[slotName];
  if (s.key === null || ++s.missing < LOST_FRAMES) return;
  Object.assign(s, newSlot());
  synth.release(slotName);
  onChange();
}

function resetSlots() {
  synth.releaseAll();
  for (const k of Object.keys(slots)) slots[k] = newSlot();
  renderReadout();
  renderPiano();
  renderLegend();
}

function onChange() {
  renderReadout(true);
  renderPiano();
  renderLegend();
}

// ---------------------------------------------------------------- render

function drawOverlay(hands) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const w = overlay.width;
  const h = overlay.height;
  for (const hand of hands) {
    const color = COLORS[hand.side];
    const P = hand.lms.map((p) => [p.x * w, p.y * h]);
    octx.lineWidth = 3;
    octx.strokeStyle = color;
    octx.globalAlpha = 0.75;
    octx.beginPath();
    for (const { start, end } of state.connections) {
      octx.moveTo(...P[start]);
      octx.lineTo(...P[end]);
    }
    octx.stroke();
    octx.globalAlpha = 1;
    P.forEach((p, i) => {
      const tip = TIPS.indexOf(i);
      const up = tip >= 0 && hand.fingers[tip];
      octx.beginPath();
      octx.arc(p[0], p[1], up ? 9 : tip >= 0 ? 6 : 3.5, 0, Math.PI * 2);
      octx.fillStyle = up ? color : '#ffffff';
      octx.fill();
      octx.lineWidth = 2;
      octx.strokeStyle = color;
      octx.stroke();
    });
  }
}

// Referencias cacheadas; solo se escribe en el DOM cuando el valor cambia
const handCards = Object.fromEntries([...document.querySelectorAll('.hand')].map((el) => [el.dataset.side, {
  el,
  state: el.querySelector('.hand-state'),
  count: el.querySelector('.hand-count'),
  fingers: [...el.querySelectorAll('.fingers li')],
  name: el.querySelector('.hand-gesture'),
  chord: el.querySelector('.hand-chord'),
}]));
const setText = (el, v) => { v = String(v); if (el.textContent !== v) el.textContent = v; };

function renderHands(hands) {
  for (const side of ['izq', 'der']) {
    const card = handCards[side];
    const hand = hands.find((h) => h.side === side);
    card.el.classList.toggle('idle', !hand);
    setText(card.state, hand ? 'detectada' : 'sin mano');
    setText(card.count, hand ? hand.count : 0);
    card.fingers.forEach((li, i) => li.classList.toggle('up', !!hand?.fingers[i]));
    setText(card.name, hand?.gesture != null ? CHORDS[CHORD_KEYS[hand.gesture]].gesture : hand ? 'Puño (silencio)' : '–');
    const slot = state.mode === 'gesture' ? slots[side] : null;
    setText(card.chord, slot?.label ?? '–');
  }
}

function renderReadout(pop = false) {
  const main = $('#readout-main');
  const sub = $('#readout-sub');
  const active = (state.mode === 'gesture' ? ['izq', 'der'] : ['total']).filter((k) => slots[k].label);
  if (!active.length) {
    main.textContent = '–';
    sub.textContent = 'Muestra una mano a la cámara';
  } else {
    main.innerHTML = active.map((k) => `<span class="${k}">${slots[k].label}</span>`).join('<span class="sep">+</span>');
    sub.textContent = active.flatMap((k) => slots[k].notes).sort((a, b) => a - b).map(noteName).join(' · ');
  }
  if (pop) { main.classList.remove('pop'); void main.offsetWidth; main.classList.add('pop'); }
}

function buildPiano() {
  const piano = $('#piano');
  const inner = document.createElement('div');
  inner.className = 'piano-inner';
  piano.append(inner);
  const whites = [];
  for (let m = PIANO_RANGE[0]; m <= PIANO_RANGE[1]; m++) if (![1, 3, 6, 8, 10].includes(m % 12)) whites.push(m);
  whites.forEach((m) => {
    const k = document.createElement('div');
    k.className = 'w';
    k.dataset.note = m;
    if (m % 12 === 0) k.innerHTML = `<span>${noteName(m)}</span>`;
    inner.append(k);
  });
  whites.forEach((m, i) => {
    if ([0, 2, 5, 7, 9].includes(m % 12) && m + 1 <= PIANO_RANGE[1]) {
      const k = document.createElement('div');
      k.className = 'b';
      k.dataset.note = m + 1;
      const width = 100 / whites.length;
      k.style.left = `calc(${(i + 1) * width}% - ${width * 0.3}%)`;
      k.style.width = `${width * 0.6}%`;
      inner.append(k);
    }
  });
}

function renderPiano() {
  document.querySelectorAll('#piano [data-note]').forEach((k) => k.classList.remove('on-izq', 'on-der', 'on-total'));
  for (const [name, s] of Object.entries(slots)) {
    for (const n of s.notes) document.querySelector(`#piano [data-note="${n}"]`)?.classList.add(`on-${name}`);
  }
}

function renderLegend() {
  const cells = state.mode === 'gesture'
    ? CHORD_KEYS.map((k) => ({
      big: CHORDS[k].short, small: CHORDS[k].gesture,
      on: ['izq', 'der'].filter((side) => slots[side].key === k),
    }))
    : SCALE.map((n, i) => ({
      big: NOTE_NAMES[n % 12], small: `${i + 1} dedo${i ? 's' : ''}`,
      on: slots.total.key === `t${i + 1}` ? ['total'] : [],
    }));
  $('#legend-title').textContent = state.mode === 'gesture' ? 'Gestos' : 'Dedos';
  $('#legend-cells').innerHTML = cells.map((c) =>
    `<li class="${c.on.map((s) => `on-${s}`).join(' ')}"><b>${c.big}</b><span>${c.small}</span></li>`).join('');
}

function tickFps() {
  state.frames++;
  const now = performance.now();
  if (now - state.fpsT >= 1000) {
    $('#fps').textContent = Math.round((state.frames * 1000) / (now - state.fpsT));
    $('#det').textContent = state.detMs.toFixed(1);
    state.frames = 0;
    state.fpsT = now;
  }
}

function drawScope() {
  const canvas = $('#scope');
  const ctx = canvas.getContext('2d');
  const data = new Uint8Array(synth.analyser.fftSize);
  const dpr = window.devicePixelRatio || 1;
  const frame = () => {
    // Redimensionar el canvas en cada cuadro realoca memoria; solo se hace si cambió el tamaño
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    else ctx.clearRect(0, 0, w, h);
    synth.analyser.getByteTimeDomainData(data);
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = COLORS.der;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = (v / 255) * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    requestAnimationFrame(frame);
  };
  frame();
}

function led(sel, cls, text) {
  const el = $(sel);
  el.classList.remove('on', 'warn', 'err');
  el.classList.add(cls);
  if (text) el.querySelector('span').textContent = text;
}

// ---------------------------------------------------------------- controles

function segmented(id, onPick) {
  const group = $(id);
  group.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    group.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    onPick(btn.dataset.value);
  });
}

segmented('#mode', (v) => { state.mode = v; resetSlots(); });
segmented('#instrument', (v) => {
  synth.setInstrument(v);
  for (const [name, s] of Object.entries(slots)) synth.play(name, s.notes); // re-suena con el nuevo timbre
});

const bindRange = (id, out, fn) => {
  const input = $(id);
  input.addEventListener('input', () => { $(out).textContent = Math.round(input.value * 100); fn(+input.value); });
};
bindRange('#volume', '#vol-out', (v) => synth.setVolume(v));
bindRange('#reverb', '#rev-out', (v) => synth.setReverb(v));

$('#panic').addEventListener('click', resetSlots);

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
}
$('#fullscreen').addEventListener('click', toggleFullscreen);

// Atajos para operar la demo en la TV
document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, select') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'f' || e.key === 'F') toggleFullscreen();
  else if (e.key === 'h' || e.key === 'H') $('#controls').classList.toggle('hidden');
  else if (e.key === ' ' && state.running) { e.preventDefault(); resetSlots(); }
});

buildPiano();
renderLegend();

// ?autostart enciende sin pulsar el botón (kiosco / pruebas); el audio puede requerir un clic
if (new URLSearchParams(location.search).has('autostart')) start();
