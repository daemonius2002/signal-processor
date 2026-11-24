/* Ascultă Universul - web version of Signal Processor. No dependencies. */
"use strict";

const DOWNLOAD_URL = "https://github.com/daemonius2002/signal-processor/releases/latest";
const COLORS = { pulsar: "#4dd0c4", star: "#f2b24a", variable: "#b79cf2", planet: "#7fb3ff", fiction: "#ff6b8b" };
const SKY = { bg: "#0a0f1a", grid: "#1b2636", line: "#32465f", star: "#e8eef7", text: "#8595a8", milky: "#27374d" };
const HEMISPHERE_SPAN = 115;
const BAND_CENTRES = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
  3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000];
const NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"];
const $ = (id) => document.getElementById(id);

const state = {
  catalog: null, byKey: {},
  view: "sky", skyView: "all", selected: null,
  ctx: null, gain: null, player: null, micStream: null, micNode: null, analyser: null,
  playing: true, volume: 0.6,
  source: null,          // {samples: Float32Array, fs, fft, viewHz, startedAt, live}
  loadToken: 0, blink: 0, specMode: "line", history: null, historyKey: "",
  solarDays: (Date.now() - Date.UTC(2000, 0, 1, 12)) / 86400000, solarSpeed: 0, lastFrame: performance.now(),
};

/* ------------------------------------------------------------------ DSP */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

const windows = {};
function hann(n) {
  if (!windows[n]) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
    windows[n] = w;
  }
  return windows[n];
}

/** Magnitude spectrum in dBFS (a full-scale sine reads 0 dB) and the dominant frequency. */
function analyse(frame, fs) {
  const n = frame.length, w = hann(n);
  const re = new Float32Array(n), im = new Float32Array(n);
  for (let i = 0; i < n; i++) re[i] = frame[i] * w[i];
  fft(re, im);
  const bins = n / 2, db = new Float32Array(bins), power = new Float32Array(bins);
  let best = 1, bestValue = -1;
  for (let k = 0; k < bins; k++) {
    const mag = Math.hypot(re[k], im[k]) * 4 / n;          // 2/N and the Hann coherent gain of 0.5
    db[k] = 20 * Math.log10(Math.max(mag, 1e-7));
    power[k] = mag * mag / 3;                                // mean square per bin: /2 for a sine, /1.5 for the Hann noise bandwidth
    if (k >= 2 && mag > bestValue) { bestValue = mag; best = k; }
  }
  let offset = 0;
  if (best > 1 && best < bins - 1) {                         // parabola through the three bins around the peak
    const a = db[best - 1], b = db[best], c = db[best + 1], curve = a - 2 * b + c;
    if (curve < 0) offset = Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / curve));
  }
  return { db, power, peakHz: (best + offset) * fs / n, peakDb: db[best], binHz: fs / n };
}

function noteName(hz) {
  if (!(hz >= 20 && hz <= 5000)) return "-";
  const midi = Math.round(69 + 12 * Math.log2(hz / 440));
  return `${NOTE_NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

function formatHz(hz) {
  if (!isFinite(hz)) return "-";
  if (hz >= 1000) return `${(hz / 1000).toFixed(3)} kHz`;
  return `${hz.toFixed(hz < 10 ? 4 : 2)} Hz`;
}

/* ---------------------------------------------------------------- audio */
function audio() {
  if (!state.ctx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    state.ctx = new Ctx();
    state.gain = state.ctx.createGain();
    state.gain.gain.value = state.volume;
    state.gain.connect(state.ctx.destination);
  }
  if (state.ctx.state === "suspended" && state.playing) state.ctx.resume();
  return state.ctx;
}

function stopPlayer() {
  if (state.player) { try { state.player.stop(); } catch (e) { /* already stopped */ } state.player.disconnect(); state.player = null; }
}

function stopMic() {
  if (state.micStream) state.micStream.getTracks().forEach((t) => t.stop());
  if (state.micNode) state.micNode.disconnect();
  state.micStream = state.micNode = state.analyser = null;
  $("btn-mic").textContent = "Pornește microfonul";
}

function playBuffer(buffer) {
  const ctx = audio();
  stopPlayer();
  const node = ctx.createBufferSource();
  node.buffer = buffer; node.loop = true;
  node.connect(state.gain); node.start();
  state.player = node;
  return ctx.currentTime;
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return audio().decodeAudioData(await response.arrayBuffer());
}

/* -------------------------------------------------------------- sources */
function showInfo(name, facts, story) {
  $("info-name").textContent = name; $("info-facts").textContent = facts; $("info-story").textContent = story;
}

async function selectObject(key) {
  const obj = state.byKey[key];
  if (!obj) return;
  const token = ++state.loadToken;
  state.selected = key; stopMic(); markChips();
  const where = obj.kind === "planet" ? "Sistemul Solar" : (obj.ra === null ? "pe cer se mișcă în fiecare zi" : `constelația ${obj.constellation}`);
  showInfo(obj.nick, `${obj.kindRo}, ${obj.name}, ${where}. ${obj.facts}`, obj.story);
  try {
    audio();
    const [raw, buffer] = await Promise.all([
      fetch(`data/analysis/${key}.i16`).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
      fetchBuffer(`data/audio/${key}.wav`),
    ]);
    if (token !== state.loadToken) return;                    // the user already picked something else
    const ints = new Int16Array(raw), samples = new Float32Array(ints.length);
    for (let i = 0; i < ints.length; i++) samples[i] = ints[i] / 32768;
    const startedAt = playBuffer(buffer);
    state.source = { samples, fs: obj.fs, fft: obj.fft, viewHz: Math.min(obj.viewHz, obj.fs / 2), startedAt, live: false };
  } catch (error) {
    showInfo(obj.nick, "Sunetul nu a putut fi încărcat.", String(error));
  }
}

async function selectSound(entry) {
  const token = ++state.loadToken;
  state.selected = `sound:${entry.key}`; stopMic(); markChips();
  showInfo(entry.title, `${entry.category}, ${Math.round(entry.seconds)} secunde. Sunet făcut pe calculator, nu înregistrat.`, entry.story);
  try {
    const buffer = await fetchBuffer(`data/sounds/${entry.file}`);
    if (token !== state.loadToken) return;
    const startedAt = playBuffer(buffer);
    state.source = { samples: buffer.getChannelData(0), fs: buffer.sampleRate, fft: 4096, viewHz: Math.min(entry.view_hz, buffer.sampleRate / 2), startedAt, live: false };
  } catch (error) {
    showInfo(entry.title, "Sunetul nu a putut fi încărcat.", String(error));
  }
}

async function toggleMic() {
  if (state.micStream) { stopMic(); state.source = null; return; }
  ++state.loadToken;
  try {
    const ctx = audio();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    stopPlayer();
    state.micStream = stream;
    state.micNode = ctx.createMediaStreamSource(stream);
    state.analyser = ctx.createAnalyser();
    state.analyser.fftSize = 4096;
    state.micNode.connect(state.analyser);                    // not connected to the loudspeakers: no feedback
    state.selected = "mic"; markChips();
    state.source = { samples: new Float32Array(4096), fs: ctx.sampleRate, fft: 4096, viewHz: 6000, startedAt: 0, live: true };
    $("btn-mic").textContent = "Oprește microfonul";
    showInfo("Microfonul tău", "În direct.", "Încearcă: un «aaa» lung, apoi un «iii». Fluieră. Bate din palme. Cine din familie are vocea cea mai groasă?");
  } catch (error) {
    $("mic-note").textContent = `Microfonul nu poate fi folosit: ${error.message}. Pe telefon, pagina trebuie deschisă prin https.`;
  }
}

function currentFrame() {
  const src = state.source;
  if (!src) return null;
  if (src.live) { state.analyser.getFloatTimeDomainData(src.samples); return src.samples; }
  const n = src.fft, total = src.samples.length, frame = new Float32Array(n);
  const elapsed = Math.max(0, state.ctx.currentTime - src.startedAt);
  let pos = Math.floor(elapsed * src.fs) % total;
  for (let i = n - 1; i >= 0; i--) { pos = pos === 0 ? total - 1 : pos - 1; frame[i] = src.samples[pos]; }
  return frame;
}

/* -------------------------------------------------------------- drawing */
function fit(canvas) {
  const ratio = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) { canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio); }
  const g = canvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { g, w, h };
}

function drawWave(frame, fs) {
  const { g, w, h } = fit($("wave"));
  g.fillStyle = "#161d27"; g.fillRect(0, 0, w, h);
  g.strokeStyle = SKY.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
  if (!frame) return;
  let peak = 0.05;
  for (let i = 0; i < frame.length; i++) peak = Math.max(peak, Math.abs(frame[i]));
  const scale = (h / 2 - 6) / peak, step = frame.length / w;
  g.strokeStyle = COLORS.pulsar; g.lineWidth = 1.2; g.beginPath();
  for (let x = 0; x < w; x++) {                               // min/max of the samples under each pixel
    let lo = 1e9, hi = -1e9;
    for (let i = Math.floor(x * step), end = Math.max(i + 1, Math.floor((x + 1) * step)); i < end && i < frame.length; i++) { lo = Math.min(lo, frame[i]); hi = Math.max(hi, frame[i]); }
    g.moveTo(x + 0.5, h / 2 - hi * scale); g.lineTo(x + 0.5, h / 2 - lo * scale + 1);
  }
  g.stroke();
  const seconds = frame.length / fs;
  $("wave-info").textContent = seconds >= 1 ? `ultimele ${seconds.toFixed(1)} secunde` : `ultimele ${(seconds * 1000).toFixed(0)} milisecunde`;
}

function drawSpectrum(result, viewHz) {
  const { g, w, h } = fit($("spec"));
  const left = 38, bottom = 22, pw = w - left - 6, ph = h - bottom - 8, top = 0, floor = -100;
  g.fillStyle = "#161d27"; g.fillRect(0, 0, w, h);
  g.font = "11px system-ui, sans-serif"; g.fillStyle = SKY.text; g.strokeStyle = SKY.grid; g.lineWidth = 1;
  for (let d = 0; d >= floor; d -= 20) {
    const y = 8 + ph * (top - d) / (top - floor);
    g.beginPath(); g.moveTo(left, y); g.lineTo(left + pw, y); g.stroke();
    g.textAlign = "right"; g.fillText(`${d} dB`, left - 4, y + 4);
  }
  const span = viewHz || 5000, rough = span / 8, pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const tick = [1, 2, 5, 10].map((m) => m * pow).find((v) => v >= rough);
  g.textAlign = "center";
  for (let f = 0; f <= span + 1e-9; f += tick) {
    const x = left + pw * f / span;
    g.beginPath(); g.moveTo(x, 8); g.lineTo(x, 8 + ph); g.stroke();
    g.fillText(f >= 1000 ? `${+(f / 1000).toFixed(2)} kHz` : `${+f.toFixed(2)} Hz`, x, h - 6);
  }
  if (!result) return;
  const yOf = (d) => 8 + ph * (top - Math.max(floor, Math.min(top, d))) / (top - floor);
  const lastBin = Math.min(result.db.length - 1, Math.ceil(span / result.binHz));
  g.strokeStyle = COLORS.star; g.lineWidth = 1.3; g.beginPath();
  if (lastBin > pw * 2) {                                     // more bins than pixels: draw the maximum per pixel
    for (let x = 0; x < pw; x++) {
      let hi = -1e9;
      for (let k = Math.floor(x * lastBin / pw), end = Math.max(k + 1, Math.floor((x + 1) * lastBin / pw)); k < end; k++) hi = Math.max(hi, result.db[k]);
      if (x === 0) g.moveTo(left, yOf(hi)); else g.lineTo(left + x, yOf(hi));
    }
  } else {
    for (let k = 0; k <= lastBin; k++) { const x = left + pw * (k * result.binHz) / span; if (k === 0) g.moveTo(x, yOf(result.db[k])); else g.lineTo(x, yOf(result.db[k])); }
  }
  g.stroke();
  if (result.peakDb > -70 && result.peakHz <= span) {
    const x = left + pw * result.peakHz / span, y = yOf(result.peakDb);
    g.fillStyle = COLORS.fiction; g.beginPath(); g.moveTo(x, y - 2); g.lineTo(x - 6, y - 13); g.lineTo(x + 6, y - 13); g.closePath(); g.fill();
    g.textAlign = x > w - 110 ? "right" : "left"; g.font = "600 12px system-ui, sans-serif";
    g.fillText(formatHz(result.peakHz), x + (x > w - 110 ? -10 : 10), Math.max(18, y - 8));
  }
}

/** 28 third-octave bands as LED columns (3 dB per LED), like a hardware real-time analyser. */
function drawBands(result) {
  const { g, w, h } = fit($("spec"));
  g.fillStyle = "#161d27"; g.fillRect(0, 0, w, h);
  const left = 38, bottom = 22, top = 6, ph = h - bottom - top, floor = -72, segs = 24, sh = ph / segs, cw = (w - left - 4) / BAND_CENTRES.length;
  g.font = "10px system-ui, sans-serif"; g.fillStyle = SKY.text; g.textAlign = "right";
  for (let d = 0; d >= floor; d -= 12) g.fillText(`${d} dB`, left - 4, top + ph * (-d) / (-floor) + 4);
  g.textAlign = "center";
  BAND_CENTRES.forEach((fc, i) => {
    let lit = 0;
    if (result && fc * 1.1225 < result.binHz * result.power.length) {
      const lo = fc / 1.1225 / result.binHz, hi = fc * 1.1225 / result.binHz;
      let sum = 0;
      for (let k = Math.floor(lo); k <= Math.ceil(hi) && k < result.power.length; k++) {
        const overlap = Math.min(hi, k + 0.5) - Math.max(lo, k - 0.5);        // share of this bin inside the band
        if (overlap > 0) sum += result.power[k] * overlap;
      }
      lit = Math.ceil((10 * Math.log10(Math.max(sum, 1e-12)) - floor) / 3);
    }
    for (let sgm = 0; sgm < segs; sgm++) {
      const dbTop = floor + (sgm + 1) * 3;
      g.fillStyle = sgm < lit ? (dbTop > -6 ? "#ff6b6b" : dbTop > -18 ? "#f0c04a" : "#57d08a") : "#1e2733";
      g.fillRect(left + i * cw + 1.5, top + ph - (sgm + 1) * sh + 1, cw - 3, sh - 2);
    }
    if (i % 3 === 0) { g.fillStyle = SKY.text; g.fillText(fc >= 1000 ? `${fc / 1000}k` : `${fc}`, left + (i + 0.5) * cw, h - 6); }
  });
}

/** Spectrogram: frequency across (logarithmic), time down, level as colour. */
function drawSpectrogram(result, viewHz) {
  const { g, w, h } = fit($("spec"));
  const cols = 320, rows = 200, left = 38, bottom = 22, top = 6;
  g.fillStyle = "#161d27"; g.fillRect(0, 0, w, h);
  if (!result) return;
  const fHi = Math.min(viewHz, 20000), fLo = fHi >= 2000 ? 20 : Math.max(2 * result.binHz, fHi / 200), key = `${fLo}:${fHi}:${result.binHz}`;
  if (!state.history || state.historyKey !== key) {
    state.history = document.createElement("canvas"); state.history.width = cols; state.history.height = rows; state.historyKey = key;
    const hg = state.history.getContext("2d"); hg.fillStyle = "#080a14"; hg.fillRect(0, 0, cols, rows);
  }
  const hg = state.history.getContext("2d");
  hg.drawImage(state.history, 0, -1);                          // scroll one row up
  const row = hg.createImageData(cols, 1), stops = [[8, 10, 20], [60, 20, 110], [190, 50, 90], [245, 150, 40], [255, 250, 200]];
  for (let x = 0; x < cols; x++) {
    const f = fLo * Math.pow(fHi / fLo, x / (cols - 1)), k = Math.min(result.db.length - 1, f / result.binHz), k0 = Math.floor(k);
    const d = result.db[k0] + (result.db[Math.min(k0 + 1, result.db.length - 1)] - result.db[k0]) * (k - k0);
    const t = Math.max(0, Math.min(0.9999, (d + 110) / 100)) * 4, i = Math.floor(t), u = t - i;
    for (let c = 0; c < 3; c++) row.data[x * 4 + c] = stops[i][c] + (stops[i + 1][c] - stops[i][c]) * u;
    row.data[x * 4 + 3] = 255;
  }
  hg.putImageData(row, 0, rows - 1);
  g.imageSmoothingEnabled = false;
  g.drawImage(state.history, left, top, w - left - 4, h - bottom - top);
  g.font = "10px system-ui, sans-serif"; g.fillStyle = SKY.text; g.textAlign = "center";
  for (const f of [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]) {
    if (f < fLo || f > fHi) continue;
    g.fillText(f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`, left + (w - left - 4) * Math.log(f / fLo) / Math.log(fHi / fLo), h - 6);
  }
  g.textAlign = "right"; g.fillText("acum", left - 4, h - bottom - 2); g.fillText("înainte", left - 2, top + 10);
}

function project(ra, dec, rect) {
  if (state.skyView === "all") return [rect.x + rect.w * (1 - ra / 24), rect.y + rect.h * (0.5 - dec / 180)];
  const fromPole = state.skyView === "north" ? 90 - dec : 90 + dec;
  if (fromPole > HEMISPHERE_SPAN) return null;
  const r = 0.5 * rect.w * fromPole / HEMISPHERE_SPAN, a = ra * 15 * Math.PI / 180, dir = state.skyView === "north" ? -1 : 1;
  return [rect.x + rect.w / 2 + r * Math.cos(a), rect.y + rect.h / 2 + dir * r * Math.sin(a)];
}

function skyRect(w, h) {
  if (state.skyView === "all") return { x: 6, y: 4, w: w - 12, h: h - 22 };
  const side = Math.min(w - 12, h - 8);
  return { x: (w - side) / 2, y: (h - side) / 2, w: side, h: side };
}

function drawSky() {
  const canvas = $("sky");
  canvas.classList.toggle("tall-map", state.skyView !== "all");
  const { g, w, h } = fit(canvas), rect = skyRect(w, h), whole = state.skyView === "all", cat = state.catalog;
  g.fillStyle = SKY.bg; g.fillRect(0, 0, w, h);
  g.save(); g.beginPath();
  if (whole) g.rect(rect.x, rect.y, rect.w, rect.h); else g.arc(rect.x + rect.w / 2, rect.y + rect.h / 2, rect.w / 2, 0, 2 * Math.PI);
  g.clip();
  g.strokeStyle = SKY.milky; g.lineWidth = 14; g.lineCap = g.lineJoin = "round"; g.globalAlpha = 0.55; g.beginPath();
  let prev = null;
  for (const [ra, dec] of cat.milkyWay) {
    const p = project(ra, dec, rect);
    if (!p) { prev = null; continue; }
    if (!prev || Math.abs(p[0] - prev[0]) > rect.w / 2) g.moveTo(p[0], p[1]); else g.lineTo(p[0], p[1]);
    prev = p;
  }
  g.stroke(); g.restore();

  g.lineWidth = 1; g.strokeStyle = SKY.grid; g.fillStyle = SKY.text; g.font = "10px system-ui, sans-serif"; g.textAlign = "center";
  if (whole) {
    for (let hour = 0; hour <= 24; hour += 3) { const x = project(hour, 0, rect)[0]; g.beginPath(); g.moveTo(x, rect.y); g.lineTo(x, rect.y + rect.h); g.stroke(); if (hour < 24) g.fillText(`${hour}h`, x, rect.y + rect.h + 13); }
    for (const d of [-60, -30, 0, 30, 60]) { const y = project(0, d, rect)[1]; g.beginPath(); g.moveTo(rect.x, y); g.lineTo(rect.x + rect.w, y); g.stroke(); }
    g.strokeStyle = SKY.line; g.strokeRect(rect.x, rect.y, rect.w, rect.h);
  } else {
    const sign = state.skyView === "north" ? 1 : -1, cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
    for (const d of [60, 30, 0]) { g.beginPath(); g.arc(cx, cy, 0.5 * rect.w * (90 - d) / HEMISPHERE_SPAN, 0, 2 * Math.PI); g.stroke(); }
    for (let hour = 0; hour < 24; hour += 3) {
      const edge = project(hour, sign * (90 - HEMISPHERE_SPAN), rect), label = project(hour, sign * (90 - HEMISPHERE_SPAN + 7), rect);
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(edge[0], edge[1]); g.stroke(); g.fillText(`${hour}h`, label[0], label[1] + 4);
    }
    g.strokeStyle = SKY.line; g.beginPath(); g.arc(cx, cy, rect.w / 2, 0, 2 * Math.PI); g.stroke();
  }
  const pos = {};
  for (const [name, ra, dec] of cat.stars) pos[name] = [ra, dec];
  g.strokeStyle = SKY.line; g.lineWidth = 1.1;
  for (const [, , names] of cat.lines) {
    for (let i = 1; i < names.length; i++) {
      const a = project(pos[names[i - 1]][0], pos[names[i - 1]][1], rect), b = project(pos[names[i]][0], pos[names[i]][1], rect);
      if (a && b && (!whole || Math.abs(a[0] - b[0]) < rect.w / 2)) { g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
    }
  }
  g.fillStyle = "#5f7896";
  for (const [label, at] of cat.lines) { if (at[0] || at[1]) { const p = project(at[0], at[1], rect); if (p) g.fillText(label, p[0], p[1] + 3); } }
  for (const [name, ra, dec, mag] of cat.stars) {
    const p = project(ra, dec, rect);
    if (!p) continue;
    g.fillStyle = SKY.star; g.beginPath(); g.arc(p[0], p[1], Math.max(0.9, 3.1 - 0.62 * mag), 0, 2 * Math.PI); g.fill();
    if (cat.labelled.includes(name) && w > 420) {
      g.fillStyle = SKY.text; const tw = g.measureText(name).width, right = p[0] + 8 + tw < rect.x + rect.w;
      g.textAlign = "left"; g.fillText(name, right ? p[0] + 6 : p[0] - 6 - tw, p[1] + 4); g.textAlign = "center";
    }
  }
  for (const obj of cat.objects) {
    if (obj.ra === null || obj.kind === "planet") continue;
    const p = project(obj.ra, obj.dec, rect);
    if (!p) continue;
    const color = COLORS[obj.kind], on = obj.key === state.selected;
    if (on) { const grow = state.blink; g.globalAlpha = 1 - grow; g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); g.arc(p[0], p[1], 7 + 16 * grow, 0, 2 * Math.PI); g.stroke(); g.globalAlpha = 1; }
    g.strokeStyle = color; g.lineWidth = 2; g.fillStyle = on ? color : SKY.bg; g.beginPath(); g.arc(p[0], p[1], 5.5, 0, 2 * Math.PI); g.fill(); g.stroke();
    if (on) {
      g.font = "600 12px system-ui, sans-serif"; const tw = g.measureText(obj.nick).width + 10;
      const x = Math.max(4, Math.min(w - tw - 4, p[0] - tw / 2)), y = p[1] > 40 ? p[1] - 28 : p[1] + 12;
      g.fillStyle = "rgba(10,15,26,.9)"; g.fillRect(x, y, tw, 18); g.fillStyle = color; g.textAlign = "left"; g.fillText(obj.nick, x + 5, y + 13); g.textAlign = "center"; g.font = "10px system-ui, sans-serif";
    }
  }
}

function orbitRadius(au, w, h) {
  const limit = 0.5 * Math.min(w - 50, h) - 18;
  return 24 + (limit - 24) * Math.log1p(au / 0.3) / Math.log1p(30.07 / 0.3);
}

function planetPosition(key, w, h) {
  const [au, lon0, rate] = state.catalog.orbits[key];
  const a = ((lon0 + rate * state.solarDays) % 360) * Math.PI / 180, r = orbitRadius(au, w, h);
  return [w / 2 + r * Math.cos(a), h / 2 - r * Math.sin(a)];
}

function drawSolar() {
  const { g, w, h } = fit($("solar"));
  g.fillStyle = SKY.bg; g.fillRect(0, 0, w, h);
  g.strokeStyle = SKY.grid; g.lineWidth = 1;
  for (const key in state.catalog.orbits) { g.beginPath(); g.arc(w / 2, h / 2, orbitRadius(state.catalog.orbits[key][0], w, h), 0, 2 * Math.PI); g.stroke(); }
  const glow = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, 22);
  glow.addColorStop(0, "#fff3c4"); glow.addColorStop(0.45, COLORS.star); glow.addColorStop(1, "rgba(242,178,74,0)");
  g.fillStyle = glow; g.beginPath(); g.arc(w / 2, h / 2, 22, 0, 2 * Math.PI); g.fill();
  if (state.selected === "sun") { g.strokeStyle = COLORS.star; g.lineWidth = 2; g.globalAlpha = 1 - state.blink; g.beginPath(); g.arc(w / 2, h / 2, 16 + 14 * state.blink, 0, 2 * Math.PI); g.stroke(); g.globalAlpha = 1; }
  g.font = "11px system-ui, sans-serif"; g.textAlign = "center";
  for (const obj of state.catalog.objects) {
    if (obj.kind !== "planet") continue;
    const [, , , color, size] = state.catalog.orbits[obj.key], [x, y] = planetPosition(obj.key, w, h), on = obj.key === state.selected;
    if (obj.key === "saturn") { g.strokeStyle = color; g.lineWidth = 1.4; g.beginPath(); g.ellipse(x, y, size * 1.9, size * 0.6, 0, 0, 2 * Math.PI); g.stroke(); }
    g.fillStyle = color; g.beginPath(); g.arc(x, y, size, 0, 2 * Math.PI); g.fill();
    if (on) { g.strokeStyle = color; g.lineWidth = 2; g.globalAlpha = 1 - state.blink; g.beginPath(); g.arc(x, y, size + 4 + 14 * state.blink, 0, 2 * Math.PI); g.stroke(); g.globalAlpha = 1; }
    g.fillStyle = on ? "#fff" : SKY.text; g.fillText(obj.nick, x, y + size + 13);
  }
  const date = new Date(Date.UTC(2000, 0, 1, 12) + state.solarDays * 86400000);
  $("solar-date").textContent = `Data: ${date.toLocaleDateString("ro-RO")}`;
}

/* ------------------------------------------------------------ interface */
function markChips() {
  document.querySelectorAll(".chip[data-key]").forEach((chip) => chip.setAttribute("aria-pressed", String(chip.dataset.key === state.selected)));
}

function canvasPoint(canvas, event) {
  const box = canvas.getBoundingClientRect();
  return [event.clientX - box.left, event.clientY - box.top, box.width, box.height];
}

function buildInterface() {
  const cat = state.catalog;
  const list = $("sky-list");
  for (const obj of cat.objects) {
    if (obj.kind === "planet") continue;
    const chip = document.createElement("button");
    chip.type = "button"; chip.className = `chip ${obj.kind}`; chip.dataset.key = obj.key; chip.textContent = obj.nick; chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => selectObject(obj.key));
    list.appendChild(chip);
  }
  const sounds = $("sound-list");
  let category = null, box = null;
  for (const entry of cat.sounds) {
    if (entry.category !== category) {
      category = entry.category;
      const head = document.createElement("h3"); head.textContent = category; sounds.appendChild(head);
      box = document.createElement("div"); box.className = "chips"; box.style.maxHeight = "none"; sounds.appendChild(box);
    }
    const chip = document.createElement("button");
    chip.type = "button"; chip.className = "chip"; chip.dataset.key = `sound:${entry.key}`; chip.textContent = entry.title; chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => selectSound(entry));
    box.appendChild(chip);
  }
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    state.view = tab.dataset.view;
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    for (const view of ["sky", "solar", "sounds", "mic"]) $(`view-${view}`).hidden = view !== state.view;
  }));
  $("sky-view").addEventListener("change", (e) => { state.skyView = e.target.value; });
  $("sky").addEventListener("click", (event) => {
    const [x, y, w, h] = canvasPoint($("sky"), event), rect = skyRect(w, h);
    let best = null, distance = 18;
    for (const obj of cat.objects) {
      if (obj.ra === null || obj.kind === "planet") continue;
      const p = project(obj.ra, obj.dec, rect);
      if (p && Math.hypot(p[0] - x, p[1] - y) < distance) { distance = Math.hypot(p[0] - x, p[1] - y); best = obj.key; }
    }
    if (best) selectObject(best);
  });
  $("solar").addEventListener("click", (event) => {
    const [x, y, w, h] = canvasPoint($("solar"), event);
    if (Math.hypot(x - w / 2, y - h / 2) < 18) { selectObject("sun"); return; }
    let best = null, distance = 20;
    for (const key in cat.orbits) { const p = planetPosition(key, w, h), d = Math.hypot(p[0] - x, p[1] - y); if (d < distance) { distance = d; best = key; } }
    if (best) selectObject(best);
  });
  const speed = () => { const v = +$("solar-speed").value; state.solarSpeed = v === 0 ? 0 : 0.5 * Math.pow(2000, v / 100); };
  $("solar-speed").addEventListener("input", speed); speed();
  $("volume").addEventListener("input", (e) => { state.volume = e.target.value / 100; if (state.gain) state.gain.gain.value = state.volume; });
  $("btn-play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("btn-play").textContent = state.playing ? "Pauză" : "Pornește";
    $("btn-play").setAttribute("aria-pressed", String(state.playing));
    if (state.ctx) { if (state.playing) state.ctx.resume(); else state.ctx.suspend(); }
  });
  $("btn-mic").addEventListener("click", toggleMic);
  document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => {
    state.specMode = button.dataset.mode; state.history = null;
    document.querySelectorAll(".mode").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  }));
  $("download").href = DOWNLOAD_URL;
}

function loop(now) {
  const dt = Math.min(0.1, (now - state.lastFrame) / 1000);
  state.lastFrame = now;
  state.blink = (now / 1000) % 1;
  if (state.playing) state.solarDays += state.solarSpeed * dt;
  if (state.view === "sky") drawSky();
  if (state.view === "solar") drawSolar();
  const src = state.source;
  if (src && state.ctx) {
    const frame = currentFrame(), result = analyse(frame, src.fs);
    drawWave(frame, src.fs);
    if (state.specMode === "bands") drawBands(result); else if (state.specMode === "time") drawSpectrogram(result, src.viewHz); else drawSpectrum(result, src.viewHz);
    const loud = result.peakDb > -70;
    $("r-freq").textContent = loud ? formatHz(result.peakHz) : "-";
    $("r-level").textContent = `${result.peakDb.toFixed(1)} dB`;
    $("r-note").textContent = loud ? noteName(result.peakHz) : "-";
    $("spec-info").textContent = `un pas = ${formatHz(result.binHz)}`;
  } else {
    drawWave(null, 1); drawSpectrum(null, 5000);
  }
  requestAnimationFrame(loop);
}

if (typeof module !== "undefined") module.exports = { fft, analyse, noteName, formatHz };   // for tests under Node

if (typeof document !== "undefined") {
  fetch("data/catalog.json").then((r) => r.json()).then((catalog) => {
    state.catalog = catalog;
    for (const obj of catalog.objects) state.byKey[obj.key] = obj;
    buildInterface();
    // Browsers only start sound after a tap or click: wake the audio on the first one.
    document.addEventListener("pointerdown", () => { if (state.ctx && state.playing && state.ctx.state === "suspended") state.ctx.resume(); });
    const mode = document.querySelector(`.mode[data-mode="${new URLSearchParams(location.search).get("mode")}"]`);
    if (mode) mode.click();                                                // .../?mode=bands or ?mode=time
    const wanted = decodeURIComponent(location.hash.slice(1));          // link straight to an object: .../#crab
    if (state.byKey[wanted]) {
      if (state.byKey[wanted].kind === "planet" || wanted === "sun") document.querySelector('.tab[data-view="solar"]').click();
      selectObject(wanted);
    }
    requestAnimationFrame(loop);
  }).catch((error) => showInfo("Datele nu au putut fi încărcate", String(error), "Pagina trebuie deschisă printr-un server web, nu direct din fișier."));
}
