export const OUTPUT_RATE = 16_000;
export const CHUNK_SAMPLES = 1_600;
export const SPEECH_THRESHOLD = 0.015;
export const SILENCE_THRESHOLD = 0.008;
export const MIN_SPEECH_SAMPLES = 8_000;
export const MAX_BUFFER_SAMPLES = 48_000;
export const SILENCE_CHUNKS_TO_FLUSH = 4;
export const RTT_SAMPLE_LIMIT = 720;
export const MAX_CANVAS_DPR = 2;
export const MAX_CANVAS_WIDTH = 4096;
export const MAX_CANVAS_HEIGHT = 2048;

export function canvasBackingSize(cssWidth, cssHeight, devicePixelRatio = 1) {
  if (![cssWidth, cssHeight, devicePixelRatio].every(Number.isFinite) || cssWidth <= 0 || cssHeight <= 0) {
    return null;
  }
  const dpr = Math.min(MAX_CANVAS_DPR, Math.max(1, devicePixelRatio));
  return {
    width: Math.max(1, Math.min(MAX_CANVAS_WIDTH, Math.floor(cssWidth * dpr))),
    height: Math.max(1, Math.min(MAX_CANVAS_HEIGHT, Math.floor(cssHeight * dpr))),
    dpr,
  };
}

export function shouldDrawCanvas(hidden, cssWidth, cssHeight) {
  return hidden !== true && Number.isFinite(cssWidth) && Number.isFinite(cssHeight)
    && cssWidth > 0 && cssHeight > 0;
}

export const METRIC_DEFINITIONS = Object.freeze({
  AUDIO_DURATION_MS: "Duración del PCM enviado al modelo; no es latencia y puede incluir silencio final.",
  SERVER_ENDPOINTING_MS: "perf_counter del servidor: último chunk clasificado como voz hasta entrega del job al broker; incluye cierre por VAD y scheduling previo a la cola.",
  SERVER_QUEUE_WAIT_MS: "perf_counter del servidor: entrega al broker hasta el inicio real de backend.transcribe().",
  SERVER_MODEL_INFERENCE_MS: "perf_counter del servidor alrededor de backend.transcribe(); wall time del adaptador, no tiempo de kernel GPU.",
  SERVER_POSTPROCESS_MS: "perf_counter del servidor: retorno del adaptador hasta transcript listo para send_json; no incluye envío por red.",
  SERVER_EOS_TO_TRANSCRIPT_MS: "perf_counter del servidor: último chunk clasificado como voz hasta transcript listo para send_json; no incluye navegador ni red.",
  CLIENT_EOS_TO_TRANSCRIPT_MS: "performance.now del navegador: último chunk que el shadow VAD cliente clasifica como voz hasta recepción del transcript; incluye endpointing aproximado, navegador, red y servidor.",
  PROXY_WS_RTT_MS: "RTT completo de benchmark_ping/pong por el WebSocket autenticado; incluye Gateway/nginx, red y scheduling. No es latencia unidireccional.",
  SERVER_RECEIVE_TO_TRANSCRIPT_MS: "Diagnóstico heredado desde el inicio del buffer; puede incluir silencio inicial y duración del segmento hablado. No es KPI de latencia ASR.",
  SEGMENT_WAIT_MS: "Alias heredado: último chunk con voz hasta decisión del cierre/flush, antes de la entrega al broker.",
  MODEL_INFERENCE_MS: "Alias de SERVER_MODEL_INFERENCE_MS.",
  queue_wait_ms: "Alias de SERVER_QUEUE_WAIT_MS.",
  SERVER_AUDIO_END_TO_TRANSCRIPT_MS: "Alias de SERVER_EOS_TO_TRANSCRIPT_MS.",
  CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: "Alias de CLIENT_EOS_TO_TRANSCRIPT_MS.",
});

export function boundedPush(items, value, limit = RTT_SAMPLE_LIMIT) {
  items.push(value);
  if (items.length > limit) items.splice(0, items.length - limit);
  return items;
}

export function rttPercentiles(samples) {
  const values = samples.map((sample) => sample.PROXY_WS_RTT_MS).filter(Number.isFinite);
  return { p50: median(values), p95: percentile(values, 0.95) };
}

export function formatMilliseconds(value) {
  return Number.isFinite(value) ? `${Number(value).toFixed(2)} ms` : "—";
}

export function gpuStatusLabel(compute, telemetry) {
  if (compute?.available === true || compute?.cuda === true) {
    return compute.device || "CUDA disponible";
  }
  if (telemetry?.device) return `${telemetry.device} · cómputo CUDA no confirmado`;
  return "GPU de cómputo no confirmada";
}

export function qwenDeviceTelemetryLabel(point = {}) {
  const label = gpuStatusLabel(point.gpu_compute, { device: point.gpu_device });
  const value = (number) => Number.isFinite(number) ? number : "—";
  return `${label} · ${value(point.vram_used_mib)}/${value(point.vram_total_mib)} MiB · ${value(point.process_rss_mib)} MiB RAM`;
}

const TERMINAL_QWEN_ERRORS = new Set([
  "stream_duration_limit", "stream_not_started", "stream_worker_failed",
  "stream_worker_timeout", "stream_scheduler_overrun", "stream_result_queue_full",
  "invalid_worker_response", "stream_fenced", "stream_terminal",
]);

export function isTerminalQwenError(code) {
  return TERMINAL_QWEN_ERRORS.has(code);
}

export function floatToPcm16LE(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, Number.isFinite(samples[i]) ? samples[i] : 0));
    const quantized = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
    view.setInt16(i * 2, quantized, true);
  }
  return buffer;
}

export function pcm16RmsNormalized(buffer) {
  const view = new DataView(buffer);
  const count = Math.floor(buffer.byteLength / 2);
  if (!count) return 0;
  let sum = 0;
  for (let i = 0; i < count; i += 1) {
    const sample = view.getInt16(i * 2, true);
    sum += sample * sample;
  }
  return Math.sqrt(sum / count) / 32768;
}

// Bounded streaming windowed-sinc resampler. Web Audio may already perform
// device-to-context conversion; this covers any rate the AudioWorklet receives.
export class StreamingResampler {
  constructor(inputRate, outputRate = OUTPUT_RATE, radius = 16) {
    if (!Number.isFinite(inputRate) || inputRate < 1 || !Number.isFinite(outputRate) || outputRate < 1) {
      throw new RangeError("sample rates must be positive");
    }
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.radius = radius;
    this.step = inputRate / outputRate;
    this.cutoff = 0.47 * Math.min(1, outputRate / inputRate);
    this.samples = [];
    this.bufferStart = 0;
    this.inputCount = 0;
    this.outputCount = 0;
    this.nextPosition = 0;
    this.firstSample = 0;
    this.lastSample = 0;
    this.finished = false;
  }

  push(input) {
    if (this.finished) throw new Error("resampler is already finished");
    if (input.length && this.inputCount === 0) this.firstSample = input[0];
    for (let i = 0; i < input.length; i += 1) {
      const sample = Number.isFinite(input[i]) ? input[i] : 0;
      this.samples.push(sample);
      this.lastSample = sample;
    }
    this.inputCount += input.length;
    return this._produce(false);
  }

  finish() {
    if (this.finished) return new Float32Array(0);
    this.finished = true;
    return this._produce(true);
  }

  _at(index) {
    if (index < 0) return this.firstSample;
    if (index >= this.inputCount) return this.lastSample;
    const local = index - this.bufferStart;
    return local >= 0 && local < this.samples.length ? this.samples[local] : this.firstSample;
  }

  _sample(position) {
    const center = Math.floor(position);
    const start = center - this.radius + 1;
    const end = center + this.radius;
    let weighted = 0;
    let weightSum = 0;
    for (let index = start; index <= end; index += 1) {
      const distance = index - position;
      const normalized = distance / this.radius;
      if (Math.abs(normalized) >= 1) continue;
      const window = 0.42 + 0.5 * Math.cos(Math.PI * normalized) + 0.08 * Math.cos(2 * Math.PI * normalized);
      const x = 2 * this.cutoff * distance;
      const sinc = Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const weight = 2 * this.cutoff * sinc * window;
      weighted += this._at(index) * weight;
      weightSum += weight;
    }
    return weightSum ? weighted / weightSum : this._at(center);
  }

  _produce(flushing) {
    if (this.inputRate === this.outputRate) {
      if (!this.inputCount) return new Float32Array(0);
      const copy = Float32Array.from(this.samples);
      this.samples = [];
      this.bufferStart = this.inputCount;
      this.nextPosition = this.inputCount;
      return copy;
    }
    const output = [];
    const limit = flushing ? this.inputCount : this.inputCount - this.radius;
    const finalOutputCount = Math.ceil(this.inputCount / this.step - 1e-9);
    while (this.nextPosition < limit && (!flushing || this.outputCount + output.length < finalOutputCount)) {
      output.push(this._sample(this.nextPosition));
      this.nextPosition += this.step;
    }
    const keepFrom = Math.max(0, Math.floor(this.nextPosition) - this.radius - 1);
    const discard = Math.max(0, keepFrom - this.bufferStart);
    if (discard) {
      this.samples.splice(0, discard);
      this.bufferStart += discard;
    }
    this.outputCount += output.length;
    return Float32Array.from(output);
  }
}

export class ShadowVad {
  constructor() { this.reset(); }

  push(pcmBuffer, nowMs) {
    const count = Math.floor(pcmBuffer.byteLength / 2);
    const rms = pcm16RmsNormalized(pcmBuffer);
    if (rms >= SPEECH_THRESHOLD) {
      this.speaking = true;
      this.silenceStreak = 0;
      this.samples += count;
      this.lastVoiceAt = nowMs;
    } else if (rms < SILENCE_THRESHOLD) {
      if (this.speaking) {
        this.silenceStreak += 1;
        this.samples += count;
      }
    } else if (this.speaking) {
      this.samples += count;
    }

    const shouldFlush = (this.speaking && this.silenceStreak >= SILENCE_CHUNKS_TO_FLUSH)
      || this.samples >= MAX_BUFFER_SAMPLES;
    const result = {
      rms,
      inSegment: this.speaking,
      closed: shouldFlush,
      canTranscribe: this.speaking && this.samples >= MIN_SPEECH_SAMPLES,
      sampleCount: this.samples,
      lastVoiceAt: this.lastVoiceAt,
    };
    if (shouldFlush) this.reset();
    return result;
  }

  reset() {
    this.speaking = false;
    this.silenceStreak = 0;
    this.samples = 0;
    this.lastVoiceAt = null;
  }
}

export function percentile(values, fraction) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const index = Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1));
  return sorted[index];
}

export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function normalizeWords(text) {
  return String(text).toLocaleLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
}

export function wordErrorRate(reference, hypothesis) {
  const ref = normalizeWords(reference);
  const hyp = normalizeWords(hypothesis);
  if (!ref.length) return null;
  let row = Array.from({ length: hyp.length + 1 }, (_, index) => index);
  for (let i = 1; i <= ref.length; i += 1) {
    const next = [i];
    for (let j = 1; j <= hyp.length; j += 1) {
      next[j] = Math.min(next[j - 1] + 1, row[j] + 1, row[j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
    }
    row = next;
  }
  return row[hyp.length] / ref.length;
}

export function csvCell(value) {
  let text = value == null ? "" : String(value);
  // Avoid spreadsheet formula execution when a transcript is opened as CSV.
  if (typeof value === "string" && /^[\s\u0000-\u001f]*[=+\-@]/.test(text)) text = `'${text}`;
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
