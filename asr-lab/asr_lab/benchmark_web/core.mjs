export const OUTPUT_RATE = 16_000;
export const AUDIO_PUSH_INTERVAL_MS = 100;
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
  FIRST_STREAM_INIT_MS: "Wall time del RPC init de estado Qwen, excluye registro en scheduler.",
  FIRST_STREAM_STATE_INIT_WALL_MS: "Wall time medido alrededor de model.init_streaming_state dentro del worker.",
  FIRST_STREAM_INIT_RPC_OVERHEAD_MS: "Diferencia no negativa entre RPC parent y init del worker; nula si falta medición interna.",
  FIRST_AUDIO_TO_FIRST_DECODE_READY_MS: "Reloj monotónico servidor: primer PCM aceptado hasta job real listo para despacho.",
  FIRST_SCHEDULER_WAIT_MS: "Reloj monotónico servidor: job listo hasta dispatch del primer decode con step real.",
  FIRST_AUDIO_TO_FIRST_DECODE_START_MS: "Reloj monotónico servidor: primer PCM aceptado hasta dispatch del primer decode con step real.",
  SERVER_FIRST_PARTIAL_MS: "Primer PCM aceptado por servidor hasta el primer partial candidate no vacío.",
  CLIENT_FIRST_PARTIAL_MS: "Primer envío real de audio del navegador/CLI hasta recepción del primer partial candidate no vacío.",
  EFFECTIVE_MAX_BACKLOG_MS: "Límite efectivo configurado: max_backlog_chunks multiplicado por model_chunk_ms; no se autoajusta.",
  QWEN_DECODE_SLO_TARGET_MS: "Objetivo absoluto de wall time por llamada Qwen: menos de 100 ms.",
  QWEN_DECODE_SLO_VIOLATION: "Verdadero si un decode real tarda 100 ms o más, independiente de la ventana de audio.",
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

export function decodeSloLabel(violation) {
  if (violation === true) return "violation";
  if (violation === false) return "within target";
  return "unmeasured";
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

export function clientFirstPartialMs(firstAudioSentAt, receivedAt) {
  if (![firstAudioSentAt, receivedAt].every(Number.isFinite)) return null;
  return Math.max(0, receivedAt - firstAudioSentAt);
}

const STARTUP_VERSION_FIELDS = [
  "qwen_asr_version", "vllm_version", "transformers_version", "torch_version", "torch_cuda_version",
];
const STARTUP_EXPERIMENT_FIELDS = [
  "gpu_memory_utilization", "max_active_sessions", "max_num_seqs", "max_new_tokens",
  "max_model_len", "warmup_chunk_ms", "unfixed_chunk_num", "unfixed_token_num",
];

function safeStartupScalar(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string" && value.length <= 160) return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return null;
}

function safeWorkerStartupMetric(worker, key) {
  if (!worker || typeof worker !== "object") return null;
  const value = key === "warmup_chunk_ms"
    ? worker.experiment_config?.warmup_chunk_ms ?? worker.warmup_chunk_ms
    : worker[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function safeQwenStartupEvidence({ readinessAtStart, qwenStartMetrics, health } = {}) {
  const readiness = readinessAtStart && typeof readinessAtStart === "object" ? readinessAtStart : {};
  const start = qwenStartMetrics && typeof qwenStartMetrics === "object" ? qwenStartMetrics : {};
  const snapshot = health && typeof health === "object" ? health : {};
  const workers = Array.isArray(snapshot.worker_metrics) ? snapshot.worker_metrics : [];
  const provenance = snapshot.runtime_provenance && typeof snapshot.runtime_provenance === "object"
    ? snapshot.runtime_provenance : {};
  const runtimeProvenance = Object.fromEntries(STARTUP_VERSION_FIELDS.map((key) => [key, safeStartupScalar(provenance[key])]));
  const experiment = provenance.experiment_config && typeof provenance.experiment_config === "object"
    ? provenance.experiment_config : {};
  runtimeProvenance.experiment_config = Object.fromEntries(
    STARTUP_EXPERIMENT_FIELDS.map((key) => [key, safeStartupScalar(experiment[key])]),
  );
  return {
    readiness_at_start: {
      ready: typeof readiness.ready === "boolean" ? readiness.ready : null,
      backend: typeof readiness.backend === "string" ? readiness.backend : null,
      production_backend: typeof readiness.production_backend === "string" ? readiness.production_backend : null,
      model_loaded: typeof readiness.model_loaded === "boolean" ? readiness.model_loaded : null,
    },
    FIRST_STREAM_INIT_MS: safeStartupScalar(start.FIRST_STREAM_INIT_MS),
    FIRST_STREAM_STATE_INIT_WALL_MS: safeStartupScalar(start.FIRST_STREAM_STATE_INIT_WALL_MS),
    FIRST_STREAM_INIT_RPC_OVERHEAD_MS: safeStartupScalar(start.FIRST_STREAM_INIT_RPC_OVERHEAD_MS),
    model_id: typeof snapshot.model_id === "string" ? snapshot.model_id : null,
    model_revision: typeof snapshot.model_revision === "string" ? snapshot.model_revision : null,
    model_load_ms: workers.map((worker) => safeWorkerStartupMetric(worker, "model_load_ms")),
    warmup_ms: workers.map((worker) => safeWorkerStartupMetric(worker, "warmup_ms")),
    warmup_chunk_ms: workers.map((worker) => safeWorkerStartupMetric(worker, "warmup_chunk_ms")),
    workers: Number.isInteger(snapshot.workers) ? snapshot.workers : null,
    runtime_provenance: runtimeProvenance,
  };
}

const FORBIDDEN_EPOCH_SNAPSHOT_KEYS = new Set([
  "text", "transcript", "pcm16le", "pcm16le_base64", "audio", "secret", "token",
]);

const PCM_ACCOUNTING_KEYS = new Set([
  "source_head_cursor", "unique_primary_admitted_cursor",
  "current_epoch_admitted_cursor", "processed_cursor", "old_processed_cursor",
  "cutover_cursor", "replay_start_cursor", "retained_range_floor",
  "retained_range_head", "received_samples", "unique_primary_admitted_samples",
  "replay_admitted_samples", "replay_retained_samples",
  "replay_inflight_copy_samples", "transition_queued_samples",
  "max_transition_queued_samples", "retained_source_samples",
  "max_retained_source_samples", "released_source_samples",
  "downstream_admission_rejected_samples", "explicit_source_rejected_samples",
]);

function isSafePcmAccounting(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)
      || ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return false;
  const entries = Object.entries(value);
  return entries.length > 0 && entries.length <= PCM_ACCOUNTING_KEYS.size
    && entries.every(([key, item]) => PCM_ACCOUNTING_KEYS.has(key.toLowerCase())
      && (item === null || (typeof item === "number" && Number.isFinite(item))));
}

function isContentFreeEpochSnapshot(value, depth = 0, path = []) {
  if (depth > 12) return false;
  if (Array.isArray(value)) {
    return value.length <= 16
      && value.every((item) => isContentFreeEpochSnapshot(item, depth + 1, [...path, "[]"]));
  }
  if (value && typeof value === "object") {
    return Object.entries(value).every(([key, item]) => {
      const normalizedKey = key.toLowerCase();
      if (normalizedKey === "pcm") {
        const isCurrentEpochAccounting = path.length === 1 && path[0] === "current_epoch";
        const isEpochHistoryAccounting = path.length === 2
          && path[0] === "epoch_history" && path[1] === "[]";
        return (isCurrentEpochAccounting || isEpochHistoryAccounting) && isSafePcmAccounting(item);
      }
      return !FORBIDDEN_EPOCH_SNAPSHOT_KEYS.has(normalizedKey)
        && isContentFreeEpochSnapshot(item, depth + 1, [...path, normalizedKey]);
    });
  }
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

export function safeQwenEpochObservabilitySnapshot(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const currentEpoch = value.current_epoch;
  const logical = value.logical_cumulative;
  if (!currentEpoch || typeof currentEpoch !== "object" || Array.isArray(currentEpoch)
      || !logical || typeof logical !== "object" || Array.isArray(logical)
      || !(value.last_transition === null || (value.last_transition && typeof value.last_transition === "object" && !Array.isArray(value.last_transition)))
      || !Array.isArray(value.epoch_history) || value.epoch_history.length > 8
      || !Array.isArray(value.transition_history) || value.transition_history.length > 8) return null;
  if (typeof currentEpoch.epoch_id !== "string" || !currentEpoch.epoch_id || currentEpoch.epoch_id.length > 160
      || !Number.isInteger(currentEpoch.epoch_seq) || currentEpoch.epoch_seq < 0
      || !Number.isInteger(logical.EPOCH_ROLLOVER_COUNT) || logical.EPOCH_ROLLOVER_COUNT < 0) return null;
  for (const transition of value.transition_history) {
    if (!transition || typeof transition !== "object" || Array.isArray(transition)
        || !Array.isArray(transition.stage_events) || transition.stage_events.length > 16) return null;
  }
  let encoded;
  try { encoded = JSON.stringify(value); } catch { return null; }
  if (encoded.length > 64_000 || !isContentFreeEpochSnapshot(value)) return null;
  return JSON.parse(encoded);
}
