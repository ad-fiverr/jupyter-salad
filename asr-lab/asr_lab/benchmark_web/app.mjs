import {
  AUDIO_PUSH_INTERVAL_MS, METRIC_DEFINITIONS, ShadowVad, boundedPush, clientFirstPartialMs, canvasBackingSize, csvCell, decodeSloLabel, formatMilliseconds, gpuStatusLabel,
  isTerminalQwenError, median, percentile, qwenDeviceTelemetryLabel, rttPercentiles, wordErrorRate,
  shouldDrawCanvas, safeQwenStartupEvidence, safeQwenEpochObservabilitySnapshot,
} from "/asr/benchmark/core.mjs";

const $ = (id) => document.getElementById(id);
const state = {
  token: null,
  ws: null,
  stream: null,
  context: null,
  sourceNode: null,
  workletNode: null,
  zeroGain: null,
  isRecording: false,
  isStopping: false,
  startedAt: null,
  runTimestamp: null,
  reference: "",
  backend: null,
  modelId: null,
  modelRevision: null,
  runtimeProvenance: null,
  workers: null,
  segments: [],
  errors: [],
  telemetry: [],
  rttSamples: [],
  rttTimer: null,
  rttPending: null,
  requestTimes: new Map(),
  segmentRequestIds: [],
  vad: new ShadowVad(),
  runPrefix: "",
  chunkSequence: 0,
  capturedSamples: 0,
  sentBytes: 0,
  flushId: null,
  flushResolver: null,
  workletResolver: null,
  timer: null,
  health: null,
  visibility: [],
  micSettings: null,
  activeSegmentRequestId: null,
  finalised: false,
  streamId: null,
  streamStartRequestId: null,
  streamStartResolver: null,
  streamEvents: [],
  qwenPublicStreamId: null,
  qwenLocalStreamId: null,
  qwenEpochObservability: null,
  lastQwenLifecycleError: null,
  lastPartialProducerKey: null,
  lastPartialRevision: 0,
  lastPartialAt: null,
  clientEosAt: null,
  clientFinalizationMs: null,
  qwenChunkMs: 1000,
  effectiveMaxBacklogMs: null,
  firstAudioSentAt: null,
  readinessAtStart: null,
  qwenStartMetrics: null,
  qwenLanguage: "auto",
  qwenContext: "",
};

const charts = {
  inference: $("chart-inference"), endpoint: $("chart-endpoint"), serverEos: $("chart-server-eos"),
  clientEos: $("chart-client-eos"), rtt: $("chart-rtt"), gpu: $("chart-gpu"),
  vram: $("chart-vram"), ram: $("chart-ram"),
  qwenWait: $("chart-qwen-wait"), qwenBacklog: $("chart-qwen-backlog"),
  qwenLag: $("chart-qwen-lag"), qwenDecode: $("chart-qwen-decode"),
};

function showNotice(message, level = "info") {
  const node = $("notice");
  node.textContent = message;
  node.dataset.state = level === "error" ? "error" : level === "ok" ? "ok" : "";
}

function status(node, label, value, stateName) {
  node.textContent = `${label}: ${value}`;
  node.dataset.state = stateName;
}

const fmtMs = formatMilliseconds;

function bytesToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x4000) {
    binary += String.fromCharCode(...bytes.subarray(offset, Math.min(offset + 0x4000, bytes.length)));
  }
  return btoa(binary);
}

function sortedSegments() {
  return [...state.segments].sort((a, b) => {
    const left = Number.isFinite(a.start) ? a.start : Number.MAX_SAFE_INTEGER;
    const right = Number.isFinite(b.start) ? b.start : Number.MAX_SAFE_INTEGER;
    return left - right || a.receivedAt - b.receivedAt;
  });
}

function transcriptText() {
  return sortedSegments().map((row) => row.text.trim()).filter(Boolean).join(" ");
}

function updateTranscript() {
  const text = transcriptText();
  const qwen = state.backend === "qwen3_asr";
  $("transcript-title").textContent = qwen ? "Candidato final · no confirmado" : "Transcripción final";
  $("transcript").textContent = text || (
    state.backend === "qwen3_asr"
      ? "El candidato final aparecerá al cerrar el stream; requiere reconciliación aguas abajo."
      : "La transcripción aparecerá cuando el servidor cierre un segmento."
  );
  const latest = state.telemetry.at(-1);
  const gpu = gpuStatusLabel(
    latest?.gpu_compute,
    { device: latest?.gpu_device, provider: latest?.gpu_telemetry_provider },
  );
  $("run-meta").textContent = [state.backend, state.modelId, `workers=${state.workers ?? "?"}`, gpu].filter(Boolean).join(" · ");
  const headers = $("segments-body")?.parentElement?.querySelectorAll("thead th");
  if (headers?.length >= 9) {
    headers[1].textContent = qwen ? "Candidato final · no confirmado" : "Transcripción";
    headers[7].textContent = qwen ? "Server EOS → candidato final ms" : "Server EOS → transcript ms";
    headers[8].textContent = qwen ? "Client EOS → candidato final ms*" : "Client EOS → transcript ms*";
  }
  document.querySelectorAll(".segment-offline-column").forEach((node) => { node.hidden = qwen; });
}

function updateSummary() {
  const rows = sortedSegments();
  const value = (row, canonical, legacy) => Number.isFinite(row[canonical]) ? row[canonical] : row[legacy];
  const inference = rows.map((row) => value(row, "SERVER_MODEL_INFERENCE_MS", "MODEL_INFERENCE_MS")).filter(Number.isFinite);
  const endpoint = rows.map((row) => row.SERVER_ENDPOINTING_MS).filter(Number.isFinite);
  const queue = rows.map((row) => value(row, "SERVER_QUEUE_WAIT_MS", "queue_wait_ms")).filter(Number.isFinite);
  const postprocess = rows.map((row) => row.SERVER_POSTPROCESS_MS).filter(Number.isFinite);
  const clientEos = rows.map((row) => value(row, "CLIENT_EOS_TO_TRANSCRIPT_MS", "CLIENT_AUDIO_END_TO_TRANSCRIPT_MS")).filter(Number.isFinite);
  const serverEos = rows.map((row) => value(row, "SERVER_EOS_TO_TRANSCRIPT_MS", "SERVER_AUDIO_END_TO_TRANSCRIPT_MS")).filter(Number.isFinite);
  const rtt = rttPercentiles(state.rttSamples);
  const audioMs = rows.map((row) => row.audio_duration_ms).filter(Number.isFinite).reduce((sum, value) => sum + value, 0);
  const inferenceMs = inference.reduce((sum, value) => sum + value, 0);
  const wer = state.backend === "qwen3_asr" || !state.reference.trim()
    ? null : wordErrorRate(state.reference, transcriptText());
  $("summary-segments").textContent = String(rows.length);
  $("summary-errors").textContent = String(state.errors.length);
  $("summary-audio").textContent = `${(state.capturedSamples / 16_000).toFixed(1)} s`;
  $("summary-rtf").textContent = state.backend === "qwen3_asr" ? "—"
    : audioMs > 0 ? (inferenceMs / audioMs).toFixed(3) : "—";
  $("summary-wer").textContent = wer == null ? "—" : `${(wer * 100).toFixed(1)}%`;
  const displayPair = (values) => values.length
    ? `${formatMilliseconds(median(values))} / ${formatMilliseconds(percentile(values, 0.95))}`
    : "—";
  $("summary-client-eos").textContent = displayPair(clientEos);
  $("summary-server-eos").textContent = displayPair(serverEos);
  $("summary-inference").textContent = displayPair(inference);
  $("summary-endpoint").textContent = displayPair(endpoint);
  $("summary-queue").textContent = displayPair(queue);
  $("summary-postprocess").textContent = displayPair(postprocess);
  $("summary-proxy-rtt").textContent = rtt.p50 == null ? "—"
    : `${formatMilliseconds(rtt.p50)} / ${formatMilliseconds(rtt.p95)}`;
}

function updateQwenSummary() {
  if (!$("qwen-first-partial")) return;
  const partials = state.streamEvents;
  const first = partials.find((item) => Number.isFinite(item.CLIENT_FIRST_PARTIAL_MS));
  const intervals = partials.map((item) => item.CLIENT_PARTIAL_UPDATE_INTERVAL_MS).filter(Number.isFinite);
  const latest = partials.at(-1);
  const finalRow = state.segments.at(-1);
  $("qwen-audio-duration").textContent = Number.isFinite(finalRow?.AUDIO_DURATION_MS)
    ? fmtMs(finalRow.AUDIO_DURATION_MS) : "—";
  $("qwen-first-partial").textContent = first ? fmtMs(first.CLIENT_FIRST_PARTIAL_MS) : "—";
  $("qwen-partial-interval").textContent = intervals.length ? fmtMs(median(intervals)) : "—";
  $("qwen-partial-stability").textContent = Number.isFinite(latest?.PARTIAL_STABILITY)
    ? (latest.PARTIAL_STABILITY * 100).toFixed(1) + "%" : "—";
  const serverEos = Number.isFinite(finalRow?.SERVER_EOS_TO_FINAL_CANDIDATE_MS)
    ? finalRow.SERVER_EOS_TO_FINAL_CANDIDATE_MS : finalRow?.FINALIZATION_AFTER_SERVER_EOS_MS;
  $("qwen-server-finalization").textContent = Number.isFinite(serverEos) ? fmtMs(serverEos) : "—";
  $("qwen-client-finalization").textContent = Number.isFinite(finalRow?.CLIENT_EOS_TO_FINAL_CANDIDATE_MS)
    ? fmtMs(finalRow.CLIENT_EOS_TO_FINAL_CANDIDATE_MS) : "—";
  const rtt = rttPercentiles(state.rttSamples);
  $("qwen-proxy-rtt").textContent = rtt.p50 == null
    ? "—" : fmtMs(rtt.p50) + " / " + fmtMs(rtt.p95);
  $("qwen-final-wer").textContent = state.reference.trim() && finalRow
    ? (wordErrorRate(state.reference, finalRow.text) * 100).toFixed(2) + "%" : "—";
  $("qwen-decode-wall").textContent = Number.isFinite(finalRow?.QWEN_CUMULATIVE_DECODE_WALL_MS)
    ? fmtMs(finalRow.QWEN_CUMULATIVE_DECODE_WALL_MS) : "—";
  $("qwen-stream-rtf").textContent = Number.isFinite(finalRow?.QWEN_STREAM_RTF)
    ? finalRow.QWEN_STREAM_RTF.toFixed(3) : Number.isFinite(latest?.QWEN_STREAM_RTF)
      ? latest.QWEN_STREAM_RTF.toFixed(3) : "—";
  $("qwen-partial-count").textContent = String(finalRow?.PARTIAL_COUNT ?? partials.length);
  const scheduler = state.health?.qwen_scheduler ?? {};
  const latestEvent = finalRow ?? partials.at(-1) ?? {};
  const latestTelemetry = state.telemetry.at(-1) ?? {};
  $("qwen-active-streams").textContent = `${scheduler.active_stream_count ?? "—"} / ${scheduler.max_active_streams ?? "—"}`;
  $("qwen-pending-active").textContent = `${scheduler.pending_decode_count ?? latestEvent.PENDING_DECODE_COUNT ?? "—"} / ${scheduler.active_decode_count ?? "—"}`;
  $("qwen-backlog").textContent = `${fmtMs(scheduler.qwen_scheduler_backlog_ms)} / ${fmtMs(scheduler.qwen_scheduler_max_stream_backlog_ms)}`;
  $("qwen-decode-percentiles").textContent = `${fmtMs(scheduler.qwen_decode_wall_p50_ms)} / ${fmtMs(scheduler.qwen_decode_wall_p95_ms)}`;
  $("qwen-wait-percentiles").textContent = `${fmtMs(scheduler.qwen_scheduler_wait_p50_ms)} / ${fmtMs(scheduler.qwen_scheduler_wait_p95_ms)}`;
  $("qwen-stream-lag").textContent = `${fmtMs(latestEvent.STREAM_LAG_MS)} / ${fmtMs(latestEvent.STREAM_LAG_MAX_MS ?? scheduler.qwen_scheduler_max_stream_lag_ms)}`;
  $("qwen-overruns").textContent = `${scheduler.qwen_decode_budget_overrun_total ?? "—"} / ${scheduler.qwen_scheduler_overrun_total ?? "—"}`;
  const startMetrics = state.qwenStartMetrics ?? {};
  $("qwen-cadence-latency").textContent = `init ${fmtMs(startMetrics.FIRST_STREAM_INIT_MS)} (worker ${fmtMs(startMetrics.FIRST_STREAM_STATE_INIT_WALL_MS)}, RPC overhead ${fmtMs(startMetrics.FIRST_STREAM_INIT_RPC_OVERHEAD_MS)}); first ready ${fmtMs(latestEvent.FIRST_AUDIO_TO_FIRST_DECODE_READY_MS)} + wait ${fmtMs(latestEvent.FIRST_SCHEDULER_WAIT_MS)} = start ${fmtMs(latestEvent.FIRST_AUDIO_TO_FIRST_DECODE_START_MS)}; decode #1 ${fmtMs(latestEvent.EPOCH_FIRST_DECODE_WALL_MS)}, #2 ${fmtMs(latestEvent.EPOCH_SECOND_DECODE_WALL_MS)}, steady p50/p95 ${fmtMs(latestEvent.EPOCH_STEADY_DECODE_WALL_P50_MS)} / ${fmtMs(latestEvent.EPOCH_STEADY_DECODE_WALL_P95_MS)}; first partial client/server ${fmtMs(first?.CLIENT_FIRST_PARTIAL_MS)} / ${fmtMs(first?.SERVER_FIRST_PARTIAL_MS)}; bound ${fmtMs(latestEvent.EFFECTIVE_MAX_BACKLOG_MS ?? state.effectiveMaxBacklogMs)}; SLO 100 ms ${decodeSloLabel(latestEvent.QWEN_DECODE_SLO_VIOLATION)}`;
  // collectTelemetry flattens GPU fields onto each point; keep the device
  // attribution explicitly global instead of expecting a nested snapshot.
  $("qwen-device-telemetry").textContent = qwenDeviceTelemetryLabel(latestTelemetry);
  renderQwenCharts();
}

function renderQwenPartials() {
  const body = $("qwen-partials-body");
  body.replaceChildren();
  if (!state.streamEvents.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 11;
    cell.className = "empty";
    cell.textContent = "Aún no hay parciales.";
    row.append(cell);
    body.append(row);
    $("qwen-current-partial").textContent = "Esperando el primer partial…";
    return;
  }
  for (const item of state.streamEvents) {
    const row = document.createElement("tr");
    const values = [
      fmtMs(item.audio_cursor_ms),
      fmtMs(item.CLIENT_ELAPSED_MS),
      item.revision,
      `${item.epoch_seq ?? "—"} · ${item.epoch_id ?? "—"}`,
      item.qwen_local_stream_id,
      item.text,
      fmtMs(item.SERVER_CHUNK_TO_PARTIAL_MS),
      Number.isFinite(item.PARTIAL_REVISION_RATE) ? (item.PARTIAL_REVISION_RATE * 100).toFixed(1) + "%" : "—",
      fmtMs(item.QWEN_DECODE_CALL_WALL_MS),
      fmtMs(item.QWEN_NEW_AUDIO_MS),
      fmtMs(item.EPOCH_AUDIO_ACCUMULATED_MS),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value == null ? "" : String(value);
      row.append(cell);
    }
    body.append(row);
  }
  const current = state.streamEvents.at(-1);
  $("qwen-current-partial").textContent = current.text;
  $("qwen-stream-meta").textContent = [
    "stream=" + String(state.streamId ?? "").slice(0, 8),
    "language=" + state.qwenLanguage,
    `push=${AUDIO_PUSH_INTERVAL_MS} ms`,
    "decode=" + (state.qwenChunkMs ?? "?") + " ms",
  ].join(" · ");
  updateQwenSummary();
}

function renderSegments() {
  const body = $("segments-body");
  body.replaceChildren();
  const rows = sortedSegments();
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "empty";
    td.textContent = "Aún no hay segmentos.";
    tr.append(td);
    body.append(tr);
    return;
  }
  rows.forEach((item, index) => {
    const tr = document.createElement("tr");
    const values = [
      `${index + 1}${Number.isFinite(item.start) ? ` · ${item.start.toFixed(2)} s` : ""}`,
      item.text,
      fmtMs(item.audio_duration_ms),
      fmtMs(item.SERVER_ENDPOINTING_MS),
      fmtMs(Number.isFinite(item.SERVER_QUEUE_WAIT_MS) ? item.SERVER_QUEUE_WAIT_MS : item.queue_wait_ms),
      fmtMs(Number.isFinite(item.SERVER_MODEL_INFERENCE_MS) ? item.SERVER_MODEL_INFERENCE_MS : item.MODEL_INFERENCE_MS),
      fmtMs(item.SERVER_POSTPROCESS_MS),
      fmtMs(item.candidate_only
        ? item.SERVER_EOS_TO_FINAL_CANDIDATE_MS
        : Number.isFinite(item.SERVER_EOS_TO_TRANSCRIPT_MS) ? item.SERVER_EOS_TO_TRANSCRIPT_MS : item.SERVER_AUDIO_END_TO_TRANSCRIPT_MS),
      fmtMs(item.candidate_only
        ? item.CLIENT_EOS_TO_FINAL_CANDIDATE_MS
        : Number.isFinite(item.CLIENT_EOS_TO_TRANSCRIPT_MS) ? item.CLIENT_EOS_TO_TRANSCRIPT_MS : item.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS),
    ];
    values.forEach((value, index) => {
      const td = document.createElement("td");
      if (index >= 3 && index <= 6) {
        td.className = "segment-offline-column";
        td.hidden = state.backend === "qwen3_asr";
      }
      td.textContent = value == null ? "" : String(value);
      tr.append(td);
    });
    body.append(tr);
  });
}

function drawChart(canvas, points, key, color = "#80e0b2", secondKey = null) {
  const rect = canvas.getBoundingClientRect();
  if (!shouldDrawCanvas(canvas.closest("[hidden]") !== null, rect.width, rect.height)) return;
  const dimensions = canvasBackingSize(rect.width, rect.height, window.devicePixelRatio || 1);
  if (!dimensions) return;
  const { width, height, dpr } = dimensions;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const margin = { left: 42, right: 10, top: 12, bottom: 24 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const series = [key, ...(secondKey ? [secondKey] : [])].map((name) => points.map((point) => point[name]).filter(Number.isFinite));
  const values = series.flat();
  ctx.strokeStyle = "#273845";
  ctx.fillStyle = "#9aabba";
  ctx.font = `${11 * dpr}px system-ui`;
  ctx.lineWidth = dpr;
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
    const max = values.reduce((largest, value) => Math.max(largest, value), 1);
    ctx.fillText((max * (1 - i / 4)).toFixed(0), 3, y + 4 * dpr);
  }
  const maxValue = values.reduce((largest, value) => Math.max(largest, value), 1);
  const lineColors = [color, "#f0c878"];
  series.forEach((items, seriesIndex) => {
    const actualPoints = points.map((point, index) => ({ x: index, y: point[seriesIndex === 0 ? key : secondKey] })).filter((point) => Number.isFinite(point.y));
    if (!actualPoints.length) return;
    ctx.strokeStyle = lineColors[seriesIndex];
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    actualPoints.forEach((point, index) => {
      const x = margin.left + (points.length <= 1 ? 0 : point.x / (points.length - 1)) * plotW;
      const y = margin.top + plotH - (point.y / maxValue) * plotH;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function renderQwenCharts() {
  if (!charts.qwenWait || $("qwen-stream-panel").hidden) return;
  const qwenTelemetry = state.telemetry
    .filter((point) => point.qwen_scheduler && typeof point.qwen_scheduler === "object")
    .map((point) => point.qwen_scheduler);
  drawChart(charts.qwenWait, qwenTelemetry.map((item) => ({
    QWEN_SCHEDULER_WAIT_P95_MS: item.qwen_scheduler_wait_p95_ms,
  })), "QWEN_SCHEDULER_WAIT_P95_MS", "#80e0b2");
  drawChart(charts.qwenBacklog, qwenTelemetry.map((item) => ({
    QWEN_DECODE_BACKLOG_MS: item.qwen_scheduler_backlog_ms,
    PENDING_DECODE_COUNT: item.pending_decode_count,
  })), "QWEN_DECODE_BACKLOG_MS", "#7dc4ff", "PENDING_DECODE_COUNT");
  drawChart(charts.qwenLag, qwenTelemetry.map((item) => ({
    STREAM_LAG_MS: item.qwen_scheduler_max_stream_lag_ms,
  })), "STREAM_LAG_MS", "#f0c878");
  drawChart(charts.qwenDecode, qwenTelemetry.map((item) => ({
    QWEN_DECODE_WALL_P95_MS: item.qwen_decode_wall_p95_ms,
    MODEL_DECODE_CHUNK_MS: state.qwenChunkMs,
  })), "QWEN_DECODE_WALL_P95_MS", "#f08080", "MODEL_DECODE_CHUNK_MS");
}

function renderCharts() {
  drawChart(charts.inference, sortedSegments(), "SERVER_MODEL_INFERENCE_MS");
  drawChart(charts.endpoint, sortedSegments(), "SERVER_ENDPOINTING_MS", "#f0c878");
  drawChart(charts.serverEos, sortedSegments(), "SERVER_EOS_TO_TRANSCRIPT_MS", "#80e0b2");
  drawChart(charts.clientEos, sortedSegments(), "CLIENT_EOS_TO_TRANSCRIPT_MS", "#7dc4ff");
  drawChart(charts.rtt, state.rttSamples, "PROXY_WS_RTT_MS", "#f0c878");
  drawChart(charts.gpu, state.telemetry, "gpu_utilization_pct", "#80e0b2");
  drawChart(charts.vram, state.telemetry, "vram_used_mib", "#7dc4ff", "vram_total_mib");
  drawChart(charts.ram, state.telemetry, "process_rss_mib", "#f0c878");
  renderQwenCharts();
}

function updateTimer() {
  if (state.startedAt == null) { $("timer").textContent = "00:00"; return; }
  const seconds = Math.max(0, Math.floor((performance.now() - state.startedAt) / 1000));
  $("timer").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

async function getTelemetry() {
  const response = await fetch("/asr/telemetry", {
    headers: { Authorization: `Bearer ${state.token}` },
    cache: "no-store",
    credentials: "omit",
  });
  if (response.status === 401) throw new Error("unauthorized");
  if (!response.ok) throw new Error("telemetry_unavailable");
  return response.json();
}

function recordTelemetry(snapshot) {
  const gpuTelemetry = snapshot.gpu_telemetry ?? snapshot.gpu ?? {};
  const gpuCompute = snapshot.gpu_compute ?? { available: Boolean(snapshot.cuda), cuda: Boolean(snapshot.cuda) };
  const point = {
    timestamp: snapshot.timestamp,
    process_rss_mib: snapshot.process_rss_mib,
    system_ram_total_mib: snapshot.system_ram?.total_mib ?? null,
    system_ram_used_mib: snapshot.system_ram?.used_mib ?? null,
    gpu_compute: gpuCompute,
    gpu_compute_available: gpuCompute.available === true || gpuCompute.cuda === true,
    gpu_device: gpuCompute.device ?? gpuTelemetry.device ?? null,
    gpu_telemetry_available: gpuTelemetry.available === true,
    gpu_telemetry_provider: gpuTelemetry.provider ?? "unavailable",
    gpu_utilization_pct: gpuTelemetry.utilization_pct ?? null,
    vram_used_mib: gpuTelemetry.vram_used_mib ?? null,
    vram_total_mib: gpuTelemetry.vram_total_mib ?? null,
    gpu_temperature_c: gpuTelemetry.temperature_c ?? null,
    gpu_power_w: gpuTelemetry.power_w ?? null,
    queue_depth: snapshot.queue_depth,
    model_loaded: snapshot.model_loaded,
    workers: snapshot.workers,
    qwen_scheduler: snapshot.qwen_scheduler ?? null,
  };
  state.telemetry.push(point);
  if (state.telemetry.length > 3600) state.telemetry.shift();
  status($("model-status"), "Modelo", snapshot.ready ? "listo" : "cargando / no listo", snapshot.ready ? "ready" : "unknown");
  const qwen = snapshot.backend === "qwen3_asr";
  if (snapshot.runtime_provenance && typeof snapshot.runtime_provenance === "object") {
    state.runtimeProvenance = snapshot.runtime_provenance;
  }
  $("transcript-mode").textContent = snapshot.transcript_mode ?? (qwen ? "STREAMING_PARTIALS" : "FINAL_SEGMENT");
  $("qwen-stream-controls").hidden = !qwen;
  $("qwen-stream-panel").hidden = !qwen;
  renderQwenLifecycle();
  $("offline-latency-groups").hidden = qwen;
  document.querySelectorAll(".offline-only").forEach((node) => { node.hidden = qwen; });
  updateQwenSummary();
  updateTranscript();
}

async function pollTelemetry() {
  if (!state.token || state.finalised) return;
  try {
    const snapshot = await getTelemetry();
    if (!state.token || state.finalised) return;
    state.health = snapshot;
    recordTelemetry(snapshot);
    if (snapshot.backend) state.backend = snapshot.backend;
    renderQwenLifecycle();
    if (snapshot.model_id) state.modelId = snapshot.model_id;
    if (snapshot.model_revision) state.modelRevision = snapshot.model_revision;
    if (snapshot.runtime_provenance && typeof snapshot.runtime_provenance === "object") {
      state.runtimeProvenance = snapshot.runtime_provenance;
    }
    if (Number.isInteger(snapshot.workers)) state.workers = snapshot.workers;
    updateTranscript();
    renderCharts();
  } catch (error) {
    if (error?.message === "unauthorized") showNotice("El token ASR no fue autorizado.", "error");
    else showNotice("No se pudo leer la telemetría. La captura y el WebSocket siguen separados.", "error");
  }
}

function onPartialTranscript(message) {
  if (message.stream_id !== state.streamId || message.final !== false || message.replace !== true) return;
  const currentEpoch = state.qwenEpochObservability?.current_epoch ?? null;
  const producerLocalId = typeof message.qwen_local_stream_id === "string"
    ? message.qwen_local_stream_id
    : null;
  const currentLocalId = state.qwenLocalStreamId ?? currentEpoch?.local_stream_id ?? null;
  if (!producerLocalId || !currentLocalId || producerLocalId !== currentLocalId) return;
  const producerEpoch = message.qwen_epoch_observability?.current_epoch ?? null;
  const producerEpochSeq = Number.isInteger(message.epoch_seq)
    ? message.epoch_seq
    : producerEpoch?.epoch_seq;
  const producerEpochId = typeof message.epoch_id === "string"
    ? message.epoch_id
    : producerEpoch?.epoch_id;
  if (currentEpoch && (
    (Number.isInteger(producerEpochSeq) && producerEpochSeq !== currentEpoch.epoch_seq)
    || (typeof producerEpochId === "string" && producerEpochId !== currentEpoch.epoch_id)
  )) return;
  const producerKey = [
    state.qwenPublicStreamId ?? message.stream_id,
    producerLocalId,
    producerEpochId ?? currentEpoch?.epoch_id ?? "",
    producerEpochSeq ?? currentEpoch?.epoch_seq ?? "",
  ].join("\u0000");
  if (state.lastPartialProducerKey !== producerKey) {
    state.lastPartialProducerKey = producerKey;
    state.lastPartialRevision = 0;
  }
  if (!Number.isInteger(message.revision) || message.revision <= state.lastPartialRevision) return;
  const receivedAt = performance.now();
  const previousAt = state.lastPartialAt;
  const clientElapsedMs = state.startedAt == null ? null : Math.max(0, receivedAt - state.startedAt);
  const clientFirstAudioMs = clientFirstPartialMs(state.firstAudioSentAt, receivedAt);
  const item = {
    event: "partial_candidate",
    stream_id: message.stream_id,
    public_stream_id: state.qwenPublicStreamId ?? message.stream_id ?? null,
    qwen_local_stream_id: state.qwenLocalStreamId,
    epoch_id: state.qwenEpochObservability?.current_epoch?.epoch_id ?? null,
    epoch_seq: state.qwenEpochObservability?.current_epoch?.epoch_seq ?? null,
    revision: message.revision,
    text: String(message.text ?? ""),
    language: typeof message.language === "string" ? message.language : null,
    audio_cursor_ms: Number.isFinite(message.audio_cursor_ms) ? message.audio_cursor_ms : null,
    CLIENT_ELAPSED_MS: clientElapsedMs,
    CLIENT_FIRST_PARTIAL_MS: state.streamEvents.length === 0 ? clientFirstAudioMs : null,
    CLIENT_PARTIAL_UPDATE_INTERVAL_MS: previousAt == null ? null : Math.max(0, receivedAt - previousAt),
    FIRST_PARTIAL_MS: message.FIRST_PARTIAL_MS,
    SERVER_FIRST_PARTIAL_MS: message.SERVER_FIRST_PARTIAL_MS,
    FIRST_AUDIO_TO_FIRST_DECODE_READY_MS: message.FIRST_AUDIO_TO_FIRST_DECODE_READY_MS,
    FIRST_SCHEDULER_WAIT_MS: message.FIRST_SCHEDULER_WAIT_MS,
    FIRST_AUDIO_TO_FIRST_DECODE_START_MS: message.FIRST_AUDIO_TO_FIRST_DECODE_START_MS,
    EPOCH_FIRST_DECODE_WALL_MS: message.EPOCH_FIRST_DECODE_WALL_MS,
    EPOCH_SECOND_DECODE_WALL_MS: message.EPOCH_SECOND_DECODE_WALL_MS,
    EPOCH_STEADY_DECODE_WALL_P50_MS: message.EPOCH_STEADY_DECODE_WALL_P50_MS,
    EPOCH_STEADY_DECODE_WALL_P95_MS: message.EPOCH_STEADY_DECODE_WALL_P95_MS,
    EFFECTIVE_MAX_BACKLOG_MS: message.EFFECTIVE_MAX_BACKLOG_MS,
    QWEN_MAX_BACKLOG_CHUNKS: message.QWEN_MAX_BACKLOG_CHUNKS,
    QWEN_DECODE_SLO_TARGET_MS: message.QWEN_DECODE_SLO_TARGET_MS,
    QWEN_DECODE_SLO_VIOLATION: message.QWEN_DECODE_SLO_VIOLATION,
    PARTIAL_UPDATE_INTERVAL_MS: message.PARTIAL_UPDATE_INTERVAL_MS,
    PARTIAL_COUNT: message.PARTIAL_COUNT,
    PARTIAL_REVISION_RATE: message.PARTIAL_REVISION_RATE,
    PARTIAL_STABILITY: message.PARTIAL_STABILITY,
    SERVER_CHUNK_TO_PARTIAL_MS: message.SERVER_CHUNK_TO_PARTIAL_MS,
    QWEN_DECODE_CALL_WALL_MS: message.QWEN_DECODE_CALL_WALL_MS,
    QWEN_CUMULATIVE_DECODE_WALL_MS: message.QWEN_CUMULATIVE_DECODE_WALL_MS,
    QWEN_STREAM_RTF: message.QWEN_STREAM_RTF,
    QWEN_SCHEDULER_WAIT_MS: message.QWEN_SCHEDULER_WAIT_MS,
    QWEN_DECODE_BACKLOG_MS: message.QWEN_DECODE_BACKLOG_MS,
    QWEN_DECODE_BACKLOG_MAX_MS: message.QWEN_DECODE_BACKLOG_MAX_MS,
    STREAM_LAG_MS: message.STREAM_LAG_MS,
    STREAM_LAG_MAX_MS: message.STREAM_LAG_MAX_MS,
    PENDING_DECODE_COUNT: message.PENDING_DECODE_COUNT,
    ACTIVE_STREAM_COUNT: message.ACTIVE_STREAM_COUNT,
    DECODE_OVERRUN: message.DECODE_OVERRUN === true,
    MODEL_DECODE_CHUNK_MS: message.MODEL_DECODE_CHUNK_MS,
    AUDIO_PUSH_INTERVAL_MS: message.AUDIO_PUSH_INTERVAL_MS,
    QWEN_NEW_AUDIO_MS: message.QWEN_NEW_AUDIO_MS ?? null,
    EPOCH_AUDIO_ACCUMULATED_MS: message.EPOCH_AUDIO_ACCUMULATED_MS ?? null,
    QWEN_AUDIO_ACCUM_MS: message.QWEN_AUDIO_ACCUM_MS ?? null,
    client_received_at: receivedAt,
    candidate_only: true,
    provisional: true,
    final: false,
    truth_status: message.truth_status ?? "candidate_only",
  };
  state.streamEvents.push(item);
  if (state.streamEvents.length > 1000) state.streamEvents.shift();
  state.lastPartialRevision = message.revision;
  state.lastPartialAt = receivedAt;
  renderQwenPartials();
}

function appendLifecycleRow(body, values, className = "", colSpan = 1) {
  const row = document.createElement("tr");
  if (className) row.className = className;
  for (const value of values) {
    const cell = document.createElement("td");
    if (values.length === 1) cell.colSpan = colSpan;
    cell.textContent = value == null ? "—" : String(value);
    row.append(cell);
  }
  body.append(row);
}

function captureQwenEpochEvidence(message) {
  const publicStreamId = message?.stream_id ?? message?.qwen_public_stream_id;
  if (publicStreamId && state.streamId && publicStreamId !== state.streamId) return;
  const snapshot = safeQwenEpochObservabilitySnapshot(message?.qwen_epoch_observability);
  const incomingEpoch = snapshot?.current_epoch ?? null;
  const currentEpoch = state.qwenEpochObservability?.current_epoch ?? null;
  const messageLocalId = typeof message?.qwen_local_stream_id === "string"
    ? message.qwen_local_stream_id
    : null;
  const snapshotLocalId = typeof incomingEpoch?.local_stream_id === "string"
    ? incomingEpoch.local_stream_id
    : null;
  if (messageLocalId && snapshotLocalId && messageLocalId !== snapshotLocalId) return;
  if (incomingEpoch && currentEpoch) {
    if (incomingEpoch.epoch_seq < currentEpoch.epoch_seq) return;
    if (incomingEpoch.epoch_seq === currentEpoch.epoch_seq && (
      incomingEpoch.epoch_id !== currentEpoch.epoch_id
      || (snapshotLocalId && currentEpoch.local_stream_id && snapshotLocalId !== currentEpoch.local_stream_id)
    )) return;
  } else if (!incomingEpoch && messageLocalId && state.qwenLocalStreamId
      && messageLocalId !== state.qwenLocalStreamId) {
    return;
  }
  if (typeof publicStreamId === "string" && publicStreamId
      && (!state.streamId || publicStreamId === state.streamId)) {
    state.qwenPublicStreamId = publicStreamId;
  }
  if (snapshotLocalId) {
    state.qwenLocalStreamId = snapshotLocalId;
  } else if (messageLocalId) {
    state.qwenLocalStreamId = messageLocalId;
  }
  if (snapshot) {
    state.qwenEpochObservability = snapshot;
    const currentLocalId = snapshot.current_epoch?.local_stream_id;
    if (typeof currentLocalId === "string" && currentLocalId) state.qwenLocalStreamId = currentLocalId;
  }
  renderQwenLifecycle();
}

function renderQwenLifecycle() {
  const panel = $("qwen-lifecycle-panel");
  if (!panel) return;
  const snapshot = state.qwenEpochObservability;
  const lastError = state.lastQwenLifecycleError;
  panel.hidden = state.backend !== "qwen3_asr" || (!snapshot && !lastError);
  const current = snapshot?.current_epoch ?? {};
  const logical = snapshot?.logical_cumulative ?? {};
  const transition = snapshot?.last_transition ?? null;
  $("qwen-lifecycle-public-id").textContent = state.qwenPublicStreamId ?? lastError?.qwen_public_stream_id ?? "—";
  $("qwen-lifecycle-current").textContent = `${current.epoch_seq ?? "—"} / ${current.epoch_id ?? "—"} · ${state.qwenLocalStreamId ?? current.local_stream_id ?? "—"}`;
  $("qwen-lifecycle-state").textContent = current.lifecycle_state ?? "—";
  $("qwen-lifecycle-rollovers").textContent = `${logical.rollover_success_count ?? 0} / ${logical.rollover_attempt_count ?? 0}`;
  $("qwen-lifecycle-transition").textContent = transition ? `${transition.state ?? "—"} / ${transition.stage ?? "—"}` : "Sin transition";
  $("qwen-lifecycle-last-completed").textContent = transition?.last_completed_stage ?? "—";
  $("qwen-lifecycle-reason").textContent = transition
    ? `${transition.category ?? "—"} / ${transition.reason ?? "—"}` : "—";
  $("qwen-lifecycle-failure").textContent = transition?.failure_stage || transition?.failure_reason
    ? `${transition.failure_stage ?? "—"} / ${transition.failure_reason ?? "—"}`
    : lastError?.code ?? "—";
  $("qwen-lifecycle-latency").textContent = transition
    ? `${fmtMs(transition.EPOCH_HANDOFF_WALL_MS)} / ${fmtMs(transition.EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS)}` : "—";
  const pcm = transition?.latest_pcm_accounting ?? {};
  $("qwen-lifecycle-pcm").textContent = `${fmtMs(transition?.transition_buffer_current_ms)} / ${fmtMs(transition?.transition_buffer_max_ms)} / ${pcm.explicit_source_rejected_samples ?? "—"}`;
  $("qwen-lifecycle-retained").textContent = `${fmtMs(transition?.retained_pcm_current_ms)} / ${fmtMs(transition?.retained_pcm_max_ms)}`;
  $("qwen-lifecycle-stale").textContent = String(transition?.stale_result_rejects_delta ?? "—");
  $("qwen-lifecycle-audio-accum").textContent = snapshot?.QWEN_AUDIO_ACCUM_MS == null
    ? "null · no medible" : `${fmtMs(snapshot?.QWEN_AUDIO_ACCUM_MS)} · version-coupled`;

  const epochBody = $("qwen-epoch-history-body");
  epochBody.replaceChildren();
  const epochs = Array.isArray(snapshot?.epoch_history) ? snapshot?.epoch_history.slice(-8) : [];
  if (!epochs.length) appendLifecycleRow(epochBody, ["No hay historial de epochs."], "empty", 6);
  for (const epoch of epochs) {
    appendLifecycleRow(epochBody, [
      epoch.epoch_seq, epoch.epoch_id, epoch.local_stream_id, epoch.lifecycle_state,
      fmtMs(epoch.epoch_audio_ms), `${fmtMs(epoch.replay_audio_ms)} / ${fmtMs(epoch.replay_wall_ms)}`,
    ]);
  }

  const transitionBody = $("qwen-transition-history-body");
  transitionBody.replaceChildren();
  const transitions = Array.isArray(snapshot?.transition_history) ? snapshot?.transition_history.slice(-8) : [];
  if (!transitions.length) appendLifecycleRow(transitionBody, ["No hay historial de transitions."], "empty", 8);
  for (const item of transitions) {
    const stages = Array.isArray(item.stage_events)
      ? item.stage_events.slice(-16).map((event) => `${event.stage ?? "?"} ${fmtMs(event.elapsed_ms)}`).join(" → ")
      : "—";
    appendLifecycleRow(transitionBody, [
      item.transition_seq,
      `${item.predecessor_epoch_id ?? "?"}/${item.predecessor_local_stream_id ?? "?"} → ${item.successor_epoch_id ?? "?"}/${item.successor_local_stream_id ?? "?"}`,
      `${item.state ?? "?"} / ${item.stage ?? "?"}`,
      stages,
      `${item.replay_admitted_samples ?? 0} / ${item.transition_primary_samples ?? 0}`,
      `${fmtMs(item.EPOCH_HANDOFF_WALL_MS)} / ${fmtMs(item.EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS)}`,
      item.failure_stage || item.failure_reason ? `${item.failure_stage ?? "?"} / ${item.failure_reason ?? "?"}` : "—",
      `${item.PCM_LOST ?? "unmeasured"} / ${item.PRIMARY_DUP ?? "unmeasured"}`,
    ]);
  }
}

function onTranscript(message, candidateOnly = false) {
  const receivedAt = performance.now();
  const clientStart = typeof message.client_request_id === "string"
    ? state.requestTimes.get(message.client_request_id)
    : null;
  if (typeof message.client_request_id === "string") state.requestTimes.delete(message.client_request_id);
  const clientEosMs = state.backend === "qwen3_asr" && Number.isFinite(state.clientEosAt)
    ? Math.max(0, receivedAt - state.clientEosAt)
    : Number.isFinite(clientStart) ? Math.max(0, receivedAt - clientStart) : null;
  const modelMs = Number.isFinite(message.SERVER_MODEL_INFERENCE_MS)
    ? message.SERVER_MODEL_INFERENCE_MS
    : Number.isFinite(message.MODEL_INFERENCE_MS) ? message.MODEL_INFERENCE_MS : null;
  const queueMs = Number.isFinite(message.SERVER_QUEUE_WAIT_MS)
    ? message.SERVER_QUEUE_WAIT_MS
    : Number.isFinite(message.queue_wait_ms) ? message.queue_wait_ms : null;
  const serverEosMs = !candidateOnly && Number.isFinite(message.SERVER_EOS_TO_TRANSCRIPT_MS)
    ? message.SERVER_EOS_TO_TRANSCRIPT_MS
    : !candidateOnly && Number.isFinite(message.FINALIZATION_AFTER_SERVER_EOS_MS)
      ? message.FINALIZATION_AFTER_SERVER_EOS_MS
      : !candidateOnly && Number.isFinite(message.SERVER_AUDIO_END_TO_TRANSCRIPT_MS)
        ? message.SERVER_AUDIO_END_TO_TRANSCRIPT_MS : null;
  state.segments.push({
    event: candidateOnly ? "final_candidate" : "transcript",
    candidate_only: candidateOnly,
    text: String(message.text ?? ""),
    start: Number.isFinite(message.start) ? message.start : null,
    end: Number.isFinite(message.end) ? message.end : null,
    speaker: String(message.speaker ?? "you"),
    backend: String(message.backend ?? state.backend ?? "unknown"),
    SERVER_MODEL_INFERENCE_MS: modelMs,
    MODEL_INFERENCE_MS: modelMs,
    SERVER_ENDPOINTING_MS: Number.isFinite(message.SERVER_ENDPOINTING_MS) ? message.SERVER_ENDPOINTING_MS : null,
    SERVER_QUEUE_WAIT_MS: queueMs,
    SERVER_POSTPROCESS_MS: Number.isFinite(message.SERVER_POSTPROCESS_MS) ? message.SERVER_POSTPROCESS_MS : null,
    SERVER_EOS_TO_TRANSCRIPT_MS: serverEosMs,
    SERVER_EOS_TO_FINAL_CANDIDATE_MS: candidateOnly && Number.isFinite(message.SERVER_EOS_TO_FINAL_CANDIDATE_MS)
      ? message.SERVER_EOS_TO_FINAL_CANDIDATE_MS : null,
    CLIENT_EOS_TO_TRANSCRIPT_MS: candidateOnly ? null : clientEosMs,
    CLIENT_EOS_TO_FINAL_CANDIDATE_MS: candidateOnly ? clientEosMs : null,
    SEGMENT_WAIT_MS: Number.isFinite(message.SEGMENT_WAIT_MS) ? message.SEGMENT_WAIT_MS : null,
    SERVER_TO_TRANSCRIPT_MS: Number.isFinite(message.SERVER_TO_TRANSCRIPT_MS) ? message.SERVER_TO_TRANSCRIPT_MS : null,
    SERVER_RECEIVE_TO_TRANSCRIPT_MS: Number.isFinite(message.SERVER_RECEIVE_TO_TRANSCRIPT_MS) ? message.SERVER_RECEIVE_TO_TRANSCRIPT_MS : null,
    SERVER_AUDIO_END_TO_TRANSCRIPT_MS: serverEosMs,
    CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: candidateOnly ? null : clientEosMs,
    FINALIZATION_AFTER_SERVER_EOS_MS: Number.isFinite(message.FINALIZATION_AFTER_SERVER_EOS_MS)
      ? message.FINALIZATION_AFTER_SERVER_EOS_MS : null,
    QWEN_CUMULATIVE_DECODE_WALL_MS: Number.isFinite(message.QWEN_CUMULATIVE_DECODE_WALL_MS)
      ? message.QWEN_CUMULATIVE_DECODE_WALL_MS : null,
    QWEN_STREAM_RTF: Number.isFinite(message.QWEN_STREAM_RTF) ? message.QWEN_STREAM_RTF : null,
    QWEN_SCHEDULER_WAIT_MS: Number.isFinite(message.QWEN_SCHEDULER_WAIT_MS) ? message.QWEN_SCHEDULER_WAIT_MS : null,
    QWEN_SCHEDULER_WAIT_P50_MS: Number.isFinite(message.QWEN_SCHEDULER_WAIT_P50_MS) ? message.QWEN_SCHEDULER_WAIT_P50_MS : null,
    QWEN_SCHEDULER_WAIT_P95_MS: Number.isFinite(message.QWEN_SCHEDULER_WAIT_P95_MS) ? message.QWEN_SCHEDULER_WAIT_P95_MS : null,
    QWEN_DECODE_WALL_P50_MS: Number.isFinite(message.QWEN_DECODE_WALL_P50_MS) ? message.QWEN_DECODE_WALL_P50_MS : null,
    QWEN_DECODE_WALL_P95_MS: Number.isFinite(message.QWEN_DECODE_WALL_P95_MS) ? message.QWEN_DECODE_WALL_P95_MS : null,
    QWEN_DECODE_BACKLOG_MS: Number.isFinite(message.QWEN_DECODE_BACKLOG_MS) ? message.QWEN_DECODE_BACKLOG_MS : null,
    QWEN_DECODE_BACKLOG_MAX_MS: Number.isFinite(message.QWEN_DECODE_BACKLOG_MAX_MS) ? message.QWEN_DECODE_BACKLOG_MAX_MS : null,
    STREAM_LAG_MS: Number.isFinite(message.STREAM_LAG_MS) ? message.STREAM_LAG_MS : null,
    STREAM_LAG_MAX_MS: Number.isFinite(message.STREAM_LAG_MAX_MS) ? message.STREAM_LAG_MAX_MS : null,
    PENDING_DECODE_COUNT: Number.isInteger(message.PENDING_DECODE_COUNT) ? message.PENDING_DECODE_COUNT : null,
    ACTIVE_STREAM_COUNT: Number.isInteger(message.ACTIVE_STREAM_COUNT) ? message.ACTIVE_STREAM_COUNT : null,
    DECODE_OVERRUN: message.DECODE_OVERRUN === true,
    QWEN_DECODE_STEPS_DELTA: Number.isInteger(message.QWEN_DECODE_STEPS_DELTA) ? message.QWEN_DECODE_STEPS_DELTA : null,
    QWEN_DECODE_STEPS_DELTA_TOTAL: Number.isInteger(message.QWEN_DECODE_STEPS_DELTA_TOTAL) ? message.QWEN_DECODE_STEPS_DELTA_TOTAL : null,
    QWEN_FINISH_DECODE_WALL_MS: Number.isFinite(message.QWEN_FINISH_DECODE_WALL_MS) ? message.QWEN_FINISH_DECODE_WALL_MS : null,
    FIRST_STREAM_INIT_MS: message.FIRST_STREAM_INIT_MS ?? null,
    FIRST_STREAM_STATE_INIT_WALL_MS: message.FIRST_STREAM_STATE_INIT_WALL_MS ?? null,
    FIRST_STREAM_INIT_RPC_OVERHEAD_MS: message.FIRST_STREAM_INIT_RPC_OVERHEAD_MS ?? null,
    FIRST_AUDIO_TO_FIRST_DECODE_READY_MS: message.FIRST_AUDIO_TO_FIRST_DECODE_READY_MS ?? null,
    FIRST_SCHEDULER_WAIT_MS: message.FIRST_SCHEDULER_WAIT_MS ?? null,
    FIRST_AUDIO_TO_FIRST_DECODE_START_MS: message.FIRST_AUDIO_TO_FIRST_DECODE_START_MS ?? null,
    EPOCH_FIRST_DECODE_WALL_MS: message.EPOCH_FIRST_DECODE_WALL_MS ?? null,
    EPOCH_SECOND_DECODE_WALL_MS: message.EPOCH_SECOND_DECODE_WALL_MS ?? null,
    EPOCH_STEADY_DECODE_WALL_P50_MS: message.EPOCH_STEADY_DECODE_WALL_P50_MS ?? null,
    EPOCH_STEADY_DECODE_WALL_P95_MS: message.EPOCH_STEADY_DECODE_WALL_P95_MS ?? null,
    EFFECTIVE_MAX_BACKLOG_MS: message.EFFECTIVE_MAX_BACKLOG_MS ?? null,
    QWEN_MAX_BACKLOG_CHUNKS: message.QWEN_MAX_BACKLOG_CHUNKS ?? null,
    QWEN_DECODE_SLO_TARGET_MS: message.QWEN_DECODE_SLO_TARGET_MS ?? 100,
    QWEN_DECODE_SLO_VIOLATION: message.QWEN_DECODE_SLO_VIOLATION === true,
    PARTIAL_COUNT: Number.isFinite(message.PARTIAL_COUNT) ? message.PARTIAL_COUNT : null,
    FIRST_PARTIAL_MS: Number.isFinite(message.FIRST_PARTIAL_MS) ? message.FIRST_PARTIAL_MS : null,
    PARTIAL_REVISION_RATE: Number.isFinite(message.PARTIAL_REVISION_RATE) ? message.PARTIAL_REVISION_RATE : null,
    PARTIAL_STABILITY: Number.isFinite(message.PARTIAL_STABILITY) ? message.PARTIAL_STABILITY : null,
    MODEL_DECODE_CHUNK_MS: Number.isFinite(message.MODEL_DECODE_CHUNK_MS) ? message.MODEL_DECODE_CHUNK_MS : null,
    AUDIO_PUSH_INTERVAL_MS: Number.isFinite(message.AUDIO_PUSH_INTERVAL_MS) ? message.AUDIO_PUSH_INTERVAL_MS : null,
    queue_wait_ms: queueMs,
    audio_duration_ms: Number.isFinite(message.AUDIO_DURATION_MS)
      ? message.AUDIO_DURATION_MS
      : Number.isFinite(message.audio_duration_ms) ? message.audio_duration_ms : null,
    request_id: typeof message.request_id === "string" ? message.request_id : null,
    client_request_id: typeof message.client_request_id === "string" ? message.client_request_id : null,
    stream_id: typeof message.stream_id === "string" ? message.stream_id : null,
    revision: Number.isInteger(message.revision) ? message.revision : null,
    audio_cursor_ms: Number.isFinite(message.audio_cursor_ms) ? message.audio_cursor_ms : null,
    receivedAt,
    truth_status: candidateOnly ? "candidate_only" : message.truth_status ?? null,
    provisional: candidateOnly,
  });
  renderSegments();
  updateTranscript();
  updateSummary();
  updateQwenSummary();
}

function onSocketMessage(raw) {
  let message;
  try { message = JSON.parse(raw); } catch { state.errors.push("invalid_server_message"); updateSummary(); return; }
  captureQwenEpochEvidence(message);
  if (state.backend === "qwen3_asr" && (
    message.event === "partial_transcript" || message.event === "transcript" || message.type === "transcript"
  )) return;
  if (message.event === "benchmark_pong") {
    const pending = state.rttPending;
    if (pending && message.request_id === pending.request_id) {
      clearTimeout(pending.timeoutId);
      state.rttPending = null;
      const receivedAt = performance.now();
      boundedPush(state.rttSamples, {
        elapsed_ms: state.startedAt == null ? null : Math.max(0, receivedAt - state.startedAt),
        PROXY_WS_RTT_MS: Math.max(0, receivedAt - pending.sentAt),
        status: "ok",
      });
      updateSummary();
      renderCharts();
    }
    return;
  }
  if (message.event === "stream_started") {
    if (message.request_id && message.request_id !== state.streamStartRequestId) return;
    state.streamId = message.stream_id ?? null;
    state.qwenChunkMs = Number.isFinite(message.model_chunk_ms) ? message.model_chunk_ms : (Number.isFinite(message.model_decode_chunk_ms) ? message.model_decode_chunk_ms : state.qwenChunkMs);
    state.effectiveMaxBacklogMs = Number.isFinite(message.effective_max_backlog_ms) ? message.effective_max_backlog_ms : state.effectiveMaxBacklogMs;
    state.qwenStartMetrics = {
      FIRST_STREAM_INIT_MS: message.FIRST_STREAM_INIT_MS ?? null,
      FIRST_STREAM_STATE_INIT_WALL_MS: message.FIRST_STREAM_STATE_INIT_WALL_MS ?? null,
      FIRST_STREAM_INIT_RPC_OVERHEAD_MS: message.FIRST_STREAM_INIT_RPC_OVERHEAD_MS ?? null,
      model_chunk_ms: state.qwenChunkMs,
      audio_push_interval_ms: AUDIO_PUSH_INTERVAL_MS,
      effective_max_backlog_ms: state.effectiveMaxBacklogMs,
      qwen_max_backlog_chunks: message.qwen_max_backlog_chunks ?? null,
      readiness_at_start: state.readinessAtStart,
    };
    state.qwenLanguage = message.language ?? state.qwenLanguage;
    state.streamStartResolver?.resolve(message);
    state.streamStartResolver = null;
    $("qwen-stream-meta").textContent = "stream=" + String(state.streamId ?? "").slice(0, 8)
      + " · language=" + state.qwenLanguage + " · push=100 ms · decode=" + state.qwenChunkMs + " ms";
    return;
  }
  if (message.event === "partial_transcript") {
    onPartialTranscript(message);
    return;
  }
  if (message.event === "partial_candidate") {
    if (message.truth_status !== "candidate_only" || message.final !== false) return;
    onPartialTranscript(message);
    return;
  }
  if (message.event === "final_candidate") {
    if (message.truth_status !== "candidate_only" || message.candidate_only !== true || message.final !== true) return;
    onTranscript(message, true);
    return;
  }
  if (message.event === "transcript" || message.type === "transcript") {
    onTranscript(message);
    return;
  }
  if (message.event === "error") {
    const code = /^[a-z0-9_]{1,64}$/i.test(message.code ?? "") ? message.code : "asr_error";
    state.errors.push(code);
    if (state.backend === "qwen3_asr" || state.qwenEpochObservability) {
      state.lastQwenLifecycleError = {
        code,
        qwen_public_stream_id: state.qwenPublicStreamId,
        qwen_local_stream_id: state.qwenLocalStreamId,
        lifecycle_snapshot_received: state.qwenEpochObservability !== null,
      };
      renderQwenLifecycle();
    }
    if (state.streamStartResolver) {
      state.streamStartResolver.reject(new Error(code));
      state.streamStartResolver = null;
    }
    showNotice(`Error ASR: ${code}`, "error");
    updateSummary();
    if (state.backend === "qwen3_asr" && isTerminalQwenError(code)) {
      state.isRecording = false;
      state.isStopping = false;
      showNotice(`El stream Qwen terminó (${code}); inicia una sesión explícita para volver a enviar audio.`, "error");
      void cleanup(false);
    }
    return;
  }
  if (message.event === "flush_complete" && message.request_id === state.flushId) {
    state.flushResolver?.();
    state.flushResolver = null;
    state.flushId = null;
  }
}

function recordRttFailure(statusName) {
  const now = performance.now();
  boundedPush(state.rttSamples, {
    elapsed_ms: state.startedAt == null ? null : Math.max(0, now - state.startedAt),
    PROXY_WS_RTT_MS: null,
    status: statusName,
  });
  updateSummary();
  renderCharts();
}

function sendRttPing() {
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN || state.rttPending) return;
  const requestId = crypto.randomUUID();
  const sentAt = performance.now();
  const pending = { request_id: requestId, sentAt, timeoutId: null };
  pending.timeoutId = setTimeout(() => {
    if (state.rttPending !== pending) return;
    state.rttPending = null;
    recordRttFailure("timeout");
  }, 4500);
  state.rttPending = pending;
  try {
    ws.send(JSON.stringify({ event: "benchmark_ping", request_id: requestId }));
  } catch {
    clearTimeout(pending.timeoutId);
    if (state.rttPending === pending) state.rttPending = null;
    recordRttFailure("send_error");
  }
}

function startRttSampler() {
  stopRttSampler();
  sendRttPing();
  state.rttTimer = setInterval(sendRttPing, 5000);
}

function stopRttSampler() {
  if (state.rttTimer) clearInterval(state.rttTimer);
  state.rttTimer = null;
  if (state.rttPending) clearTimeout(state.rttPending.timeoutId);
  state.rttPending = null;
}

function openSocket(token) {
  const url = new URL("/asr/ws", window.location.href);
  url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("token", token);
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url.href);
    state.ws = ws;
    ws.onopen = () => { status($("ws-status"), "WebSocket", "conectado", "open"); resolve(ws); };
    ws.onmessage = (event) => onSocketMessage(event.data);
    ws.onerror = () => { status($("ws-status"), "WebSocket", "error", "error"); };
    ws.onclose = (event) => {
      status($("ws-status"), "WebSocket", "cerrado", "closed");
      if (!state.finalised && event.code === 4401) showNotice("El token ASR no fue autorizado.", "error");
      if (!state.finalised && !state.isStopping) {
        reject(new Error("websocket_closed"));
        if (state.isRecording) {
          state.errors.push("websocket_closed");
          updateSummary();
          showNotice("Se perdió la conexión ASR; se detuvo la captura y no se confirmó el último flush.", "error");
          void cleanup(false);
        }
      }
    };
    setTimeout(() => {
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.close();
        reject(new Error("websocket_timeout"));
      }
    }, 15_000);
  });
}

function waitForQwenStreamStart(requestId) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      if (state.streamStartResolver?.requestId !== requestId) return;
      state.streamStartResolver = null;
      reject(new Error("stream_start_timeout"));
    }, 15_000);
    state.streamStartResolver = {
      requestId,
      resolve: (message) => { clearTimeout(timeoutId); resolve(message); },
      reject: (error) => { clearTimeout(timeoutId); reject(error); },
    };
  });
}

async function startQwenStream() {
  try {
    const readinessResponse = await fetch("/asr/readiness", { cache: "no-store" });
    state.readinessAtStart = await readinessResponse.json();
  } catch {
    state.readinessAtStart = { ready: false };
  }
  state.qwenChunkMs = Number($("qwen-chunk-size").value);
  state.qwenLanguage = $("qwen-language").value;
  state.qwenContext = $("qwen-context").value.trim();
  const requestId = crypto.randomUUID();
  state.streamStartRequestId = requestId;
  const started = waitForQwenStreamStart(requestId);
  state.ws.send(JSON.stringify({
    event: "stream_start",
    source: "mic",
    request_id: requestId,
    language: state.qwenLanguage,
    context: state.qwenContext,
    model_chunk_ms: state.qwenChunkMs,
  }));
  await started;
}

function receivePcmChunk(buffer, sampleCount) {
  if ((!state.isRecording && !state.isStopping) || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.capturedSamples += sampleCount;
  if (state.backend === "qwen3_asr") {
    state.chunkSequence += 1;
    if (state.ws.bufferedAmount > 512_000) {
      state.errors.push("websocket_backpressure");
      updateSummary();
      showNotice("La red no alcanza la cadencia de captura; deteniendo para no descartar audio silenciosamente.", "error");
      void stopRun();
      return;
    }
    const bytes = new Uint8Array(buffer);
    if (state.firstAudioSentAt == null) state.firstAudioSentAt = performance.now();
    state.sentBytes += bytes.byteLength;
    state.ws.send(JSON.stringify({
      type: "audio", source: "mic", speaker: "you", encoding: "pcm_int16",
      sample_rate: 16_000, audio: bytesToBase64(buffer),
    }));
    return;
  }
  const now = performance.now();
  const shadow = state.vad.push(buffer, now);
  let requestId = null;
  if ((shadow.inSegment || shadow.closed) && shadow.lastVoiceAt != null) {
    if (!state.activeSegmentRequestId) {
      state.activeSegmentRequestId = `${state.runPrefix}:${state.chunkSequence}`;
    }
    requestId = state.activeSegmentRequestId;
    state.requestTimes.set(requestId, shadow.lastVoiceAt);
    while (state.requestTimes.size > 8192) state.requestTimes.delete(state.requestTimes.keys().next().value);
  }
  state.chunkSequence += 1;
  if (shadow.closed) {
    if (!shadow.canTranscribe && state.activeSegmentRequestId) {
      state.requestTimes.delete(state.activeSegmentRequestId);
    }
    state.activeSegmentRequestId = null;
  }
  if (state.ws.bufferedAmount > 512_000) {
    state.errors.push("websocket_backpressure");
    updateSummary();
    showNotice("La red no alcanza la cadencia de captura; deteniendo para no descartar audio silenciosamente.", "error");
    void stopRun();
    return;
  }
  const bytes = new Uint8Array(buffer);
  const payload = {
    type: "audio", source: "mic", speaker: "you", encoding: "pcm_int16",
    sample_rate: 16_000, audio: bytesToBase64(buffer),
  };
  if (requestId) payload.request_id = requestId;
  const json = JSON.stringify(payload);
  if (state.backend === "qwen3_asr" && state.firstAudioSentAt == null) state.firstAudioSentAt = performance.now();
  state.sentBytes += bytes.byteLength;
  state.ws.send(json);
}

async function setupMicrophone() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    throw new Error("secure_context_or_audio_worklet_required");
  }
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: { ideal: 1 }, sampleRate: { ideal: 16_000 },
      echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    },
  });
  const trackSettings = state.stream.getAudioTracks()[0]?.getSettings?.() ?? {};
  state.micSettings = Object.fromEntries(
    ["sampleRate", "channelCount", "latency", "echoCancellation", "noiseSuppression", "autoGainControl"]
      .filter((key) => trackSettings[key] != null)
      .map((key) => [key, trackSettings[key]]),
  );
  try { state.context = new AudioContext({ sampleRate: 16_000 }); }
  catch { state.context = new AudioContext(); }
  await state.context.audioWorklet.addModule("/asr/benchmark/audio-worklet.mjs");
  state.sourceNode = state.context.createMediaStreamSource(state.stream);
  state.workletNode = new AudioWorkletNode(state.context, "asr-benchmark-microphone", {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
  });
  state.zeroGain = state.context.createGain();
  state.zeroGain.gain.value = 0;
  state.workletNode.port.onmessage = (event) => {
    const message = event.data;
    if (message?.type === "pcm_chunk") receivePcmChunk(message.pcm, message.sample_count);
    if (message?.type === "flushed") state.workletResolver?.();
  };
  state.workletNode.onprocessorerror = () => {
    state.errors.push("audio_worklet_error");
    showNotice("Falló el procesamiento local del micrófono.", "error");
    void stopRun();
  };
  state.sourceNode.connect(state.workletNode);
  state.workletNode.connect(state.zeroGain);
  state.zeroGain.connect(state.context.destination);
  await state.context.resume();
}

function activateCapture() {
  state.startedAt = performance.now();
  state.isRecording = true;
}

async function startRun() {
  const input = $("api-token");
  const token = input.value || state.token;
  if (!token) { showNotice("Escribe el ASR API token para iniciar.", "error"); input.focus(); return; }
  if (!window.isSecureContext) { showNotice("El micrófono requiere HTTPS (o localhost).", "error"); return; }
  resetRun();
  state.token = token;
  state.reference = $("reference-text").value;
  state.runTimestamp = new Date().toISOString();
  state.runPrefix = crypto.randomUUID().replaceAll("-", "");
  state.firstAudioSentAt = null;
  state.readinessAtStart = null;
  state.effectiveMaxBacklogMs = null;
  state.qwenStartMetrics = null;
  state.isStopping = false;
  state.finalised = false;
  showNotice("Comprobando readiness y solicitando permiso del micrófono…");
  $("start-button").disabled = true;
  try {
    const first = await getTelemetry();
    recordTelemetry(first);
    if (!first.ready) throw new Error("model_not_ready");
    state.backend = first.backend;
    state.modelId = first.model_id;
    state.modelRevision = first.model_revision ?? null;
    state.workers = first.workers;
    await openSocket(token);
    state.token = token;
    await setupMicrophone();
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) throw new Error("websocket_closed");
    if (state.backend === "qwen3_asr") await startQwenStream();
    activateCapture();
    $("stop-button").disabled = false;
    $("disconnect-button").disabled = false;
    startRttSampler();
    state.timer = setInterval(() => { updateTimer(); void pollTelemetry(); }, 1000);
    document.addEventListener("visibilitychange", onVisibilityChange);
    showNotice(
      state.backend === "qwen3_asr"
        ? "Grabando. Los parciales son candidatos reemplazables; el transcript final llega al detener."
        : "Grabando. Habla con naturalidad; se mostrarán segmentos finales.",
      "ok",
    );
    status($("model-status"), "Modelo", first.ready ? "listo" : "no listo", first.ready ? "ready" : "unknown");
  } catch (error) {
    const reason = error?.message === "model_not_ready" ? "El modelo todavía no está listo." :
      error?.message === "unauthorized" ? "El token ASR no fue autorizado." :
        error?.message === "secure_context_or_audio_worklet_required" ? "Este navegador requiere HTTPS y soporte de AudioWorklet." :
          "No se pudo iniciar. Revisa permiso del micrófono y conectividad.";
    showNotice(reason, "error");
    await cleanup(false);
    $("start-button").disabled = false;
  }
}

function onVisibilityChange() {
  if (state.startedAt != null) state.visibility.push({
    elapsed_ms: Math.max(0, performance.now() - state.startedAt),
    visibility: document.visibilityState,
  });
}

function waitForWorkletFlush() {
  if (!state.workletNode) return Promise.resolve();
  return new Promise((resolve, reject) => {
    state.workletResolver = resolve;
    state.workletNode.port.postMessage({ type: "flush" });
    setTimeout(() => {
      if (state.workletResolver === resolve) {
        state.workletResolver = null;
        reject(new Error("audio_worklet_flush_timeout"));
      }
    }, 3000);
  });
}

function waitForFlushBarrier() {
  return new Promise((resolve, reject) => {
    state.flushResolver = resolve;
    setTimeout(() => {
      if (state.flushResolver === resolve) {
        state.flushResolver = null;
        reject(new Error("flush_timeout"));
      }
    }, 120_000);
  });
}

async function stopRun() {
  if (!state.ws || state.isStopping || state.finalised) return;
  state.isStopping = true;
  stopRttSampler();
  $("stop-button").disabled = true;
  showNotice("Cerrando el último bloque y esperando flush_complete…");
  try {
    await waitForWorkletFlush();
    state.isRecording = false;
    const barrier = waitForFlushBarrier();
    const requestId = crypto.randomUUID();
    state.flushId = requestId;
    if (state.backend === "qwen3_asr") state.clientEosAt = performance.now();
    state.ws.send(JSON.stringify({ event: "flush", source: "mic", request_id: requestId }));
    await barrier;
    showNotice("Sesión terminada. El audio no se conserva.", "ok");
  } catch {
    state.isRecording = false;
    state.errors.push("flush_or_connection_error");
    showNotice("No se confirmó el flush del servidor; se libera el micrófono y los resultados quedan parciales.", "error");
  }
  await cleanup(true);
}

async function cleanup(complete) {
  state.isRecording = false;
  state.isStopping = false;
  state.finalised = true;
  stopRttSampler();
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  document.removeEventListener("visibilitychange", onVisibilityChange);
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  try { state.sourceNode?.disconnect(); } catch {}
  try { state.workletNode?.disconnect(); } catch {}
  try { state.zeroGain?.disconnect(); } catch {}
  state.sourceNode = state.workletNode = state.zeroGain = null;
  if (state.context && state.context.state !== "closed") await state.context.close().catch(() => {});
  state.context = null;
  if (state.ws && state.ws.readyState < WebSocket.CLOSING) state.ws.close(1000);
  state.ws = null;
  state.requestTimes.clear();
  state.activeSegmentRequestId = null;
  if (state.streamStartResolver) {
    state.streamStartResolver.reject(new Error("stream_closed"));
    state.streamStartResolver = null;
  }
  $("stop-button").disabled = true;
  $("disconnect-button").disabled = true;
  $("start-button").disabled = false;
  $("export-json").disabled = state.segments.length === 0 && state.errors.length === 0;
  $("export-csv").disabled = state.segments.length === 0;
  updateTimer();
  updateSummary();
  renderSegments();
  renderCharts();
  if (complete) state.health = { ...(state.health || {}), finished_at: new Date().toISOString() };
}

function resetRun() {
  stopRttSampler();
  state.segments = [];
  state.errors = [];
  state.telemetry = [];
  state.rttSamples = [];
  state.requestTimes.clear();
  state.segmentRequestIds = [];
  state.vad = new ShadowVad();
  state.chunkSequence = 0;
  state.capturedSamples = 0;
  state.sentBytes = 0;
  state.visibility = [];
  state.health = null;
  state.micSettings = null;
  state.activeSegmentRequestId = null;
  state.startedAt = null;
  state.runTimestamp = null;
  state.backend = null;
  state.modelId = null;
  state.modelRevision = null;
  state.workers = null;
  state.streamId = null;
  state.qwenPublicStreamId = null;
  state.qwenLocalStreamId = null;
  state.qwenEpochObservability = null;
  state.lastQwenLifecycleError = null;
  state.streamStartRequestId = null;
  state.streamEvents = [];
  state.lastPartialProducerKey = null;
  state.lastPartialRevision = 0;
  state.lastPartialAt = null;
  state.clientEosAt = null;
  state.clientFinalizationMs = null;
  state.qwenChunkMs = 1000;
  state.effectiveMaxBacklogMs = null;
  state.firstAudioSentAt = null;
  state.readinessAtStart = null;
  state.qwenStartMetrics = null;
  state.qwenLanguage = $("qwen-language").value;
  state.qwenContext = "";
  state.finalised = false;
  $("transcript").textContent = "La transcripción aparecerá cuando el servidor cierre un segmento.";
  renderQwenLifecycle();
  renderQwenPartials();
  $("export-json").disabled = true;
  $("export-csv").disabled = true;
  renderSegments();
  updateSummary();
  renderCharts();
}

function safeResult() {
  const orderedRows = sortedSegments();
  const isQwen = state.backend === "qwen3_asr";
  const reference = state.reference.trim();
  const candidateText = transcriptText();
  const candidateWer = isQwen && reference ? wordErrorRate(reference, candidateText) : null;
  const productionWer = !isQwen && reference ? wordErrorRate(reference, candidateText) : null;
  const finalCandidateSource = [...orderedRows].reverse().find((row) => row.candidate_only) ?? null;
  const segments = orderedRows.map((row) => ({
    event: row.event, candidate_only: row.candidate_only, truth_status: row.truth_status, provisional: row.provisional,
    text: row.text, start: row.start, end: row.end, speaker: row.speaker, backend: row.backend,
    stream_id: row.stream_id, revision: row.revision, audio_cursor_ms: row.audio_cursor_ms,
    FINAL_CANDIDATE_WER: row === finalCandidateSource ? candidateWer : null,
    AUDIO_DURATION_MS: row.audio_duration_ms,
    SERVER_ENDPOINTING_MS: row.SERVER_ENDPOINTING_MS,
    SERVER_QUEUE_WAIT_MS: row.SERVER_QUEUE_WAIT_MS,
    SERVER_MODEL_INFERENCE_MS: row.SERVER_MODEL_INFERENCE_MS,
    SERVER_POSTPROCESS_MS: row.SERVER_POSTPROCESS_MS,
    SERVER_EOS_TO_TRANSCRIPT_MS: row.SERVER_EOS_TO_TRANSCRIPT_MS,
    SERVER_EOS_TO_FINAL_CANDIDATE_MS: row.SERVER_EOS_TO_FINAL_CANDIDATE_MS,
    CLIENT_EOS_TO_TRANSCRIPT_MS: row.CLIENT_EOS_TO_TRANSCRIPT_MS,
    CLIENT_EOS_TO_FINAL_CANDIDATE_MS: row.CLIENT_EOS_TO_FINAL_CANDIDATE_MS,
    MODEL_INFERENCE_MS: row.MODEL_INFERENCE_MS,
    SEGMENT_WAIT_MS: row.SEGMENT_WAIT_MS,
    SERVER_TO_TRANSCRIPT_MS: row.SERVER_TO_TRANSCRIPT_MS,
    SERVER_RECEIVE_TO_TRANSCRIPT_MS: row.SERVER_RECEIVE_TO_TRANSCRIPT_MS,
    SERVER_AUDIO_END_TO_TRANSCRIPT_MS: row.SERVER_AUDIO_END_TO_TRANSCRIPT_MS,
    CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: row.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS,
    FINALIZATION_AFTER_SERVER_EOS_MS: row.FINALIZATION_AFTER_SERVER_EOS_MS,
    QWEN_CUMULATIVE_DECODE_WALL_MS: row.QWEN_CUMULATIVE_DECODE_WALL_MS,
    QWEN_STREAM_RTF: row.QWEN_STREAM_RTF,
    QWEN_SCHEDULER_WAIT_MS: row.QWEN_SCHEDULER_WAIT_MS,
    QWEN_SCHEDULER_WAIT_P50_MS: row.QWEN_SCHEDULER_WAIT_P50_MS,
    QWEN_SCHEDULER_WAIT_P95_MS: row.QWEN_SCHEDULER_WAIT_P95_MS,
    QWEN_DECODE_WALL_P50_MS: row.QWEN_DECODE_WALL_P50_MS,
    QWEN_DECODE_WALL_P95_MS: row.QWEN_DECODE_WALL_P95_MS,
    QWEN_DECODE_BACKLOG_MS: row.QWEN_DECODE_BACKLOG_MS,
    QWEN_DECODE_BACKLOG_MAX_MS: row.QWEN_DECODE_BACKLOG_MAX_MS,
    STREAM_LAG_MS: row.STREAM_LAG_MS,
    STREAM_LAG_MAX_MS: row.STREAM_LAG_MAX_MS,
    PENDING_DECODE_COUNT: row.PENDING_DECODE_COUNT,
    ACTIVE_STREAM_COUNT: row.ACTIVE_STREAM_COUNT,
    DECODE_OVERRUN: row.DECODE_OVERRUN,
    QWEN_DECODE_STEPS_DELTA: row.QWEN_DECODE_STEPS_DELTA,
    QWEN_DECODE_STEPS_DELTA_TOTAL: row.QWEN_DECODE_STEPS_DELTA_TOTAL,
    QWEN_FINISH_DECODE_WALL_MS: row.QWEN_FINISH_DECODE_WALL_MS,
    PARTIAL_COUNT: row.PARTIAL_COUNT,
    FIRST_PARTIAL_MS: row.FIRST_PARTIAL_MS,
    PARTIAL_REVISION_RATE: row.PARTIAL_REVISION_RATE,
    PARTIAL_STABILITY: row.PARTIAL_STABILITY,
    MODEL_DECODE_CHUNK_MS: row.MODEL_DECODE_CHUNK_MS,
    AUDIO_PUSH_INTERVAL_MS: row.AUDIO_PUSH_INTERVAL_MS,
    queue_wait_ms: row.queue_wait_ms, audio_duration_ms: row.audio_duration_ms,
  }));
  const inference = segments.map((row) => row.MODEL_INFERENCE_MS).filter(Number.isFinite);
  const endpoint = segments.map((row) => row.SERVER_ENDPOINTING_MS).filter(Number.isFinite);
  const queue = segments.map((row) => row.SERVER_QUEUE_WAIT_MS).filter(Number.isFinite);
  const postprocess = segments.map((row) => row.SERVER_POSTPROCESS_MS).filter(Number.isFinite);
  const serverEos = segments.map((row) => row.SERVER_AUDIO_END_TO_TRANSCRIPT_MS).filter(Number.isFinite);
  const clientEos = segments.map((row) => row.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS).filter(Number.isFinite);
  const finalCandidate = [...segments].reverse().find((row) => row.candidate_only) ?? null;
  const rtt = rttPercentiles(state.rttSamples);
  const audioMs = segments.map((row) => row.audio_duration_ms).filter(Number.isFinite).reduce((sum, value) => sum + value, 0);
  const inferenceMs = inference.reduce((sum, value) => sum + value, 0);
  const lastTelemetry = state.telemetry.at(-1);
  const result = {
    timestamp: state.runTimestamp,
    finished_at: new Date().toISOString(),
    backend: state.backend,
    production_backend: state.health?.production_backend ?? null,
    model_id: state.modelId,
    model_revision: state.modelRevision,
    runtime_provenance: state.runtimeProvenance,
    startup_evidence: safeQwenStartupEvidence({
      readinessAtStart: state.readinessAtStart,
      qwenStartMetrics: state.qwenStartMetrics,
      health: state.health,
    }),
    gpu: lastTelemetry?.gpu_device ?? null,
    GPU_COMPUTE_AVAILABLE: lastTelemetry?.gpu_compute_available === true,
    GPU_TELEMETRY_AVAILABLE: lastTelemetry?.gpu_telemetry_available === true,
    gpu_compute: lastTelemetry?.gpu_compute ?? null,
    gpu_telemetry: lastTelemetry ? {
      available: lastTelemetry.gpu_telemetry_available,
      provider: lastTelemetry.gpu_telemetry_provider,
      scope: "device_global",
      backend_attributed: false,
      device: lastTelemetry.gpu_device,
      utilization_pct: lastTelemetry.gpu_utilization_pct,
      vram_used_mib: lastTelemetry.vram_used_mib,
      vram_total_mib: lastTelemetry.vram_total_mib,
      temperature_c: lastTelemetry.gpu_temperature_c,
      power_w: lastTelemetry.gpu_power_w,
    } : null,
    workers: state.workers,
    capture: {
      sample_rate_hz: 16_000,
      channels: 1,
      encoding: "PCM16LE",
      chunk_ms: 100,
      captured_audio_ms: state.capturedSamples * 1000 / 16_000,
      sent_audio_bytes: state.sentBytes,
      microphone_track_settings: state.micSettings,
    },
    transcript_mode: state.backend === "qwen3_asr" ? "STREAMING_PARTIALS" : "FINAL_SEGMENT",
    streaming: state.backend === "qwen3_asr" ? {
      streaming_class: "accumulated-audio-pseudostreaming",
      public_stream_id: state.qwenPublicStreamId ?? state.streamId,
      qwen_local_stream_id: state.qwenLocalStreamId,
      rolling_epoch: state.qwenEpochObservability ? {
        current_epoch: state.qwenEpochObservability.current_epoch,
        last_transition: state.qwenEpochObservability.last_transition,
        logical_cumulative: state.qwenEpochObservability.logical_cumulative,
        QWEN_AUDIO_ACCUM_MS: state.qwenEpochObservability.QWEN_AUDIO_ACCUM_MS ?? null,
        soft_rollover_policy: state.qwenEpochObservability.soft_rollover_policy ?? null,
      } : null,
      epoch_history: state.qwenEpochObservability?.epoch_history ?? [],
      transition_history: state.qwenEpochObservability?.transition_history ?? [],
      lifecycle_error: state.lastQwenLifecycleError,
      QWEN_AUDIO_ACCUM_MS: state.qwenEpochObservability?.QWEN_AUDIO_ACCUM_MS ?? null,
      audio_push_interval_ms: AUDIO_PUSH_INTERVAL_MS,
      model_chunk_ms: state.qwenChunkMs,
      model_decode_chunk_ms: state.qwenChunkMs,
      effective_max_backlog_ms: state.effectiveMaxBacklogMs,
      language: state.qwenLanguage,
      context_supplied: Boolean(state.qwenContext),
      partials: [...state.streamEvents],
      metrics: {
        partial_count: state.streamEvents.length,
        client_first_partial_ms: state.streamEvents[0]?.CLIENT_FIRST_PARTIAL_MS ?? null,
        server_first_partial_ms: state.streamEvents[0]?.FIRST_PARTIAL_MS ?? null,
        client_partial_interval_ms_median: median(state.streamEvents.map((item) => item.CLIENT_PARTIAL_UPDATE_INTERVAL_MS)),
        latest_partial_stability: state.streamEvents.at(-1)?.PARTIAL_STABILITY ?? null,
        latest_partial_revision_rate: state.streamEvents.at(-1)?.PARTIAL_REVISION_RATE ?? null,
        finalization_after_server_eos_ms: segments.at(-1)?.FINALIZATION_AFTER_SERVER_EOS_MS ?? null,
        finalization_after_client_eos_ms: segments.at(-1)?.CLIENT_EOS_TO_FINAL_CANDIDATE_MS
          ?? segments.at(-1)?.CLIENT_EOS_TO_TRANSCRIPT_MS ?? null,
        cumulative_decode_call_wall_ms: segments.at(-1)?.QWEN_CUMULATIVE_DECODE_WALL_MS ?? null,
        decode_wall_rtf: segments.at(-1)?.QWEN_STREAM_RTF ?? null,
        scheduler_wait_ms_p50: state.health?.qwen_scheduler?.qwen_scheduler_wait_p50_ms ?? null,
        scheduler_wait_ms_p95: state.health?.qwen_scheduler?.qwen_scheduler_wait_p95_ms ?? null,
        decode_wall_ms_p50: state.health?.qwen_scheduler?.qwen_decode_wall_p50_ms ?? null,
        decode_wall_ms_p95: state.health?.qwen_scheduler?.qwen_decode_wall_p95_ms ?? null,
        scheduler_backlog_audio_ms: state.health?.qwen_scheduler?.qwen_scheduler_backlog_ms ?? null,
        scheduler_max_stream_backlog_audio_ms: state.health?.qwen_scheduler?.qwen_scheduler_max_stream_backlog_ms ?? null,
        pending_decode_count: state.health?.qwen_scheduler?.pending_decode_count ?? null,
        active_decode_count: state.health?.qwen_scheduler?.active_decode_count ?? null,
        active_stream_count: state.health?.qwen_scheduler?.active_stream_count ?? null,
        decode_budget_overrun_count: state.health?.qwen_scheduler?.qwen_decode_budget_overrun_total ?? null,
        scheduler_fence_overrun_count: state.health?.qwen_scheduler?.qwen_scheduler_overrun_total ?? null,
        QWEN_AUDIO_ACCUM_MS: state.qwenEpochObservability?.QWEN_AUDIO_ACCUM_MS ?? null,
        EPOCH_HANDOFF_WALL_MS: state.qwenEpochObservability?.last_transition?.EPOCH_HANDOFF_WALL_MS ?? null,
        EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS: state.qwenEpochObservability?.last_transition?.EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS ?? null,
        EPOCH_ROLLOVER_COUNT: state.qwenEpochObservability?.logical_cumulative?.EPOCH_ROLLOVER_COUNT ?? null,
      },
      definitions: {
        PARTIAL_STABILITY: "Identical exact-prefix tokens in prior full text / prior token count; appended suffix tokens do not change this ratio.",
        PARTIAL_REVISION_RATE: "1 - PARTIAL_STABILITY; measures changed or removed prior tokens, excluding appended tokens.",
        QWEN_CUMULATIVE_DECODE_WALL_MS: "Sum of Qwen wrapper decode-call wall durations; not GPU kernel time.",
      QWEN_STREAM_RTF: "Cumulative decode-call wall duration / audio duration; not GPU utilization or end-to-end RTF.",
      QWEN_SCHEDULER_WAIT_MS: "Time from a decode job becoming ready until the single worker RPC starts.",
      QWEN_DECODE_BACKLOG_MS: "Accepted audio not yet handed to the worker, including queued windows and residual tail.",
      STREAM_LAG_MS: "Accepted audio cursor minus last successfully dispatched audio cursor; includes active and pending work.",
      DECODE_OVERRUN: "Decode-call wall exceeded the configured audio chunk interval; PCM is retained until bounded backpressure fences this stream.",
      },
    } : null,
    reference_text: state.reference || null,
    transcript: isQwen ? null : transcriptText(),
    final_candidate: isQwen ? transcriptText() : null,
    segments,
    summary: {
      segment_count: segments.length,
      error_count: state.errors.length,
      errors: [...state.errors],
      transcribed_audio_ms: audioMs,
      MODEL_INFERENCE_RTF: state.backend === "qwen3_asr" || audioMs <= 0 ? null : inferenceMs / audioMs,
      SERVER_MODEL_INFERENCE_MS_P50: median(inference),
      SERVER_MODEL_INFERENCE_MS_P95: percentile(inference, 0.95),
      SERVER_ENDPOINTING_MS_P50: median(endpoint),
      SERVER_ENDPOINTING_MS_P95: percentile(endpoint, 0.95),
      SERVER_QUEUE_WAIT_MS_P50: median(queue),
      SERVER_QUEUE_WAIT_MS_P95: percentile(queue, 0.95),
      SERVER_POSTPROCESS_MS_P50: median(postprocess),
      SERVER_POSTPROCESS_MS_P95: percentile(postprocess, 0.95),
      SERVER_EOS_TO_TRANSCRIPT_MS_P50: median(serverEos),
      SERVER_EOS_TO_TRANSCRIPT_MS_P95: percentile(serverEos, 0.95),
      CLIENT_EOS_TO_TRANSCRIPT_MS_P50: median(clientEos),
      CLIENT_EOS_TO_TRANSCRIPT_MS_P95: percentile(clientEos, 0.95),
      PROXY_WS_RTT_MS_P50: rtt.p50,
      PROXY_WS_RTT_MS_P95: rtt.p95,
      MODEL_INFERENCE_MS_P50: median(inference),
      MODEL_INFERENCE_MS_P95: percentile(inference, 0.95),
      SERVER_AUDIO_END_TO_TRANSCRIPT_MS_P50: median(serverEos),
      SERVER_AUDIO_END_TO_TRANSCRIPT_MS_P95: percentile(serverEos, 0.95),
      CLIENT_AUDIO_END_TO_TRANSCRIPT_MS_P50: median(clientEos),
      CLIENT_AUDIO_END_TO_TRANSCRIPT_MS_P95: percentile(clientEos, 0.95),
      WER: productionWer,
      FINAL_WER: productionWer,
      FINAL_CANDIDATE_WER: candidateWer,
      SERVER_EOS_TO_FINAL_CANDIDATE_MS: isQwen ? finalCandidate?.SERVER_EOS_TO_FINAL_CANDIDATE_MS ?? null : null,
      CLIENT_EOS_TO_FINAL_CANDIDATE_MS: isQwen ? finalCandidate?.CLIENT_EOS_TO_FINAL_CANDIDATE_MS ?? null : null,
      FIRST_PARTIAL_MS: state.streamEvents[0]?.FIRST_PARTIAL_MS ?? null,
      CLIENT_FIRST_PARTIAL_MS: state.streamEvents[0]?.CLIENT_FIRST_PARTIAL_MS ?? null,
      PARTIAL_COUNT: state.streamEvents.length,
      PARTIAL_UPDATE_INTERVAL_MS_P50: median(state.streamEvents.map((item) => item.CLIENT_PARTIAL_UPDATE_INTERVAL_MS)),
      PARTIAL_STABILITY: state.streamEvents.at(-1)?.PARTIAL_STABILITY ?? null,
      PARTIAL_REVISION_RATE: state.streamEvents.at(-1)?.PARTIAL_REVISION_RATE ?? null,
      FINALIZATION_AFTER_SERVER_EOS_MS: segments.at(-1)?.FINALIZATION_AFTER_SERVER_EOS_MS ?? null,
      FINALIZATION_AFTER_CLIENT_EOS_MS: segments.at(-1)?.CLIENT_EOS_TO_FINAL_CANDIDATE_MS
        ?? segments.at(-1)?.CLIENT_EOS_TO_TRANSCRIPT_MS ?? null,
      QWEN_CUMULATIVE_DECODE_WALL_MS: segments.at(-1)?.QWEN_CUMULATIVE_DECODE_WALL_MS ?? null,
      QWEN_STREAM_RTF: segments.at(-1)?.QWEN_STREAM_RTF ?? null,
      QWEN_SCHEDULER_WAIT_MS_P50: state.health?.qwen_scheduler?.qwen_scheduler_wait_p50_ms ?? null,
      QWEN_SCHEDULER_WAIT_MS_P95: state.health?.qwen_scheduler?.qwen_scheduler_wait_p95_ms ?? null,
      QWEN_DECODE_WALL_P50_MS: state.health?.qwen_scheduler?.qwen_decode_wall_p50_ms ?? null,
      QWEN_DECODE_WALL_P95_MS: state.health?.qwen_scheduler?.qwen_decode_wall_p95_ms ?? null,
      QWEN_DECODE_BACKLOG_MS: state.health?.qwen_scheduler?.qwen_scheduler_backlog_ms ?? null,
      QWEN_MAX_STREAM_BACKLOG_MS: state.health?.qwen_scheduler?.qwen_scheduler_max_stream_backlog_ms ?? null,
      PENDING_DECODE_COUNT: state.health?.qwen_scheduler?.pending_decode_count ?? null,
      ACTIVE_DECODE_COUNT: state.health?.qwen_scheduler?.active_decode_count ?? null,
      ACTIVE_STREAM_COUNT: state.health?.qwen_scheduler?.active_stream_count ?? null,
      DECODE_BUDGET_OVERRUN_COUNT: state.health?.qwen_scheduler?.qwen_decode_budget_overrun_total ?? null,
      SCHEDULER_FENCE_OVERRUN_COUNT: state.health?.qwen_scheduler?.qwen_scheduler_overrun_total ?? null,
    },
    telemetry: [...state.telemetry],
    proxy_ws_rtt_samples: state.rttSamples.map((sample) => ({
      elapsed_ms: sample.elapsed_ms,
      PROXY_WS_RTT_MS: sample.PROXY_WS_RTT_MS,
      status: sample.status,
    })),
    metric_definitions: {
      ...METRIC_DEFINITIONS,
      SERVER_EOS_TO_FINAL_CANDIDATE_MS: "Qwen server perf_counter from receipt of client EOS to final candidate ready; experimental and not a confirmed transcript.",
      CLIENT_EOS_TO_FINAL_CANDIDATE_MS: "Browser performance.now from shadow-VAD/client EOS to final candidate WebSocket receipt; candidate-only and includes proxy/network/browser scheduling.",
    },
    measurement_provenance: {
      MODEL_INFERENCE_SCOPE: "adapter_wall_clock_not_gpu_kernel_time",
      QWEN_DECODE_WALL_SCOPE: "Qwen wrapper decode-call wall time; serialized server processing, not GPU kernel time.",
      QWEN_SERVER_CHUNK_TO_PARTIAL_SCOPE: "ASR process perf_counter from RPC push dispatch through full replacement partial availability.",
      SEMANTIC_LEAD_TIME_MS: "Not computed here; requires correlating a useful JEV hypothesis with client/server EOS.",
      GPU_TELEMETRY_SCOPE: "optional_nvml_or_torch_device_global_not_backend_attributed",
      SERVER_CLOCK: "time.perf_counter within ASR server process",
      CLIENT_CLOCK: "performance.now within browser only",
      QWEN_AUDIO_ACCUM_MS: "Only exact read-only version-coupled Qwen state measurement is accepted; null means unavailable and is never inferred from cursor, backlog, or lag.",
      EPOCH_HANDOFF_WALL_MS: "Server monotonic time from rollover begin until successor ACTIVE after required replay and transition catch-up admission.",
      EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS: "Server monotonic time from rollover begin until the first non-empty successor partial after post-cut PRIMARY processing.",
    },
    visibility_transitions: [...state.visibility],
  };
  return result;
}

function download(filename, contents, type) {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportJson() {
  const result = safeResult();
  download(`asr-benchmark-${new Date().toISOString().replaceAll(":", "-")}.json`, JSON.stringify(result, null, 2), "application/json");
}

function exportCsv() {
  const result = safeResult();
  const columns = ["record_type", "event", "candidate_only", "provisional", "truth_status", "timestamp", "backend", "model_id", "gpu", "workers",
    "PROXY_WS_RTT_MS_P50", "PROXY_WS_RTT_MS_P95", "elapsed_ms", "start", "end", "text",
    "AUDIO_DURATION_MS", "SERVER_ENDPOINTING_MS", "SERVER_QUEUE_WAIT_MS", "SERVER_MODEL_INFERENCE_MS",
    "SERVER_POSTPROCESS_MS", "SERVER_EOS_TO_TRANSCRIPT_MS", "CLIENT_EOS_TO_TRANSCRIPT_MS",
    "SERVER_EOS_TO_FINAL_CANDIDATE_MS", "CLIENT_EOS_TO_FINAL_CANDIDATE_MS", "FINAL_CANDIDATE_WER",
    "PROXY_WS_RTT_MS", "MODEL_INFERENCE_MS", "SEGMENT_WAIT_MS", "SERVER_TO_TRANSCRIPT_MS",
    "SERVER_RECEIVE_TO_TRANSCRIPT_MS", "SERVER_AUDIO_END_TO_TRANSCRIPT_MS",
    "CLIENT_AUDIO_END_TO_TRANSCRIPT_MS", "queue_wait_ms", "audio_duration_ms",
    "stream_id", "revision", "audio_cursor_ms", "CLIENT_ELAPSED_MS",
    "CLIENT_FIRST_PARTIAL_MS", "CLIENT_PARTIAL_UPDATE_INTERVAL_MS", "FIRST_PARTIAL_MS",
    "SERVER_CHUNK_TO_PARTIAL_MS", "PARTIAL_COUNT", "PARTIAL_REVISION_RATE", "PARTIAL_STABILITY",
    "FINALIZATION_AFTER_SERVER_EOS_MS", "QWEN_DECODE_CALL_WALL_MS",
    "QWEN_CUMULATIVE_DECODE_WALL_MS", "QWEN_STREAM_RTF",
    "QWEN_SCHEDULER_WAIT_MS", "QWEN_SCHEDULER_WAIT_P50_MS", "QWEN_SCHEDULER_WAIT_P95_MS",
    "QWEN_DECODE_WALL_P50_MS", "QWEN_DECODE_WALL_P95_MS", "QWEN_DECODE_BACKLOG_MS",
    "QWEN_DECODE_BACKLOG_MAX_MS", "STREAM_LAG_MAX_MS",
    "STREAM_LAG_MS", "PENDING_DECODE_COUNT", "ACTIVE_DECODE_COUNT", "ACTIVE_STREAM_COUNT",
    "DECODE_OVERRUN", "DECODE_BUDGET_OVERRUN_COUNT", "SCHEDULER_FENCE_OVERRUN_COUNT",
    "QWEN_DECODE_STEPS_DELTA", "QWEN_DECODE_STEPS_DELTA_TOTAL", "QWEN_FINISH_DECODE_WALL_MS",
    "QWEN_ACTIVE_STREAM_COUNT", "QWEN_MAX_ACTIVE_STREAMS", "QWEN_ACTIVE_DECODE_COUNT",
    "QWEN_PENDING_DECODE_COUNT", "QWEN_SCHEDULER_BACKLOG_MS", "QWEN_MAX_STREAM_BACKLOG_MS",
    "QWEN_STREAM_LAG_SUM_MS", "QWEN_MAX_STREAM_LAG_MS", "QWEN_DECODE_BUDGET_OVERRUN_TOTAL",
    "QWEN_SCHEDULER_FENCE_OVERRUN_TOTAL", "QWEN_SCHEDULER_WAIT_ROLLING_P95_MS",
    "QWEN_DECODE_WALL_ROLLING_P95_MS", "gpu_utilization_pct", "vram_used_mib",
    "vram_total_mib", "process_rss_mib",
    "MODEL_DECODE_CHUNK_MS", "AUDIO_PUSH_INTERVAL_MS",
    "qwen_asr_version", "vllm_version", "transformers_version", "torch_version", "torch_cuda_version"];
  const toCsvRow = (row) => columns.map((column) => csvCell(row[column] ?? "")).join(",");
  const lines = [toCsvRow(Object.fromEntries(columns.map((column) => [column, column])))];
  for (const row of result.segments) {
    lines.push(toCsvRow({
      record_type: row.candidate_only ? "final_candidate" : "segment",
      event: row.event, candidate_only: row.candidate_only, provisional: row.provisional,
      truth_status: row.truth_status, timestamp: result.timestamp, backend: result.backend,
      model_id: result.model_id, gpu: result.gpu, workers: result.workers,
      PROXY_WS_RTT_MS_P50: result.summary.PROXY_WS_RTT_MS_P50,
      PROXY_WS_RTT_MS_P95: result.summary.PROXY_WS_RTT_MS_P95,
      start: row.start, end: row.end, text: row.text,
      AUDIO_DURATION_MS: row.AUDIO_DURATION_MS,
      SERVER_ENDPOINTING_MS: row.SERVER_ENDPOINTING_MS,
      SERVER_QUEUE_WAIT_MS: row.SERVER_QUEUE_WAIT_MS,
      SERVER_MODEL_INFERENCE_MS: row.SERVER_MODEL_INFERENCE_MS,
      SERVER_POSTPROCESS_MS: row.SERVER_POSTPROCESS_MS,
      SERVER_EOS_TO_TRANSCRIPT_MS: row.SERVER_EOS_TO_TRANSCRIPT_MS,
      CLIENT_EOS_TO_TRANSCRIPT_MS: row.CLIENT_EOS_TO_TRANSCRIPT_MS,
      SERVER_EOS_TO_FINAL_CANDIDATE_MS: row.SERVER_EOS_TO_FINAL_CANDIDATE_MS,
      CLIENT_EOS_TO_FINAL_CANDIDATE_MS: row.CLIENT_EOS_TO_FINAL_CANDIDATE_MS,
      FINAL_CANDIDATE_WER: row.FINAL_CANDIDATE_WER,
      MODEL_INFERENCE_MS: row.MODEL_INFERENCE_MS,
      SEGMENT_WAIT_MS: row.SEGMENT_WAIT_MS,
      SERVER_TO_TRANSCRIPT_MS: row.SERVER_TO_TRANSCRIPT_MS,
      SERVER_RECEIVE_TO_TRANSCRIPT_MS: row.SERVER_RECEIVE_TO_TRANSCRIPT_MS,
      SERVER_AUDIO_END_TO_TRANSCRIPT_MS: row.SERVER_AUDIO_END_TO_TRANSCRIPT_MS,
      CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: row.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS,
      stream_id: row.stream_id, revision: row.revision,
      FINALIZATION_AFTER_SERVER_EOS_MS: row.FINALIZATION_AFTER_SERVER_EOS_MS,
      QWEN_CUMULATIVE_DECODE_WALL_MS: row.QWEN_CUMULATIVE_DECODE_WALL_MS,
      QWEN_STREAM_RTF: row.QWEN_STREAM_RTF, PARTIAL_COUNT: row.PARTIAL_COUNT,
      QWEN_SCHEDULER_WAIT_MS: row.QWEN_SCHEDULER_WAIT_MS,
      QWEN_SCHEDULER_WAIT_P50_MS: row.QWEN_SCHEDULER_WAIT_P50_MS,
      QWEN_SCHEDULER_WAIT_P95_MS: row.QWEN_SCHEDULER_WAIT_P95_MS,
      QWEN_DECODE_WALL_P50_MS: row.QWEN_DECODE_WALL_P50_MS,
      QWEN_DECODE_WALL_P95_MS: row.QWEN_DECODE_WALL_P95_MS,
      QWEN_DECODE_BACKLOG_MS: row.QWEN_DECODE_BACKLOG_MS,
      QWEN_DECODE_BACKLOG_MAX_MS: row.QWEN_DECODE_BACKLOG_MAX_MS,
      STREAM_LAG_MS: row.STREAM_LAG_MS,
      STREAM_LAG_MAX_MS: row.STREAM_LAG_MAX_MS,
      PENDING_DECODE_COUNT: row.PENDING_DECODE_COUNT,
      ACTIVE_STREAM_COUNT: row.ACTIVE_STREAM_COUNT,
      DECODE_OVERRUN: row.DECODE_OVERRUN,
      QWEN_DECODE_STEPS_DELTA: row.QWEN_DECODE_STEPS_DELTA,
      QWEN_DECODE_STEPS_DELTA_TOTAL: row.QWEN_DECODE_STEPS_DELTA_TOTAL,
      QWEN_FINISH_DECODE_WALL_MS: row.QWEN_FINISH_DECODE_WALL_MS,
      FIRST_PARTIAL_MS: row.FIRST_PARTIAL_MS, PARTIAL_STABILITY: row.PARTIAL_STABILITY,
      PARTIAL_REVISION_RATE: row.PARTIAL_REVISION_RATE,
      MODEL_DECODE_CHUNK_MS: row.MODEL_DECODE_CHUNK_MS,
      AUDIO_PUSH_INTERVAL_MS: row.AUDIO_PUSH_INTERVAL_MS,
      qwen_asr_version: result.runtime_provenance?.qwen_asr_version,
      vllm_version: result.runtime_provenance?.vllm_version,
      transformers_version: result.runtime_provenance?.transformers_version,
      torch_version: result.runtime_provenance?.torch_version,
      torch_cuda_version: result.runtime_provenance?.torch_cuda_version,
      queue_wait_ms: row.queue_wait_ms, audio_duration_ms: row.audio_duration_ms,
    }));
  }
  for (const partial of result.streaming?.partials ?? []) {
    lines.push(toCsvRow({
      record_type: partial.event ?? "partial_candidate", timestamp: result.timestamp, backend: result.backend,
      model_id: result.model_id, gpu: result.gpu, workers: result.workers,
      stream_id: partial.stream_id, revision: partial.revision, text: partial.text,
      audio_cursor_ms: partial.audio_cursor_ms, CLIENT_ELAPSED_MS: partial.CLIENT_ELAPSED_MS,
      CLIENT_FIRST_PARTIAL_MS: partial.CLIENT_FIRST_PARTIAL_MS,
      CLIENT_PARTIAL_UPDATE_INTERVAL_MS: partial.CLIENT_PARTIAL_UPDATE_INTERVAL_MS,
      FIRST_PARTIAL_MS: partial.FIRST_PARTIAL_MS,
      SERVER_CHUNK_TO_PARTIAL_MS: partial.SERVER_CHUNK_TO_PARTIAL_MS,
      PARTIAL_COUNT: partial.PARTIAL_COUNT,
      PARTIAL_REVISION_RATE: partial.PARTIAL_REVISION_RATE,
      PARTIAL_STABILITY: partial.PARTIAL_STABILITY,
      QWEN_DECODE_CALL_WALL_MS: partial.QWEN_DECODE_CALL_WALL_MS,
      QWEN_CUMULATIVE_DECODE_WALL_MS: partial.QWEN_CUMULATIVE_DECODE_WALL_MS,
      QWEN_STREAM_RTF: partial.QWEN_STREAM_RTF,
      QWEN_SCHEDULER_WAIT_MS: partial.QWEN_SCHEDULER_WAIT_MS,
      QWEN_DECODE_BACKLOG_MS: partial.QWEN_DECODE_BACKLOG_MS,
      QWEN_DECODE_BACKLOG_MAX_MS: partial.QWEN_DECODE_BACKLOG_MAX_MS,
      STREAM_LAG_MS: partial.STREAM_LAG_MS,
      STREAM_LAG_MAX_MS: partial.STREAM_LAG_MAX_MS,
      PENDING_DECODE_COUNT: partial.PENDING_DECODE_COUNT,
      ACTIVE_STREAM_COUNT: partial.ACTIVE_STREAM_COUNT,
      DECODE_OVERRUN: partial.DECODE_OVERRUN,
      MODEL_DECODE_CHUNK_MS: partial.MODEL_DECODE_CHUNK_MS,
      AUDIO_PUSH_INTERVAL_MS: partial.AUDIO_PUSH_INTERVAL_MS,
      qwen_asr_version: result.runtime_provenance?.qwen_asr_version,
      vllm_version: result.runtime_provenance?.vllm_version,
      transformers_version: result.runtime_provenance?.transformers_version,
      torch_version: result.runtime_provenance?.torch_version,
      torch_cuda_version: result.runtime_provenance?.torch_cuda_version,
    }));
  }
  for (const point of result.telemetry ?? []) {
    const scheduler = point.qwen_scheduler ?? {};
    lines.push(toCsvRow({
      record_type: "telemetry", timestamp: point.timestamp, backend: result.backend,
      model_id: result.model_id, gpu: point.gpu_device, workers: point.workers,
      QWEN_ACTIVE_STREAM_COUNT: scheduler.active_stream_count,
      QWEN_MAX_ACTIVE_STREAMS: scheduler.max_active_streams,
      QWEN_ACTIVE_DECODE_COUNT: scheduler.active_decode_count,
      QWEN_PENDING_DECODE_COUNT: scheduler.pending_decode_count,
      QWEN_SCHEDULER_BACKLOG_MS: scheduler.qwen_scheduler_backlog_ms,
      QWEN_MAX_STREAM_BACKLOG_MS: scheduler.qwen_scheduler_max_stream_backlog_ms,
      QWEN_STREAM_LAG_SUM_MS: scheduler.qwen_scheduler_stream_lag_ms,
      QWEN_MAX_STREAM_LAG_MS: scheduler.qwen_scheduler_max_stream_lag_ms,
      QWEN_SCHEDULER_WAIT_ROLLING_P95_MS: scheduler.qwen_scheduler_wait_p95_ms,
      QWEN_DECODE_WALL_ROLLING_P95_MS: scheduler.qwen_decode_wall_p95_ms,
      QWEN_DECODE_BUDGET_OVERRUN_TOTAL: scheduler.qwen_decode_budget_overrun_total,
      QWEN_SCHEDULER_FENCE_OVERRUN_TOTAL: scheduler.qwen_scheduler_overrun_total,
      gpu_utilization_pct: point.gpu_utilization_pct,
      vram_used_mib: point.vram_used_mib,
      vram_total_mib: point.vram_total_mib,
      process_rss_mib: point.process_rss_mib,
    }));
  }
  for (const sample of result.proxy_ws_rtt_samples) {
    lines.push(toCsvRow({
      record_type: "proxy_ws_rtt", timestamp: result.timestamp, backend: result.backend,
      model_id: result.model_id, gpu: result.gpu, workers: result.workers,
      PROXY_WS_RTT_MS_P50: result.summary.PROXY_WS_RTT_MS_P50,
      PROXY_WS_RTT_MS_P95: result.summary.PROXY_WS_RTT_MS_P95,
      elapsed_ms: sample.elapsed_ms, PROXY_WS_RTT_MS: sample.PROXY_WS_RTT_MS,
    }));
  }
  download(`asr-benchmark-${new Date().toISOString().replaceAll(":", "-")}.csv`, lines.join("\r\n"), "text/csv;charset=utf-8");
}

$("start-button").addEventListener("click", () => { void startRun(); });
$("stop-button").addEventListener("click", () => { void stopRun(); });
$("disconnect-button").addEventListener("click", () => { void stopRun(); });
$("export-json").addEventListener("click", exportJson);
$("export-csv").addEventListener("click", exportCsv);
$("reference-text").addEventListener("input", () => {
  state.reference = $("reference-text").value;
  updateSummary();
});
window.addEventListener("resize", renderCharts);
