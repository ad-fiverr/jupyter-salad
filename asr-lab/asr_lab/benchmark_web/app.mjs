import {
  METRIC_DEFINITIONS, ShadowVad, boundedPush, csvCell, formatMilliseconds, gpuStatusLabel,
  median, percentile, rttPercentiles, wordErrorRate,
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
};

const charts = {
  inference: $("chart-inference"), endpoint: $("chart-endpoint"), serverEos: $("chart-server-eos"),
  clientEos: $("chart-client-eos"), rtt: $("chart-rtt"), gpu: $("chart-gpu"),
  vram: $("chart-vram"), ram: $("chart-ram"),
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
  $("transcript").textContent = text || "La transcripción aparecerá cuando el servidor cierre un segmento.";
  const latest = state.telemetry.at(-1);
  const gpu = gpuStatusLabel(
    latest?.gpu_compute,
    { device: latest?.gpu_device, provider: latest?.gpu_telemetry_provider },
  );
  $("run-meta").textContent = [state.backend, state.modelId, `workers=${state.workers ?? "?"}`, gpu].filter(Boolean).join(" · ");
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
  const wer = state.reference.trim() ? wordErrorRate(state.reference, transcriptText()) : null;
  $("summary-segments").textContent = String(rows.length);
  $("summary-errors").textContent = String(state.errors.length);
  $("summary-audio").textContent = `${(state.capturedSamples / 16_000).toFixed(1)} s`;
  $("summary-rtf").textContent = audioMs > 0 ? (inferenceMs / audioMs).toFixed(3) : "—";
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
      fmtMs(Number.isFinite(item.SERVER_EOS_TO_TRANSCRIPT_MS) ? item.SERVER_EOS_TO_TRANSCRIPT_MS : item.SERVER_AUDIO_END_TO_TRANSCRIPT_MS),
      fmtMs(Number.isFinite(item.CLIENT_EOS_TO_TRANSCRIPT_MS) ? item.CLIENT_EOS_TO_TRANSCRIPT_MS : item.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS),
    ];
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value == null ? "" : String(value);
      tr.append(td);
    });
    body.append(tr);
  });
}

function drawChart(canvas, points, key, color = "#80e0b2", secondKey = null) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(260, Math.floor(rect.width * devicePixelRatio));
  const height = Math.max(125, Math.floor(rect.height * devicePixelRatio));
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
  ctx.font = `${11 * devicePixelRatio}px system-ui`;
  ctx.lineWidth = devicePixelRatio;
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
    const max = values.reduce((largest, value) => Math.max(largest, value), 1);
    ctx.fillText((max * (1 - i / 4)).toFixed(0), 3, y + 4 * devicePixelRatio);
  }
  const maxValue = values.reduce((largest, value) => Math.max(largest, value), 1);
  const lineColors = [color, "#f0c878"];
  series.forEach((items, seriesIndex) => {
    const actualPoints = points.map((point, index) => ({ x: index, y: point[seriesIndex === 0 ? key : secondKey] })).filter((point) => Number.isFinite(point.y));
    if (!actualPoints.length) return;
    ctx.strokeStyle = lineColors[seriesIndex];
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath();
    actualPoints.forEach((point, index) => {
      const x = margin.left + (points.length <= 1 ? 0 : point.x / (points.length - 1)) * plotW;
      const y = margin.top + plotH - (point.y / maxValue) * plotH;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
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
  };
  state.telemetry.push(point);
  if (state.telemetry.length > 3600) state.telemetry.shift();
  status($("model-status"), "Modelo", snapshot.ready ? "listo" : "cargando / no listo", snapshot.ready ? "ready" : "unknown");
  updateTranscript();
}

async function pollTelemetry() {
  if (!state.token || state.finalised) return;
  try {
    const snapshot = await getTelemetry();
    if (!state.token || state.finalised) return;
    recordTelemetry(snapshot);
    if (snapshot.backend) state.backend = snapshot.backend;
    if (snapshot.model_id) state.modelId = snapshot.model_id;
    if (snapshot.model_revision) state.modelRevision = snapshot.model_revision;
    if (Number.isInteger(snapshot.workers)) state.workers = snapshot.workers;
    updateTranscript();
    renderCharts();
  } catch (error) {
    if (error?.message === "unauthorized") showNotice("El token ASR no fue autorizado.", "error");
    else showNotice("No se pudo leer la telemetría. La captura y el WebSocket siguen separados.", "error");
  }
}

function onTranscript(message) {
  const receivedAt = performance.now();
  const clientStart = typeof message.client_request_id === "string"
    ? state.requestTimes.get(message.client_request_id)
    : null;
  if (typeof message.client_request_id === "string") state.requestTimes.delete(message.client_request_id);
  const clientEosMs = Number.isFinite(clientStart) ? Math.max(0, receivedAt - clientStart) : null;
  const modelMs = Number.isFinite(message.SERVER_MODEL_INFERENCE_MS)
    ? message.SERVER_MODEL_INFERENCE_MS
    : Number.isFinite(message.MODEL_INFERENCE_MS) ? message.MODEL_INFERENCE_MS : null;
  const queueMs = Number.isFinite(message.SERVER_QUEUE_WAIT_MS)
    ? message.SERVER_QUEUE_WAIT_MS
    : Number.isFinite(message.queue_wait_ms) ? message.queue_wait_ms : null;
  const serverEosMs = Number.isFinite(message.SERVER_EOS_TO_TRANSCRIPT_MS)
    ? message.SERVER_EOS_TO_TRANSCRIPT_MS
    : Number.isFinite(message.SERVER_AUDIO_END_TO_TRANSCRIPT_MS) ? message.SERVER_AUDIO_END_TO_TRANSCRIPT_MS : null;
  state.segments.push({
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
    CLIENT_EOS_TO_TRANSCRIPT_MS: clientEosMs,
    SEGMENT_WAIT_MS: Number.isFinite(message.SEGMENT_WAIT_MS) ? message.SEGMENT_WAIT_MS : null,
    SERVER_TO_TRANSCRIPT_MS: Number.isFinite(message.SERVER_TO_TRANSCRIPT_MS) ? message.SERVER_TO_TRANSCRIPT_MS : null,
    SERVER_RECEIVE_TO_TRANSCRIPT_MS: Number.isFinite(message.SERVER_RECEIVE_TO_TRANSCRIPT_MS) ? message.SERVER_RECEIVE_TO_TRANSCRIPT_MS : null,
    SERVER_AUDIO_END_TO_TRANSCRIPT_MS: serverEosMs,
    CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: clientEosMs,
    queue_wait_ms: queueMs,
    audio_duration_ms: Number.isFinite(message.AUDIO_DURATION_MS)
      ? message.AUDIO_DURATION_MS
      : Number.isFinite(message.audio_duration_ms) ? message.audio_duration_ms : null,
    request_id: typeof message.request_id === "string" ? message.request_id : null,
    client_request_id: typeof message.client_request_id === "string" ? message.client_request_id : null,
    receivedAt,
  });
  renderSegments();
  updateTranscript();
  updateSummary();
}

function onSocketMessage(raw) {
  let message;
  try { message = JSON.parse(raw); } catch { state.errors.push("invalid_server_message"); updateSummary(); return; }
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
  if (message.event === "transcript" || message.type === "transcript") {
    onTranscript(message);
    return;
  }
  if (message.event === "error") {
    const code = /^[a-z0-9_]{1,64}$/i.test(message.code ?? "") ? message.code : "asr_error";
    state.errors.push(code);
    showNotice(`Error ASR: ${code}`, "error");
    updateSummary();
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

function receivePcmChunk(buffer, sampleCount) {
  if ((!state.isRecording && !state.isStopping) || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.capturedSamples += sampleCount;
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
  state.startedAt = performance.now();
  state.isRecording = true;
  await state.context.resume();
}

async function startRun() {
  const input = $("api-token");
  const token = input.value;
  if (!token) { showNotice("Escribe el ASR API token para iniciar.", "error"); input.focus(); return; }
  if (!window.isSecureContext) { showNotice("El micrófono requiere HTTPS (o localhost).", "error"); return; }
  resetRun();
  state.token = token;
  state.reference = $("reference-text").value;
  input.value = "";
  state.runTimestamp = new Date().toISOString();
  state.runPrefix = crypto.randomUUID().replaceAll("-", "");
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
    $("stop-button").disabled = false;
    $("disconnect-button").disabled = false;
    startRttSampler();
    state.timer = setInterval(() => { updateTimer(); void pollTelemetry(); }, 1000);
    document.addEventListener("visibilitychange", onVisibilityChange);
    showNotice("Grabando. Habla con naturalidad; se mostrarán segmentos finales.", "ok");
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
  state.isRecording = false;
  $("stop-button").disabled = true;
  showNotice("Cerrando el último bloque y esperando flush_complete…");
  try {
    await waitForWorkletFlush();
    const barrier = waitForFlushBarrier();
    const requestId = crypto.randomUUID();
    state.flushId = requestId;
    state.ws.send(JSON.stringify({ event: "flush", source: "mic", request_id: requestId }));
    await barrier;
    showNotice("Sesión terminada. El audio no se conserva.", "ok");
  } catch {
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
  state.token = null;
  state.requestTimes.clear();
  state.activeSegmentRequestId = null;
  $("api-token").value = "";
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
  state.finalised = false;
  $("transcript").textContent = "La transcripción aparecerá cuando el servidor cierre un segmento.";
  $("export-json").disabled = true;
  $("export-csv").disabled = true;
  renderSegments();
  updateSummary();
  renderCharts();
}

function safeResult() {
  const segments = sortedSegments().map((row) => ({
    text: row.text, start: row.start, end: row.end, speaker: row.speaker, backend: row.backend,
    AUDIO_DURATION_MS: row.audio_duration_ms,
    SERVER_ENDPOINTING_MS: row.SERVER_ENDPOINTING_MS,
    SERVER_QUEUE_WAIT_MS: row.SERVER_QUEUE_WAIT_MS,
    SERVER_MODEL_INFERENCE_MS: row.SERVER_MODEL_INFERENCE_MS,
    SERVER_POSTPROCESS_MS: row.SERVER_POSTPROCESS_MS,
    SERVER_EOS_TO_TRANSCRIPT_MS: row.SERVER_EOS_TO_TRANSCRIPT_MS,
    CLIENT_EOS_TO_TRANSCRIPT_MS: row.CLIENT_EOS_TO_TRANSCRIPT_MS,
    MODEL_INFERENCE_MS: row.MODEL_INFERENCE_MS,
    SEGMENT_WAIT_MS: row.SEGMENT_WAIT_MS,
    SERVER_TO_TRANSCRIPT_MS: row.SERVER_TO_TRANSCRIPT_MS,
    SERVER_RECEIVE_TO_TRANSCRIPT_MS: row.SERVER_RECEIVE_TO_TRANSCRIPT_MS,
    SERVER_AUDIO_END_TO_TRANSCRIPT_MS: row.SERVER_AUDIO_END_TO_TRANSCRIPT_MS,
    CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: row.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS,
    queue_wait_ms: row.queue_wait_ms, audio_duration_ms: row.audio_duration_ms,
  }));
  const inference = segments.map((row) => row.MODEL_INFERENCE_MS).filter(Number.isFinite);
  const endpoint = segments.map((row) => row.SERVER_ENDPOINTING_MS).filter(Number.isFinite);
  const queue = segments.map((row) => row.SERVER_QUEUE_WAIT_MS).filter(Number.isFinite);
  const postprocess = segments.map((row) => row.SERVER_POSTPROCESS_MS).filter(Number.isFinite);
  const serverEos = segments.map((row) => row.SERVER_AUDIO_END_TO_TRANSCRIPT_MS).filter(Number.isFinite);
  const clientEos = segments.map((row) => row.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS).filter(Number.isFinite);
  const rtt = rttPercentiles(state.rttSamples);
  const audioMs = segments.map((row) => row.audio_duration_ms).filter(Number.isFinite).reduce((sum, value) => sum + value, 0);
  const inferenceMs = inference.reduce((sum, value) => sum + value, 0);
  const lastTelemetry = state.telemetry.at(-1);
  const result = {
    timestamp: state.runTimestamp,
    finished_at: new Date().toISOString(),
    backend: state.backend,
    model_id: state.modelId,
    model_revision: state.modelRevision,
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
    transcript_mode: "FINAL_SEGMENT",
    reference_text: state.reference || null,
    transcript: transcriptText(),
    segments,
    summary: {
      segment_count: segments.length,
      error_count: state.errors.length,
      errors: [...state.errors],
      transcribed_audio_ms: audioMs,
      MODEL_INFERENCE_RTF: audioMs > 0 ? inferenceMs / audioMs : null,
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
      WER: state.reference.trim() ? wordErrorRate(state.reference, transcriptText()) : null,
    },
    telemetry: [...state.telemetry],
    proxy_ws_rtt_samples: state.rttSamples.map((sample) => ({
      elapsed_ms: sample.elapsed_ms,
      PROXY_WS_RTT_MS: sample.PROXY_WS_RTT_MS,
      status: sample.status,
    })),
    metric_definitions: METRIC_DEFINITIONS,
    measurement_provenance: {
      MODEL_INFERENCE_SCOPE: "adapter_wall_clock_not_gpu_kernel_time",
      GPU_TELEMETRY_SCOPE: "optional_nvml_or_torch_device_global_not_backend_attributed",
      SERVER_CLOCK: "time.perf_counter within ASR server process",
      CLIENT_CLOCK: "performance.now within browser only",
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
  const columns = ["record_type", "timestamp", "backend", "model_id", "gpu", "workers",
    "PROXY_WS_RTT_MS_P50", "PROXY_WS_RTT_MS_P95", "elapsed_ms", "start", "end", "text",
    "AUDIO_DURATION_MS", "SERVER_ENDPOINTING_MS", "SERVER_QUEUE_WAIT_MS", "SERVER_MODEL_INFERENCE_MS",
    "SERVER_POSTPROCESS_MS", "SERVER_EOS_TO_TRANSCRIPT_MS", "CLIENT_EOS_TO_TRANSCRIPT_MS",
    "PROXY_WS_RTT_MS", "MODEL_INFERENCE_MS", "SEGMENT_WAIT_MS", "SERVER_TO_TRANSCRIPT_MS",
    "SERVER_RECEIVE_TO_TRANSCRIPT_MS", "SERVER_AUDIO_END_TO_TRANSCRIPT_MS",
    "CLIENT_AUDIO_END_TO_TRANSCRIPT_MS", "queue_wait_ms", "audio_duration_ms"];
  const toCsvRow = (row) => columns.map((column) => csvCell(row[column] ?? "")).join(",");
  const lines = [toCsvRow(Object.fromEntries(columns.map((column) => [column, column])))];
  for (const row of result.segments) {
    lines.push(toCsvRow({
      record_type: "segment", timestamp: result.timestamp, backend: result.backend,
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
      MODEL_INFERENCE_MS: row.MODEL_INFERENCE_MS,
      SEGMENT_WAIT_MS: row.SEGMENT_WAIT_MS,
      SERVER_TO_TRANSCRIPT_MS: row.SERVER_TO_TRANSCRIPT_MS,
      SERVER_RECEIVE_TO_TRANSCRIPT_MS: row.SERVER_RECEIVE_TO_TRANSCRIPT_MS,
      SERVER_AUDIO_END_TO_TRANSCRIPT_MS: row.SERVER_AUDIO_END_TO_TRANSCRIPT_MS,
      CLIENT_AUDIO_END_TO_TRANSCRIPT_MS: row.CLIENT_AUDIO_END_TO_TRANSCRIPT_MS,
      queue_wait_ms: row.queue_wait_ms, audio_duration_ms: row.audio_duration_ms,
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
