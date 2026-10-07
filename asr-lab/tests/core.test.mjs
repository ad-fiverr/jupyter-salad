import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import {
  AUDIO_PUSH_INTERVAL_MS,
  CHUNK_SAMPLES,
  clientFirstPartialMs,
  decodeSloLabel,
  METRIC_DEFINITIONS,
  MAX_BUFFER_SAMPLES,
  MIN_SPEECH_SAMPLES,
  SILENCE_CHUNKS_TO_FLUSH,
  SPEECH_THRESHOLD,
  SILENCE_THRESHOLD,
  ShadowVad,
  StreamingResampler,
  boundedPush,
  canvasBackingSize,
  csvCell,
  floatToPcm16LE,
  formatMilliseconds,
  gpuStatusLabel,
  isTerminalQwenError,
  median,
  pcm16RmsNormalized,
  percentile,
  rttPercentiles,
  qwenDeviceTelemetryLabel,
  safeQwenStartupEvidence,
  safeQwenEpochObservabilitySnapshot,
  wordErrorRate,
  shouldDrawCanvas,
} from "../asr_lab/benchmark_web/core.mjs";

const appSource = readFileSync(new URL("../asr_lab/benchmark_web/app.mjs", import.meta.url), "utf8");
const htmlSource = readFileSync(new URL("../asr_lab/benchmark_web/index.html", import.meta.url), "utf8");

function renderQwenLifecycleWithFakeDom(stateOverrides = {}) {
  const start = appSource.indexOf("function renderQwenLifecycle() {");
  const next = appSource.indexOf("function onTranscript", start);
  const end = appSource.lastIndexOf("}", next) + 1;
  assert.ok(start >= 0 && next >= 0 && end > start, "renderer source boundary exists");
  const renderer = appSource.slice(start, end);
  const ids = [
    "qwen-lifecycle-panel", "qwen-lifecycle-public-id", "qwen-lifecycle-current", "qwen-lifecycle-state",
    "qwen-lifecycle-rollovers", "qwen-lifecycle-transition", "qwen-lifecycle-last-completed", "qwen-lifecycle-reason",
    "qwen-lifecycle-failure", "qwen-lifecycle-latency", "qwen-lifecycle-pcm", "qwen-lifecycle-retained",
    "qwen-lifecycle-stale", "qwen-lifecycle-audio-accum", "qwen-epoch-history-body", "qwen-transition-history-body",
  ];
  const elements = new Map(ids.map((id) => [id, { id, hidden: false, textContent: "", rows: [], replaceChildren() { this.rows = []; } }]));
  const state = { backend: "qwen3_asr", qwenEpochObservability: null, lastQwenLifecycleError: null, qwenPublicStreamId: null, qwenLocalStreamId: null, ...stateOverrides };
  const context = {
    state,
    $: (id) => elements.get(id),
    fmtMs: formatMilliseconds,
    appendLifecycleRow: (body, cells, style, span) => body.rows.push({ cells, style, span }),
  };
  runInNewContext(`${renderer}\nrenderQwenLifecycle();`, context);
  return { elements, state };
}

function sine(rate, frequency, sampleCount, start = 0) {
  return Float32Array.from(
    { length: sampleCount },
    (_, index) => Math.sin(2 * Math.PI * frequency * (start + index) / rate),
  );
}

function concat(chunks) {
  const size = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const result = new Float32Array(size);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result;
}

test("PCM conversion is signed 16-bit little-endian and RMS matches the server scale", () => {
  const pcm = floatToPcm16LE(Float32Array.of(-1, -0.5, 0, 0.5, 1));
  const view = new DataView(pcm);
  assert.deepEqual(Array.from({ length: 5 }, (_, index) => view.getInt16(index * 2, true)), [-32768, -16384, 0, 16384, 32767]);
  assert.equal(pcm16RmsNormalized(floatToPcm16LE(new Float32Array(16).fill(0.1))) > SPEECH_THRESHOLD, true);
  assert.equal(pcm16RmsNormalized(floatToPcm16LE(new Float32Array(16).fill(0))), 0);
});

test("same-rate resampling passes samples through and 48 kHz streams to exactly 16 kHz", () => {
  const same = new StreamingResampler(16_000);
  assert.deepEqual(Array.from(same.push(Float32Array.of(0.1, -0.2))), Array.from(Float32Array.of(0.1, -0.2)));
  assert.deepEqual(Array.from(same.finish()), []);

  const resampler = new StreamingResampler(48_000);
  const output = [];
  for (let offset = 0; offset < 48_000; offset += 480) {
    output.push(resampler.push(sine(48_000, 1000, 480, offset)));
  }
  output.push(resampler.finish());
  assert.equal(concat(output).length, 16_000);
});

test("44.1 kHz resampling has stable duration and rejects frequencies above output Nyquist", () => {
  const resampler = new StreamingResampler(44_100);
  const output = [];
  for (let offset = 0; offset < 44_100; offset += 441) {
    output.push(resampler.push(sine(44_100, 1000, 441, offset)));
  }
  output.push(resampler.finish());
  assert.equal(concat(output).length, 16_000);

  const antiAlias = new StreamingResampler(48_000);
  const filtered = [];
  for (let offset = 0; offset < 24_000; offset += 480) {
    filtered.push(antiAlias.push(sine(48_000, 12_000, 480, offset)));
  }
  filtered.push(antiAlias.finish());
  const signal = concat(filtered);
  const rms = Math.sqrt(signal.reduce((sum, value) => sum + value * value, 0) / signal.length);
  assert.ok(rms < 0.01, `expected 12 kHz alias to be attenuated, got RMS ${rms}`);
});

test("shadow VAD mirrors the server thresholds and flush boundary", () => {
  assert.equal(CHUNK_SAMPLES, 1600);
  assert.equal(SPEECH_THRESHOLD, 0.015);
  assert.equal(SILENCE_THRESHOLD, 0.008);
  assert.equal(MIN_SPEECH_SAMPLES, 8000);
  assert.equal(MAX_BUFFER_SAMPLES, 48000);
  assert.equal(SILENCE_CHUNKS_TO_FLUSH, 4);

  const vad = new ShadowVad();
  const speech = floatToPcm16LE(new Float32Array(CHUNK_SAMPLES).fill(0.05));
  let state;
  for (let index = 0; index < 5; index += 1) state = vad.push(speech, index * 100);
  assert.equal(state.canTranscribe, true);
  assert.equal(state.sampleCount, MIN_SPEECH_SAMPLES);
  const silence = floatToPcm16LE(new Float32Array(CHUNK_SAMPLES));
  for (let index = 0; index < SILENCE_CHUNKS_TO_FLUSH - 1; index += 1) {
    state = vad.push(silence, 500 + index * 100);
    assert.equal(state.closed, false);
  }
  state = vad.push(silence, 800);
  assert.equal(state.closed, true);
  assert.equal(state.canTranscribe, true);
  assert.equal(state.lastVoiceAt, 400);
  assert.equal(vad.speaking, false);
});

test("safe startup evidence retains real nested warmup provenance and readiness", () => {
  const evidence = safeQwenStartupEvidence({
    readinessAtStart: { ready: false, backend: "qwen3_asr", production_backend: "qwen3_asr", model_loaded: false, api_token: "secret" },
    qwenStartMetrics: {
      FIRST_STREAM_INIT_MS: 7,
      FIRST_STREAM_STATE_INIT_WALL_MS: 2,
      FIRST_STREAM_INIT_RPC_OVERHEAD_MS: 5,
      private_value: "must-not-export",
    },
    health: {
      model_id: "Qwen/Qwen3-ASR-1.7B", model_revision: "revision-fixture", workers: 1,
      worker_metrics: [{ model_load_ms: 42, warmup_ms: 5, experiment_config: { warmup_chunk_ms: 1000 } }],
      runtime_provenance: {
        qwen_asr_version: "0.0.6", vllm_version: "0.14.0", transformers_version: "4.1",
        torch_version: "2.7", torch_cuda_version: "12.8",
        experiment_config: { warmup_chunk_ms: 1000, max_num_seqs: 1, api_token: "secret" },
        process_args: "must-not-export",
      },
      environment: { API_TOKEN: "must-not-export" },
    },
  });
  assert.equal(evidence.readiness_at_start.ready, false);
  assert.equal(evidence.FIRST_STREAM_INIT_MS, 7);
  assert.equal(evidence.FIRST_STREAM_STATE_INIT_WALL_MS, 2);
  assert.equal(evidence.FIRST_STREAM_INIT_RPC_OVERHEAD_MS, 5);
  assert.equal(evidence.model_id, "Qwen/Qwen3-ASR-1.7B");
  assert.equal(evidence.model_revision, "revision-fixture");
  assert.deepEqual(evidence.model_load_ms, [42]);
  assert.deepEqual(evidence.warmup_ms, [5]);
  assert.deepEqual(evidence.warmup_chunk_ms, [1000]);
  assert.equal(evidence.runtime_provenance.vllm_version, "0.14.0");
  assert.equal(evidence.runtime_provenance.experiment_config.max_num_seqs, 1);
  assert.equal(JSON.stringify(evidence).includes("must-not-export"), false);
  assert.equal(JSON.stringify(evidence).includes("secret"), false);
});

test("startup evidence supports legacy warmup field and missing values stay null or empty", () => {
  const legacy = safeQwenStartupEvidence({
    health: { worker_metrics: [{ model_load_ms: 4, warmup_ms: 2, warmup_chunk_ms: 500 }] },
  });
  assert.deepEqual(legacy.warmup_chunk_ms, [500]);
  const missing = safeQwenStartupEvidence({ readinessAtStart: { ready: false } });
  assert.equal(missing.readiness_at_start.ready, false);
  assert.equal(missing.FIRST_STREAM_INIT_MS, null);
  assert.equal(missing.FIRST_STREAM_STATE_INIT_WALL_MS, null);
  assert.equal(missing.FIRST_STREAM_INIT_RPC_OVERHEAD_MS, null);
  assert.deepEqual(missing.model_load_ms, []);
  assert.deepEqual(missing.warmup_ms, []);
  assert.deepEqual(missing.warmup_chunk_ms, []);
  assert.equal(missing.runtime_provenance.torch_version, null);
});

test("JSON startup evidence is exported independently of partial or final candidates", () => {
  const resultBuilder = appSource.slice(appSource.indexOf("function safeResult()"), appSource.indexOf("function download("));
  assert.match(resultBuilder, /startup_evidence:\s*safeQwenStartupEvidence\(\{/);
  assert.match(resultBuilder, /readinessAtStart:\s*state\.readinessAtStart/);
  assert.match(resultBuilder, /health:\s*state\.health/);
  assert.ok(resultBuilder.indexOf("startup_evidence:") < resultBuilder.indexOf("streaming:"));
  const noCandidate = safeQwenStartupEvidence({
    readinessAtStart: { ready: true },
    qwenStartMetrics: { FIRST_STREAM_INIT_MS: 9 },
  });
  assert.equal(noCandidate.FIRST_STREAM_INIT_MS, 9);
  assert.equal(noCandidate.readiness_at_start.ready, true);
});

test("summary helpers calculate WER and percentiles deterministically", () => {
  assert.equal(wordErrorRate("Hola, mundo", "hola mundo"), 0);
  assert.equal(wordErrorRate("uno dos tres", "uno cuatro"), 2 / 3);
  assert.equal(wordErrorRate("   ", "cualquier cosa"), null);
  assert.equal(median([8, 2, 4, 6]), 5);
  assert.equal(percentile([1, 2, 3, 4, 5], 0.95), 5);
  assert.equal(percentile([], 0.95), null);
});

test("CSV escapes cells and neutralizes spreadsheet formulas", () => {
  assert.equal(csvCell('texto,"con coma"'), '"texto,""con coma"""');
  assert.equal(csvCell("=HYPERLINK(\"https://example.invalid\")"), "\"'=HYPERLINK(\"\"https://example.invalid\"\")\"");
  assert.equal(csvCell(-12), "-12");
});

test("compute availability never becomes a false GPU-unavailable label when CUDA is confirmed", () => {
  assert.equal(gpuStatusLabel({ available: true, cuda: true, device: "NVIDIA RTX 3090" }, { available: false }), "NVIDIA RTX 3090");
  assert.equal(gpuStatusLabel({ available: false, cuda: false }, { device: "NVIDIA RTX 3090" }), "NVIDIA RTX 3090 · cómputo CUDA no confirmado");
  assert.equal(gpuStatusLabel({}, {}), "GPU de cómputo no confirmada");
});

test("Qwen dashboard formats flattened device-global GPU and RAM telemetry", () => {
  const label = qwenDeviceTelemetryLabel({
    gpu_compute: { available: true, cuda: true, device: "NVIDIA RTX 3090" },
    gpu_device: "NVIDIA RTX 3090",
    vram_used_mib: 8192,
    vram_total_mib: 24576,
    process_rss_mib: 512,
  });
  assert.equal(label, "NVIDIA RTX 3090 · 8192/24576 MiB · 512 MiB RAM");
  assert.match(appSource, /qwenDeviceTelemetryLabel\(latestTelemetry\)/);
  assert.doesNotMatch(appSource.slice(appSource.indexOf("function updateQwenSummary"), appSource.indexOf("function renderQwenCharts")), /latestTelemetry\.gpu_telemetry/);
});

test("browser control RTT samples are bounded and expose separate p50/p95", () => {
  const samples = [{ PROXY_WS_RTT_MS: 20 }, { PROXY_WS_RTT_MS: 31 }, { PROXY_WS_RTT_MS: 44 }, { PROXY_WS_RTT_MS: null }];
  assert.deepEqual(rttPercentiles(samples), { p50: 31, p95: 44 });
  for (let index = 0; index < 800; index += 1) boundedPush(samples, { PROXY_WS_RTT_MS: index }, 720);
  assert.equal(samples.length, 720);
});

test("latency rendering preserves hundredth-millisecond queue values", () => {
  assert.equal(formatMilliseconds(0.02), "0.02 ms");
  assert.equal(formatMilliseconds(112), "112.00 ms");
  assert.equal(formatMilliseconds(null), "—");
});

test("mobile canvas backing dimensions cap DPR and skip hidden or zero-size canvases", () => {
  assert.deepEqual(canvasBackingSize(390, 150, 3), { width: 780, height: 300, dpr: 2 });
  assert.deepEqual(canvasBackingSize(5000, 3000, 2), { width: 4096, height: 2048, dpr: 2 });
  assert.equal(canvasBackingSize(0, 150, 2), null);
  assert.equal(shouldDrawCanvas(true, 390, 150), false);
  assert.equal(shouldDrawCanvas(false, 0, 150), false);
  assert.equal(shouldDrawCanvas(false, 390, 150), true);
});

test("metric definitions distinguish duration, server-only clocks, client path and RTT", () => {
  assert.match(METRIC_DEFINITIONS.AUDIO_DURATION_MS, /no es latencia/);
  assert.match(METRIC_DEFINITIONS.SERVER_EOS_TO_TRANSCRIPT_MS, /no incluye navegador ni red/);
  assert.match(METRIC_DEFINITIONS.PROXY_WS_RTT_MS, /No es latencia unidireccional/);
  assert.match(METRIC_DEFINITIONS.SERVER_RECEIVE_TO_TRANSCRIPT_MS, /No es KPI/);
});

test("application RTT ping uses the authenticated ASR websocket and performance.now only", () => {
  assert.match(appSource, /new URL\("\/asr\/ws"/);
  assert.match(appSource, /event: "benchmark_ping"/);
  assert.match(appSource, /event === "benchmark_pong"/);
  assert.match(appSource, /const sentAt = performance\.now\(\)/);
  assert.match(appSource, /const receivedAt = performance\.now\(\)/);
  assert.match(appSource, /setInterval\(sendRttPing, 5000\)/);
  assert.doesNotMatch(appSource, /localStorage|sessionStorage|document\.cookie/);
});

test("Qwen live UI exposes replaceable partial revisions, chunk experiments and separate final", () => {
  assert.match(htmlSource, /id="transcript-mode"/);
  assert.match(htmlSource, /id="qwen-chunk-size"/);
  for (const id of ["qwen-active-streams", "qwen-pending-active", "qwen-backlog", "qwen-decode-percentiles", "qwen-wait-percentiles", "qwen-stream-lag", "qwen-overruns"]) {
    assert.match(htmlSource, new RegExp(`id="${id}"`));
  }
  for (const id of ["chart-qwen-wait", "chart-qwen-backlog", "chart-qwen-lag", "chart-qwen-decode"]) {
    assert.match(htmlSource, new RegExp(`id="${id}"`));
  }
  assert.match(htmlSource, /Audio hablado · duración, no latencia/);
  assert.match(htmlSource, /Server EOS → transcript/);
  assert.match(htmlSource, /Client EOS → transcript/);
  assert.match(htmlSource, /Server EOS → final candidate/);
  assert.match(htmlSource, /Client EOS → final candidate/);
  assert.match(htmlSource, /WebSocket proxy RTT p50 \/ p95/);
  assert.match(htmlSource, /Final candidate WER/);
  for (const chunk of ["250", "500", "1000", "2000"]) assert.match(htmlSource, new RegExp(chunk + " ms"));
  assert.match(htmlSource, /PARTIAL ≠ TRUTH/);
  assert.match(appSource, /event: "stream_start"/);
  assert.match(appSource, /message\.event === "partial_candidate"/);
  assert.match(appSource, /message\.replace !== true/);
  assert.match(appSource, /message\.revision <= state\.lastPartialRevision/);
  assert.match(appSource, /CLIENT_FIRST_PARTIAL_MS/);
  assert.match(appSource, /CLIENT_PARTIAL_UPDATE_INTERVAL_MS/);
  assert.match(appSource, /qwen-proxy-rtt/);
  assert.match(appSource, /qwen-final-wer/);
  assert.match(appSource, /canvasBackingSize\(rect\.width, rect\.height, window\.devicePixelRatio/);
  assert.match(appSource, /shouldDrawCanvas\(canvas\.closest\("\[hidden\]"\)/);
  assert.match(appSource, /document\.querySelectorAll\("\.offline-only"\)/);
  assert.match(appSource, /const wer = state\.backend === "qwen3_asr" \|\| !state\.reference\.trim\(\)\s+\? null : wordErrorRate\(state\.reference, transcriptText\(\)\)/);
  assert.match(appSource, /const productionWer = !isQwen && reference \? wordErrorRate\(reference, candidateText\) : null/);
  assert.match(appSource, /const candidateWer = isQwen && reference \? wordErrorRate\(reference, candidateText\) : null/);
  assert.match(appSource, /WER: productionWer/);
  assert.match(appSource, /FINAL_WER: productionWer/);
  assert.match(appSource, /FINAL_CANDIDATE_WER: candidateWer/);
  assert.match(appSource, /FINAL_CANDIDATE_WER: row === finalCandidateSource \? candidateWer : null/);
  assert.match(appSource, /const finalCandidateSource = \[\.\.\.orderedRows\]\.reverse\(\)\.find\(\(row\) => row\.candidate_only\) \?\? null/);
  assert.match(appSource, /const serverEosMs = !candidateOnly/);
  assert.match(appSource, /CLIENT_EOS_TO_TRANSCRIPT_MS: candidateOnly \? null : clientEosMs/);
  assert.match(appSource, /SERVER_EOS_TO_FINAL_CANDIDATE_MS: candidateOnly && Number\.isFinite/);
  assert.match(appSource, /CLIENT_EOS_TO_FINAL_CANDIDATE_MS: candidateOnly \? clientEosMs : null/);
  assert.match(appSource, /"SERVER_EOS_TO_FINAL_CANDIDATE_MS", "CLIENT_EOS_TO_FINAL_CANDIDATE_MS", "FINAL_CANDIDATE_WER"/);
  assert.match(appSource, /SERVER_EOS_TO_FINAL_CANDIDATE_MS: row\.SERVER_EOS_TO_FINAL_CANDIDATE_MS/);
  assert.match(appSource, /CLIENT_EOS_TO_FINAL_CANDIDATE_MS: row\.CLIENT_EOS_TO_FINAL_CANDIDATE_MS/);
  assert.match(appSource, /FINAL_CANDIDATE_WER: row\.FINAL_CANDIDATE_WER/);
  assert.match(appSource, /record_type: partial\.event \?\? "partial_candidate"/);
  assert.match(appSource, /message\.event === "final_candidate"/);
  assert.equal(isTerminalQwenError("stream_scheduler_overrun"), true);
  assert.equal(isTerminalQwenError("stream_terminal"), true);
  assert.equal(isTerminalQwenError("invalid_language"), false);
  assert.match(appSource, /state\.backend === "qwen3_asr" && isTerminalQwenError\(code\)/);
  assert.match(appSource, /void cleanup\(false\)/);
  assert.match(appSource, /candidate_only: candidateOnly/);
  assert.match(appSource, /message\.event === "transcript" \|\| message\.type === "transcript"/);
  assert.match(appSource, /candidate_only: true/);
  assert.match(appSource, /transcript: isQwen \? null : transcriptText\(\)/);
  assert.match(appSource, /final_candidate: isQwen \? transcriptText\(\) : null/);
  assert.match(appSource, /transcript_mode: state\.backend === "qwen3_asr" \? "STREAMING_PARTIALS" : "FINAL_SEGMENT"/);
  assert.match(appSource, /state\.clientEosAt = performance\.now\(\)/);
  assert.doesNotMatch(appSource, /localStorage|sessionStorage/);
});

test("real WebSocket handler accepts successor revision reset and fences late predecessor partials", () => {
  const slices = [
    ["function renderQwenPartials() {", "function renderSegments()"],
    ["function onPartialTranscript(message) {", "function appendLifecycleRow"],
    ["function captureQwenEpochEvidence(message) {", "function renderQwenLifecycle()"],
    ["function onSocketMessage(raw) {", "function recordRttFailure"],
    ["function safeResult() {", "function download("],
  ].map(([startMarker, endMarker]) => {
    const start = appSource.indexOf(startMarker);
    const end = appSource.indexOf(endMarker, start + startMarker.length);
    assert.ok(start >= 0 && end > start, "real handler source boundary exists: " + startMarker);
    return appSource.slice(start, end);
  });
  const element = (id) => ({
    id,
    textContent: "",
    hidden: false,
    children: [],
    replaceChildren() { this.children = []; },
    append(...items) { this.children.push(...items); },
  });
  const elements = new Map([
    "qwen-partials-body", "qwen-current-partial", "qwen-stream-meta",
  ].map((id) => [id, element(id)]));
  const state = {
    backend: "qwen3_asr",
    streamId: "public-stream-1",
    qwenPublicStreamId: "public-stream-1",
    qwenLocalStreamId: null,
    qwenEpochObservability: null,
    lastQwenLifecycleError: null,
    lastPartialRevision: 0,
    lastPartialAt: null,
    streamEvents: [],
    startedAt: 0,
    firstAudioSentAt: 0,
    qwenLanguage: "auto",
    qwenChunkMs: 100,
    effectiveMaxBacklogMs: 400,
    qwenContext: "",
    reference: "",
    errors: [],
    telemetry: [],
    rttSamples: [],
    visibility: [],
    runTimestamp: "fixture",
    workers: 1,
    capturedSamples: 0,
    sentBytes: 0,
    micSettings: null,
    health: null,
    readinessAtStart: null,
    qwenStartMetrics: null,
    modelId: null,
    modelRevision: null,
    runtimeProvenance: null,
  };
  let now = 0;
  const context = {
    state,
    $: (id) => elements.get(id),
    document: { createElement: (tag) => ({ tag, children: [], append(...items) { this.children.push(...items); } }) },
    performance: { now: () => ++now },
    fmtMs: formatMilliseconds,
    clientFirstPartialMs,
    AUDIO_PUSH_INTERVAL_MS,
    safeQwenEpochObservabilitySnapshot,
    safeQwenStartupEvidence,
    METRIC_DEFINITIONS,
    wordErrorRate,
    median,
    percentile,
    rttPercentiles,
    renderQwenLifecycle() {},
    updateQwenSummary() {},
    sortedSegments: () => [],
    transcriptText: () => state.streamEvents.at(-1)?.text ?? "",
  };
  runInNewContext(slices.join("\n"), context);

  const epochSnapshot = (localId, epochSeq) => ({
    current_epoch: {
      epoch_id: "epoch-" + epochSeq,
      epoch_seq: epochSeq,
      local_stream_id: localId,
      lifecycle_state: "ACTIVE",
    },
    logical_cumulative: { EPOCH_ROLLOVER_COUNT: epochSeq },
    last_transition: null,
    epoch_history: [],
    transition_history: [],
  });
  const lifecycleMessage = (localId, epochSeq) => context.onSocketMessage(JSON.stringify({
    event: "epoch_observation",
    stream_id: "public-stream-1",
    qwen_local_stream_id: localId,
    qwen_epoch_observability: epochSnapshot(localId, epochSeq),
  }));
  const partialMessage = (localId, epochSeq, revision, text) => context.onSocketMessage(JSON.stringify({
    event: "partial_candidate",
    stream_id: "public-stream-1",
    qwen_local_stream_id: localId,
    qwen_epoch_observability: epochSnapshot(localId, epochSeq),
    truth_status: "candidate_only",
    final: false,
    replace: true,
    revision,
    text,
  }));

  partialMessage("local-predecessor", 0, 1, "predecessor-one");
  partialMessage("local-predecessor", 0, 2, "predecessor-two");
  lifecycleMessage("local-successor", 1);
  partialMessage("local-successor", 1, 1, "successor-one");
  partialMessage("local-successor", 1, 1, "successor-duplicate");
  partialMessage("local-successor", 1, 0, "successor-old");
  partialMessage("local-predecessor", 0, 99, "late-predecessor");
  partialMessage("local-successor", 1, 2, "successor-two");

  assert.deepEqual(state.streamEvents.map((item) => [item.epoch_seq, item.revision, item.text]), [
    [0, 1, "predecessor-one"],
    [0, 2, "predecessor-two"],
    [1, 1, "successor-one"],
    [1, 2, "successor-two"],
  ]);
  assert.equal(state.streamId, "public-stream-1");
  assert.equal(state.qwenLocalStreamId, "local-successor");
  assert.equal(state.qwenEpochObservability.current_epoch.epoch_seq, 1);
  assert.equal(elements.get("qwen-current-partial").textContent, "successor-two");

  const exported = context.safeResult();
  assert.deepEqual(Array.from(exported.streaming.partials).map((item) => item.text), [
    "predecessor-one", "predecessor-two", "successor-one", "successor-two",
  ]);
  assert.deepEqual(new Set(exported.streaming.partials.map((item) => item.public_stream_id)), new Set(["public-stream-1"]));
});

test("benchmark CSS keeps base layout and bounds Qwen/mobile dashboard", () => {
  const styleSource = readFileSync(new URL("../asr_lab/benchmark_web/style.css", import.meta.url), "utf8");
  assert.match(styleSource, /--/);
  assert.match(styleSource, /\.shell/);
  assert.match(styleSource, /\.panel/);
  assert.match(styleSource, /\.qwen-scheduler-grid/);
  assert.match(styleSource, /max-width:560px/);
  assert.match(styleSource, /\.partial-timeline-wrap\{max-height:360px/);
});

test("Qwen epoch snapshot is content-free, bounded, and preserves lifecycle evidence", () => {
  const snapshot = {
    schema_version: 1,
    current_epoch: {
      epoch_id: "epoch-3", epoch_seq: 3, local_stream_id: "local-3", lifecycle_state: "ACTIVE",
      pcm: {
        source_head_cursor: 6400,
        unique_primary_admitted_cursor: 6400,
        received_samples: 6400,
        unique_primary_admitted_samples: 6400,
        replay_admitted_samples: 800,
        transition_queued_samples: 0,
        cutover_cursor: null,
      },
    },
    last_transition: { transition_seq: 3, state: "ACTIVE", stage: "ACTIVE", stage_events: [] },
    epoch_history: Array.from({ length: 4 }, (_, index) => ({
      epoch_id: `epoch-${index}`, epoch_seq: index,
      pcm: { received_samples: index === 3 ? 6400 : null, replay_admitted_samples: 800 },
    })),
    transition_history: [{ transition_seq: 3, stage_events: [{ stage: "ACTIVE" }] }],
    logical_cumulative: { EPOCH_ROLLOVER_COUNT: 3 },
    QWEN_AUDIO_ACCUM_MS: null,
  };
  const retained = safeQwenEpochObservabilitySnapshot(snapshot);
  assert.deepEqual(retained, snapshot);
  assert.notEqual(retained, snapshot);
  assert.equal(retained.current_epoch.pcm.received_samples, 6400);
  for (const unsafePcm of ["raw PCM", [1, 2, 3], new Uint8Array([1, 2]), { received_samples: Infinity }, { raw_audio: "payload" }]) {
    assert.equal(safeQwenEpochObservabilitySnapshot({
      ...snapshot,
      current_epoch: { ...snapshot.current_epoch, pcm: unsafePcm },
    }), null);
  }
  assert.equal(safeQwenEpochObservabilitySnapshot({
    ...snapshot,
    logical_cumulative: { ...snapshot.logical_cumulative, pcm: { received_samples: 6400 } },
  }), null);
  assert.equal(safeQwenEpochObservabilitySnapshot({ ...snapshot, audio: "must not be exported" }), null);
  assert.equal(safeQwenEpochObservabilitySnapshot({
    ...snapshot,
    transition_history: Array.from({ length: 9 }, () => ({ stage_events: [] })),
  }), null);
  assert.equal(safeQwenEpochObservabilitySnapshot({
    ...snapshot,
    transition_history: [{ stage_events: Array.from({ length: 17 }, () => ({ stage: "ACTIVE" })) }],
  }), null);
  assert.equal(safeQwenEpochObservabilitySnapshot({
    ...snapshot,
    current_epoch: { epoch_id: "epoch-3", epoch_seq: 3, transcript: "not allowed" },
  }), null);
});

test("Qwen lifecycle renderer handles reset null and error-only states", () => {
  const reset = renderQwenLifecycleWithFakeDom();
  assert.equal(reset.state.backend, "qwen3_asr");
  assert.equal(reset.state.qwenEpochObservability, null);
  assert.equal(reset.state.lastQwenLifecycleError, null);
  assert.equal(reset.elements.get("qwen-lifecycle-panel").hidden, true);
  assert.equal(reset.elements.get("qwen-lifecycle-audio-accum").textContent, "null · no medible");
  assert.match(reset.elements.get("qwen-epoch-history-body").rows[0].cells[0], /No hay historial/);

  const errorOnly = renderQwenLifecycleWithFakeDom({
    lastQwenLifecycleError: { code: "stream_scheduler_overrun", qwen_public_stream_id: "public-error-1" },
  });
  assert.equal(errorOnly.elements.get("qwen-lifecycle-panel").hidden, false);
  assert.equal(errorOnly.elements.get("qwen-lifecycle-failure").textContent, "stream_scheduler_overrun");
  assert.equal(errorOnly.elements.get("qwen-lifecycle-public-id").textContent, "public-error-1");
  assert.equal(errorOnly.state.qwenEpochObservability, null);
});

test("Qwen lifecycle renderer preserves null audio accumulation as unmeasured", () => {
  const snapshot = { QWEN_AUDIO_ACCUM_MS: null, current_epoch: {}, logical_cumulative: {}, last_transition: null, epoch_history: [], transition_history: [] };
  const result = renderQwenLifecycleWithFakeDom({ qwenEpochObservability: snapshot });
  assert.equal(result.elements.get("qwen-lifecycle-audio-accum").textContent, "null · no medible");
  assert.equal(result.state.qwenEpochObservability.QWEN_AUDIO_ACCUM_MS, null);
});

test("Qwen lifecycle renderer preserves valid epoch and transition history rows", () => {
  const snapshot = {
    current_epoch: { epoch_seq: 7, epoch_id: "epoch-7", local_stream_id: "local-7", lifecycle_state: "ACTIVE" },
    logical_cumulative: { rollover_success_count: 2, rollover_attempt_count: 2 },
    last_transition: { state: "ACTIVE", stage: "ACTIVE", last_completed_stage: "ACTIVE", category: "soft", reason: "limit", latest_pcm_accounting: { explicit_source_rejected_samples: 0 } },
    epoch_history: [{ epoch_seq: 7, epoch_id: "epoch-7", local_stream_id: "local-7", lifecycle_state: "ACTIVE", epoch_audio_ms: 120, replay_audio_ms: 30, replay_wall_ms: 4 }],
    transition_history: [{ transition_seq: 3, predecessor_epoch_id: "epoch-6", predecessor_local_stream_id: "local-6", successor_epoch_id: "epoch-7", successor_local_stream_id: "local-7", state: "COMPLETE", stage: "ACTIVE", stage_events: [{ stage: "ACTIVE", elapsed_ms: 5 }], replay_admitted_samples: 480, transition_primary_samples: 1600, EPOCH_HANDOFF_WALL_MS: 3, EPOCH_FIRST_PARTIAL_AFTER_ROLLOVER_MS: 8, PCM_LOST: 0, PRIMARY_DUP: 0 }],
  };
  const result = renderQwenLifecycleWithFakeDom({ qwenEpochObservability: snapshot });
  const epochRows = result.elements.get("qwen-epoch-history-body").rows;
  const transitionRows = result.elements.get("qwen-transition-history-body").rows;
  assert.equal(epochRows.length, 1);
  assert.equal(epochRows[0].cells[1], "epoch-7");
  assert.equal(transitionRows.length, 1);
  assert.equal(transitionRows[0].cells[0], 3);
  assert.match(transitionRows[0].cells[1], /epoch-6.*epoch-7/);
});
test("Qwen browser UI and JSON retain bounded epoch lineage through terminal errors", () => {
  assert.match(htmlSource, /id="qwen-lifecycle-panel"/);
  assert.match(htmlSource, /id="qwen-epoch-history-body"/);
  assert.match(htmlSource, /id="qwen-transition-history-body"/);
  assert.match(htmlSource, /Qwen local stream id/);
  assert.match(appSource, /captureQwenEpochEvidence\(message\)/);
  assert.match(appSource, /safeQwenEpochObservabilitySnapshot/);
  assert.match(appSource, /state\.lastQwenLifecycleError\s*=\s*\{/);
  assert.match(appSource, /rolling_epoch:\s*state\.qwenEpochObservability\s*\?/);
  assert.match(appSource, /epoch_history:\s*state\.qwenEpochObservability\?\.epoch_history/);
  assert.match(appSource, /transition_history:\s*state\.qwenEpochObservability\?\.transition_history/);
  assert.match(appSource, /QWEN_NEW_AUDIO_MS/);
  assert.match(appSource, /EPOCH_AUDIO_ACCUMULATED_MS/);
  assert.match(appSource, /QWEN_AUDIO_ACCUM_MS/);
  assert.match(appSource, /transition\.failure_stage/);
  assert.match(appSource, /transition\?\.last_completed_stage/);
});

test("Stop drains the last worklet PCM chunk before freezing capture and marking EOS", () => {
  const stopSource = appSource.slice(
    appSource.indexOf("async function stopRun()"),
    appSource.indexOf("async function cleanup(complete)"),
  );
  assert.ok(stopSource.indexOf("await waitForWorkletFlush()") < stopSource.indexOf("state.isRecording = false"));
  assert.ok(stopSource.indexOf("state.isRecording = false") < stopSource.indexOf("state.clientEosAt = performance.now()"));
});


test("client first partial begins immediately before the first audio send", () => {
  const runStartedAt = 100;
  const firstAudioSentAt = 250;
  const receivedAt = 410;
  assert.equal(AUDIO_PUSH_INTERVAL_MS, 100);
  assert.equal(clientFirstPartialMs(firstAudioSentAt, receivedAt), 160);
  assert.notEqual(clientFirstPartialMs(firstAudioSentAt, receivedAt), receivedAt - runStartedAt);
  assert.equal(clientFirstPartialMs(null, receivedAt), null);
});

test("decode SLO display distinguishes unmeasured from a measured pass", () => {
  assert.equal(decodeSloLabel(null), "unmeasured");
  assert.equal(decodeSloLabel(undefined), "unmeasured");
  assert.equal(decodeSloLabel(false), "within target");
  assert.equal(decodeSloLabel(true), "violation");
  for (const metric of [
    "FIRST_STREAM_INIT_RPC_OVERHEAD_MS", "EPOCH_FIRST_DECODE_WALL_MS",
    "EPOCH_SECOND_DECODE_WALL_MS", "EPOCH_STEADY_DECODE_WALL_P50_MS",
    "EPOCH_STEADY_DECODE_WALL_P95_MS", "SERVER_FIRST_PARTIAL_MS", "CLIENT_FIRST_PARTIAL_MS",
  ]) assert.ok(appSource.includes(metric), `missing visible metric ${metric}`);
});
