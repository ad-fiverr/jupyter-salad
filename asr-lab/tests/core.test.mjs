import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  CHUNK_SAMPLES,
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
  wordErrorRate,
  shouldDrawCanvas,
} from "../asr_lab/benchmark_web/core.mjs";

const appSource = readFileSync(new URL("../asr_lab/benchmark_web/app.mjs", import.meta.url), "utf8");
const htmlSource = readFileSync(new URL("../asr_lab/benchmark_web/index.html", import.meta.url), "utf8");

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

test("benchmark CSS keeps base layout and bounds Qwen/mobile dashboard", () => {
  const styleSource = readFileSync(new URL("../asr_lab/benchmark_web/style.css", import.meta.url), "utf8");
  assert.match(styleSource, /--/);
  assert.match(styleSource, /\.shell/);
  assert.match(styleSource, /\.panel/);
  assert.match(styleSource, /\.qwen-scheduler-grid/);
  assert.match(styleSource, /max-width:560px/);
  assert.match(styleSource, /\.partial-timeline-wrap\{max-height:360px/);
});

test("Stop drains the last worklet PCM chunk before freezing capture and marking EOS", () => {
  const stopSource = appSource.slice(
    appSource.indexOf("async function stopRun()"),
    appSource.indexOf("async function cleanup(complete)"),
  );
  assert.ok(stopSource.indexOf("await waitForWorkletFlush()") < stopSource.indexOf("state.isRecording = false"));
  assert.ok(stopSource.indexOf("state.isRecording = false") < stopSource.indexOf("state.clientEosAt = performance.now()"));
});
