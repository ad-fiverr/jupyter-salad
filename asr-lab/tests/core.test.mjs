import test from "node:test";
import assert from "node:assert/strict";
import {
  CHUNK_SAMPLES,
  MAX_BUFFER_SAMPLES,
  MIN_SPEECH_SAMPLES,
  SILENCE_CHUNKS_TO_FLUSH,
  SPEECH_THRESHOLD,
  SILENCE_THRESHOLD,
  ShadowVad,
  StreamingResampler,
  csvCell,
  floatToPcm16LE,
  median,
  pcm16RmsNormalized,
  percentile,
  wordErrorRate,
} from "../asr_lab/benchmark_web/core.mjs";

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
