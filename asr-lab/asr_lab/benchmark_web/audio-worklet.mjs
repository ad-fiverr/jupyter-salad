import { CHUNK_SAMPLES, OUTPUT_RATE, StreamingResampler, floatToPcm16LE } from "./core.mjs";

class AsrBenchmarkMicrophone extends AudioWorkletProcessor {
  constructor() {
    super();
    this.resampler = new StreamingResampler(sampleRate, OUTPUT_RATE);
    this.pending = [];
    this.stopped = false;
    this.port.onmessage = (message) => {
      if (message.data?.type === "flush") this.flush();
    };
  }

  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0].length) return true;
    const frames = channels[0].length;
    const mono = new Float32Array(frames);
    for (let i = 0; i < frames; i += 1) {
      let sample = 0;
      for (let channel = 0; channel < channels.length; channel += 1) sample += channels[channel][i] || 0;
      mono[i] = sample / channels.length;
    }
    this.accept(this.resampler.push(mono));
    return true;
  }

  accept(samples) {
    for (let i = 0; i < samples.length; i += 1) this.pending.push(samples[i]);
    while (this.pending.length >= CHUNK_SAMPLES) {
      this.emit(this.pending.splice(0, CHUNK_SAMPLES));
    }
  }

  emit(samples) {
    const pcm = floatToPcm16LE(samples);
    this.port.postMessage({ type: "pcm_chunk", pcm, sample_count: samples.length }, [pcm]);
  }

  flush() {
    if (this.stopped) return;
    this.accept(this.resampler.finish());
    if (this.pending.length) this.emit(this.pending.splice(0));
    this.stopped = true;
    this.port.postMessage({ type: "flushed" });
  }
}

registerProcessor("asr-benchmark-microphone", AsrBenchmarkMicrophone);
