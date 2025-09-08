// TODO: is there a way to get type-checking on this?

function asMs(samples) {
  return ((samples * 1000) / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round((mili * sampleRate) / 1000);
}

// Simplified pipeline: smaller jitter buffer for lower latency
const DEFAULT_MAX_BUFFER_MS = 120; // Reduced from 60 seconds to 120ms
const INITIAL_BUFFER_MS = 80; // Target initial buffer
const PARTIAL_BUFFER_MS = 20; // Smaller partial buffer

const debug = (...args) => {
  // console.debug(...args);
};

class AudioOutputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    debug("AudioOutputProcessor created (simplified pipeline)", currentFrame, sampleRate);

    // Simplified buffer configuration for lower latency
    const frameSize = asSamples(80);
    // initialBufferSamples: reduced for faster start
    this.initialBufferSamples = asSamples(INITIAL_BUFFER_MS);
    // partialBufferSamples: smaller for lower latency
    this.partialBufferSamples = asSamples(PARTIAL_BUFFER_MS);
    // maxBufferSamples: much smaller for simplified pipeline
    this.maxBufferSamples = asSamples(DEFAULT_MAX_BUFFER_MS);
    // increments: smaller adjustments
    this.partialBufferIncrement = asSamples(5);
    this.maxPartialWithIncrements = asSamples(60); // Reduced from 80ms
    this.maxBufferSamplesIncrement = asSamples(5);
    this.maxMaxBufferWithIncrements = asSamples(150); // Reduced from 80ms

    // State and metrics
    this.initState();

    this.port.onmessage = (event) => {
      if (event.data.type == "reset") {
        debug("Reset audio processor state.");
        this.initState();
        return;
      }
      if (event.data.type == "flush") {
        debug("Flush audio processor buffers (response.interrupted).");
        this.flushBuffers();
        return;
      }
      let frame = event.data.frame;
      this.frames.push(frame);
      if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
        this.start();
      }
      if (this.pidx < 20) {
        debug(
          this.timestamp(),
          "Got packet",
          this.pidx++,
          asMs(this.currentSamples()),
          asMs(frame.length)
        );
      }
      if (this.currentSamples() >= this.totalMaxBufferSamples()) {
        debug(
          this.timestamp(),
          "Dropping packets",
          asMs(this.currentSamples()),
          asMs(this.totalMaxBufferSamples())
        );
        console.warn("Dropping packets", asMs(this.currentSamples()), asMs(this.totalMaxBufferSamples()));
        const target = this.initialBufferSamples + this.partialBufferSamples;
        while (
          this.currentSamples() >
          this.initialBufferSamples + this.partialBufferSamples
        ) {
          const first = this.frames[0];
          let to_remove = this.currentSamples() - target;
          to_remove = Math.min(
            first.length - this.offsetInFirstBuffer,
            to_remove
          );
          this.offsetInFirstBuffer += to_remove;
          this.timeInStream += to_remove / sampleRate;
          if (this.offsetInFirstBuffer == first.length) {
            this.frames.shift();
            this.offsetInFirstBuffer = 0;
          }
        }
        debug(this.timestamp(), "Packet dropped", asMs(this.currentSamples()));
        this.maxBufferSamples += this.maxBufferSamplesIncrement;
        this.maxBufferSamples = Math.min(
          this.maxMaxBufferWithIncrements,
          this.maxBufferSamples
        );
        debug("Increased maxBuffer to", asMs(this.maxBufferSamples));
      }
      this.port.postMessage({
        totalAudioPlayed: this.totalAudioPlayed,
        actualAudioPlayed: this.actualAudioPlayed,
        delay: event.data.micDuration - this.timeInStream,
        minDelay: this.minDelay,
        maxDelay: this.maxDelay,
      });
    };
  }

  initState() {
    this.frames = [];
    this.offsetInFirstBuffer = 0;
    this.firstOut = false;
    this.remainingPartialBufferSamples = 0;
    this.timeInStream = 0;
    this.resetStart();

    // Metrics
    this.totalAudioPlayed = 0;
    this.actualAudioPlayed = 0;
    this.maxDelay = 0;
    this.minDelay = 2000;
    // Debug
    this.pidx = 0;

    // For now let's reset the buffer params.
    this.partialBufferSamples = asSamples(10);
    this.maxBufferSamples = asSamples(DEFAULT_MAX_BUFFER_MS);
  }

  totalMaxBufferSamples() {
    return (
      this.maxBufferSamples +
      this.partialBufferSamples +
      this.initialBufferSamples
    );
  }

  timestamp() {
    return Date.now() % 1000;
  }

  currentSamples() {
    let samples = 0;
    for (let k = 0; k < this.frames.length; k++) {
      samples += this.frames[k].length;
    }
    samples -= this.offsetInFirstBuffer;
    return samples;
  }

  resetStart() {
    this.started = false;
  }

  start() {
    this.started = true;
    this.remainingPartialBufferSamples = this.partialBufferSamples;
    this.firstOut = true;
  }

  flushBuffers() {
    // Clear all audio frames immediately (for response.interrupted)
    debug("Flushing all audio buffers due to interruption");
    this.frames = [];
    this.offsetInFirstBuffer = 0;
    this.resetStart();
    console.log("Audio buffers flushed due to response.interrupted");
  }

  canPlay() {
    return (
      this.started &&
      this.frames.length > 0 &&
      this.remainingPartialBufferSamples <= 0
    );
  }

  process(inputs, outputs) {
    const delay = this.currentSamples() / sampleRate;
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }
    const output = outputs[0][0];
    if (!this.canPlay()) {
      if (this.actualAudioPlayed > 0) {
        this.totalAudioPlayed += output.length / sampleRate;
      }
      this.remainingPartialBufferSamples -= output.length;
      return true;
    }
    if (this.firstOut) {
      debug(
        this.timestamp(),
        "Audio resumed",
        asMs(this.currentSamples()),
        this.remainingPartialBufferSamples
      );
    }
    let out_idx = 0;
    let anyAudio = false;
    while (out_idx < output.length && this.frames.length) {
      const first = this.frames[0];
      const to_copy = Math.min(
        first.length - this.offsetInFirstBuffer,
        output.length - out_idx
      );
      const subArray = first.subarray(
        this.offsetInFirstBuffer,
        this.offsetInFirstBuffer + to_copy
      );
      output.set(subArray, out_idx);
      anyAudio =
        anyAudio ||
        output.some(function (x) {
          x > 1e-4 || x < -1e-4;
        });
      this.offsetInFirstBuffer += to_copy;
      out_idx += to_copy;
      if (this.offsetInFirstBuffer == first.length) {
        this.offsetInFirstBuffer = 0;
        this.frames.shift();
      }
    }
    if (this.firstOut) {
      this.firstOut = false;
      for (let i = 0; i < out_idx; i++) {
        output[i] *= i / out_idx;
      }
    }
    if (out_idx < output.length && !anyAudio) {
      // At the end of a turn, we will get some padding of 0, so we only
      // incease the buffer if we got some audio, e.g. we truly lagged in the middle of something.
      debug(this.timestamp(), "Missed some audio", output.length - out_idx);
      this.partialBufferSamples += this.partialBufferIncrement;
      this.partialBufferSamples = Math.min(
        this.partialBufferSamples,
        this.maxPartialWithIncrements
      );
      debug("Increased partial buffer to", asMs(this.partialBufferSamples));
      // We ran out of a buffer, let's revert to the started state to replenish it.
      this.resetStart();
      for (let i = 0; i < out_idx; i++) {
        output[i] *= (out_idx - i) / out_idx;
      }
    }
    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += out_idx / sampleRate;
    this.timeInStream += out_idx / sampleRate;
    return true;
  }
}
registerProcessor("audio-output-processor", AudioOutputProcessor);
