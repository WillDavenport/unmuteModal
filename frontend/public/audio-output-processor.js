// TODO: is there a way to get type-checking on this?

function asMs(samples) {
  return ((samples * 1000) / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round((mili * sampleRate) / 1000);
}

const DEFAULT_MAX_BUFFER_MS = 60 * 1000;

const debug = (...args) => {
  // console.debug(...args);
};

// Debug configuration - check for debug flag in URL params
const DEBUG_WAV_ENABLED = true;

// Audio accumulation for consolidated WAV files
let audioBuffers = new Map(); // stage -> array of audio chunks
let bufferSampleRates = new Map(); // stage -> sample rate
let bufferStartTimes = new Map(); // stage -> start timestamp

// Audio accumulation functions
function accumulateAudio(audioData, sampleRate, stage) {
  if (!DEBUG_WAV_ENABLED) return;
  
  try {
    if (!audioBuffers.has(stage)) {
      audioBuffers.set(stage, []);
      bufferSampleRates.set(stage, sampleRate);
      bufferStartTimes.set(stage, getTimestamp());
      console.log(`=== FRONTEND_WAV_DEBUG: Started accumulating audio for ${stage} ===`);
    }
    
    // Copy audio data to avoid reference issues
    const audioCopy = new Float32Array(audioData.length);
    audioCopy.set(audioData);
    audioBuffers.get(stage).push(audioCopy);
    
    const totalSamples = audioBuffers.get(stage).reduce((sum, chunk) => sum + chunk.length, 0);
    const durationSec = totalSamples / sampleRate;
    console.log(`=== FRONTEND_WAV_DEBUG: Accumulated ${audioData.length} samples for ${stage}, total: ${totalSamples} samples (${durationSec.toFixed(2)}s) ===`);
    
  } catch (error) {
    console.error(`Failed to accumulate audio data for ${stage}:`, error);
  }
}

function finalizeStage(stage) {
  if (!DEBUG_WAV_ENABLED) return;
  
  if (!audioBuffers.has(stage) || audioBuffers.get(stage).length === 0) {
    console.log(`=== FRONTEND_WAV_DEBUG: No audio data accumulated for ${stage} ===`);
    return;
  }
  
  try {
    const chunks = audioBuffers.get(stage);
    const sampleRate = bufferSampleRates.get(stage);
    const startTime = bufferStartTimes.get(stage);
    
    // Calculate total length
    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    
    // Concatenate all chunks
    const consolidatedAudio = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      consolidatedAudio.set(chunk, offset);
      offset += chunk.length;
    }
    
    // Create consolidated WAV file
    const filename = `audio-processor-${stage}-${startTime}.wav`;
    createConsolidatedWavFile(consolidatedAudio, sampleRate, filename);
    
    // Clear buffers
    audioBuffers.delete(stage);
    bufferSampleRates.delete(stage);
    bufferStartTimes.delete(stage);
    
    const durationSec = totalLength / sampleRate;
    console.log(`=== FRONTEND_WAV_DEBUG: Finalized ${stage} with ${totalLength} samples (${durationSec.toFixed(2)}s) -> ${filename} ===`);
    
  } catch (error) {
    console.error(`Failed to finalize audio for ${stage}:`, error);
  }
}

// WAV file creation utilities for debugging
function createConsolidatedWavFile(audioData, sampleRate, filename) {
  if (!DEBUG_WAV_ENABLED) return;
  
  try {
    // Convert Float32Array to Int16Array for WAV
    const int16Array = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      // Clamp to [-1, 1] and convert to 16-bit
      const sample = Math.max(-1, Math.min(1, audioData[i]));
      int16Array[i] = sample * 32767;
    }
    
    // Create WAV header
    const buffer = new ArrayBuffer(44 + int16Array.length * 2);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + int16Array.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM format
    view.setUint16(20, 1, true);  // PCM
    view.setUint16(22, 1, true);  // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true);  // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeString(36, 'data');
    view.setUint32(40, int16Array.length * 2, true);
    
    // Copy audio data
    const audioView = new Int16Array(buffer, 44);
    audioView.set(int16Array);
    
    // Send WAV data to main thread for file creation (AudioWorklet can't create Blobs)
    this.port.postMessage({
      type: 'debug-wav',
      buffer: buffer,
      filename: filename,
      samples: audioData.length
    });
    
    console.log(`=== FRONTEND_WAV_DEBUG: Prepared consolidated WAV data for ${filename} with ${audioData.length} samples ===`);
  } catch (error) {
    console.error(`Failed to prepare consolidated WAV debug data for ${filename}:`, error);
  }
}

function getTimestamp() {
  const now = new Date();
  return now.toISOString().replace(/[:.]/g, '-').slice(0, -1); // Remove milliseconds dot
}

class AudioOutputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    debug("AudioOutputProcessor created", currentFrame, sampleRate);

    // Buffer length definitions
    const frameSize = asSamples(80);
    // initialBufferSamples: we wait to have at least that many samples before starting to play
    this.initialBufferSamples = 1 * frameSize;
    // once we have enough samples, we further wait that long before starting to play.
    // This allows to have buffer lengths that are not a multiple of frameSize.
    this.partialBufferSamples = asSamples(10);
    // If the buffer length goes over that many, we will drop the oldest packets until
    // we reach back initialBufferSamples + partialBufferSamples.
    this.maxBufferSamples = asSamples(DEFAULT_MAX_BUFFER_MS);
    // increments
    this.partialBufferIncrement = asSamples(5);
    this.maxPartialWithIncrements = asSamples(80);
    this.maxBufferSamplesIncrement = asSamples(5);
    this.maxMaxBufferWithIncrements = asSamples(80);

    // State and metrics
    this.initState();

    this.port.onmessage = (event) => {
      if (event.data.type == "reset") {
        debug("Reset audio processor state.");
        
        // Finalize any accumulated debug audio before reset
        if (DEBUG_WAV_ENABLED) {
          finalizeStage("received");
          finalizeStage("output");
        }
        
        this.initState();
        return;
      }
      
      if (event.data.type == "finalize-debug") {
        debug("Finalizing debug audio stages.");
        
        // Finalize any accumulated debug audio
        if (DEBUG_WAV_ENABLED) {
          finalizeStage("received");
          finalizeStage("output");
        }
        
        return;
      }
      
      let frame = event.data.frame;
      
      // Accumulate received frames for consolidated WAV file
      if (DEBUG_WAV_ENABLED && frame && frame.length > 0) {
        accumulateAudio(frame, sampleRate, "received");
      }
      
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
    
    // Accumulate processed output audio for consolidated WAV file
    if (DEBUG_WAV_ENABLED && out_idx > 0 && anyAudio) {
      // Create a copy of the relevant portion of the output buffer
      const outputCopy = new Float32Array(out_idx);
      for (let i = 0; i < out_idx; i++) {
        outputCopy[i] = output[i];
      }
      accumulateAudio(outputCopy, sampleRate, "output");
    }
    
    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += out_idx / sampleRate;
    this.timeInStream += out_idx / sampleRate;
    return true;
  }
}
registerProcessor("audio-output-processor", AudioOutputProcessor);
