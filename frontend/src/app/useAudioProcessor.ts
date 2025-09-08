import { useRef, useCallback } from "react";
import OpusRecorder from "opus-recorder";

const getAudioWorkletNode = async (
  audioContext: AudioContext,
  name: string
) => {
  try {
    return new AudioWorkletNode(audioContext, name);
  } catch {
    await audioContext.audioWorklet.addModule(`/${name}.js`);
    return new AudioWorkletNode(audioContext, name, {});
  }
};

export interface AudioProcessor {
  audioContext: AudioContext;
  opusRecorder: OpusRecorder;
  decoder: DecoderWorker;
  outputWorklet: AudioWorkletNode;
  inputAnalyser: AnalyserNode;
  outputAnalyser: AnalyserNode;
  mediaStreamDestination: MediaStreamAudioDestinationNode;
}

export const useAudioProcessor = (
  onOpusRecorded: (chunk: Uint8Array) => void
) => {
  const audioProcessorRef = useRef<AudioProcessor | null>(null);

  const setupAudio = useCallback(
    async (mediaStream: MediaStream) => {
      if (audioProcessorRef.current) return audioProcessorRef.current;

      const audioContext = new AudioContext();
      const outputWorklet = await getAudioWorkletNode(
        audioContext,
        "audio-output-processor"
      );
      const source = audioContext.createMediaStreamSource(mediaStream);
      // source.connect(inputWorklet);
      const inputAnalyser = audioContext.createAnalyser();
      inputAnalyser.fftSize = 2048;
      source.connect(inputAnalyser);

      const mediaStreamDestination =
        audioContext.createMediaStreamDestination();
      outputWorklet.connect(mediaStreamDestination);
      source.connect(mediaStreamDestination);

      outputWorklet.connect(audioContext.destination);
      const outputAnalyser = audioContext.createAnalyser();
      outputAnalyser.fftSize = 2048;
      outputWorklet.connect(outputAnalyser);

      const createDecoder = () => {
        const worker = new Worker("/decoderWorker.min.js");
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        worker.onmessage = (event: MessageEvent<any>) => {
          if (!event.data) {
            return;
          }
          const frame = event.data[0];
          console.log(`=== FRONTEND_AUDIO_DEBUG: Decoder worker returned PCM frame with ${frame.length} samples ===`);
          const micDuration = opusRecorder.encodedSamplePosition / 48000;
          outputWorklet.port.postMessage({
            frame: frame,
            type: "audio",
            micDuration: micDuration,
          });
          console.log(`=== FRONTEND_AUDIO_DEBUG: Sent PCM frame to output worklet ===`);
        };
        worker.postMessage({
          command: "init",
          bufferLength: (960 * audioContext.sampleRate) / 24000,
          decoderSampleRate: 24000,
          outputBufferSampleRate: audioContext.sampleRate,
          resampleQuality: 0,
        });
        return worker;
      };

      const decoder = createDecoder();

      // For buffer length: 960 = 24000 / 12.5 / 2
      // The /2 is a bit optional, but won't hurt for recording the mic.
      // Note that bufferLength actually has 0 impact for mono audio, only
      // the frameSize and maxFramesPerPage seems to have any.
      const recorderOptions = {
        mediaTrackConstraints: {
          audio: {
            echoCancellation: true,
            noiseSuppression: false,
            autoGainControl: true,
            channelCount: 1,
          },
          video: false,
        },
        encoderPath: "/encoderWorker.min.js",
        bufferLength: Math.round((960 * audioContext.sampleRate) / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: 2,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
      };
      let chunk_idx = 0;
      let lastpos = 0;
      const opusRecorder = new OpusRecorder(recorderOptions);
      opusRecorder.ondataavailable = (data: Uint8Array) => {
        // logging disabled
        if (chunk_idx < 0) {
          const micDurationSec = opusRecorder.encodedSamplePosition / 48000;
          console.debug(
            Date.now() % 1000,
            "Mic Data chunk",
            chunk_idx++,
            (opusRecorder.encodedSamplePosition - lastpos) / 48000,
            micDurationSec
          );
          lastpos = opusRecorder.encodedSamplePosition;
        }
        onOpusRecorded(data);
      };
      audioProcessorRef.current = {
        audioContext,
        opusRecorder,
        decoder,
        outputWorklet,
        inputAnalyser,
        outputAnalyser,
        mediaStreamDestination,
      };
      // Resume the audio context if it was suspended
      audioProcessorRef.current.audioContext.resume();
      opusRecorder.start();

      return audioProcessorRef.current;
    },
    [onOpusRecorded]
  );

  const flushOutput = useCallback(() => {
    const ap = audioProcessorRef.current;
    if (!ap) return;
    try {
      // Flush output worklet buffers
      ap.outputWorklet.port.postMessage({ type: "reset" });
    } catch {}
    try {
      // Recreate decoder to drop any pending frames and state
      ap.decoder.terminate();
    } catch {}
    const audioContext = ap.audioContext;
    const outputWorklet = ap.outputWorklet;
    const opusRecorder = ap.opusRecorder;
    const worker = new Worker("/decoderWorker.min.js");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    worker.onmessage = (event: MessageEvent<any>) => {
      if (!event.data) return;
      const frame = event.data[0];
      const micDuration = opusRecorder.encodedSamplePosition / 48000;
      outputWorklet.port.postMessage({
        frame,
        type: "audio",
        micDuration,
      });
    };
    worker.postMessage({
      command: "init",
      bufferLength: (960 * audioContext.sampleRate) / 24000,
      decoderSampleRate: 24000,
      outputBufferSampleRate: audioContext.sampleRate,
      resampleQuality: 0,
    });
    ap.decoder = worker;
  }, []);

  const shutdownAudio = useCallback(() => {
    if (audioProcessorRef.current) {
      const { audioContext, opusRecorder, outputWorklet } =
        audioProcessorRef.current;

      // Disconnect all nodes
      outputWorklet.disconnect();
      audioContext.close();
      opusRecorder.stop();

      // Clear the reference
      audioProcessorRef.current = null;
    }
  }, []);

  return {
    setupAudio,
    shutdownAudio,
    audioProcessor: audioProcessorRef,
    flushOutput,
  };
};
