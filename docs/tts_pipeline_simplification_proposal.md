# Simplified TTS Audio Pipeline Proposal

## Purpose
Answer: Why do we have multiple queues and layers between Orpheus TTS and audio playback, and can we simplify by pushing Orpheus audio straight to the frontend and using backend VAD to interrupt?

- The existing pipeline adds layers (generation, adaptation, cadence control, encoding, transport, decode, output) to handle timing, encoding, backpressure, interruption, and monitoring.
- This proposal outlines a simplified architecture that reduces queues and intermediaries, shifting to a "frontend-first" audio streaming model while preserving critical behavior: fast first audio, responsive interruption, and acceptable lip-sync/latency.

## Current vs Proposed (High Level)

- Current: Orpheus → backend stream adapter → internal queues (timing/cadence) → encoder (Opus) → WebSocket → frontend decoder worker → audio worklet queue → output
- Proposed: Orpheus → backend Opus encoder → WebSocket → frontend decoder worker → audio worklet output
  - Remove: long-lived intermediate queues and cadence schedulers in the backend; no backend buffering
  - Keep: a small jitter buffer on frontend worklet to avoid underruns; Opus encoding for bandwidth; VAD/interrupt signaling path

## Why all the queues existed

- Backpressure smoothing: Prevent bursty upstream (model) from overflowing downstream (encoder/websocket/client).
- Timing/cadence shaping: Smooth irregular production to avoid jitter during playback.
- Interruption cut-through: Provide a place to drop superseded audio upon barge-in without waiting for upstream to stop.
- Cross-component isolation: Decouple model stalls from websocket and client playback.
- Monitoring/recovery: Measure throughput and detect stuck states.

These are valid concerns, but most can be handled with much smaller buffers and clearer ownership at only two boundaries: encode/send and frontend playback.

## Minimalistic Architecture

1) Backend: Stream-normalize and encode
- Receive raw PCM chunks from Orpheus stream
- Encode each incoming Orpheus chunk to Opus immediately (no re-batching)
- Send `response.audio.delta` messages to the client without additional timing queues
- No backend buffering; rely on WebSocket backpressure; tag deltas with `response_id` so the client can ignore late chunks after an interrupt

2) Backend: Interruption and VAD
- VAD continuously runs on mic stream
- On interrupt: send `response.interrupted` event to client; stop emitting new audio; cancel or pause TTS generation; ignore any late-arriving chunks

3) Frontend: Jitter buffer and playback
- Decoder worker decodes Opus deltas into PCM frames
- AudioWorklet maintains a small jitter buffer (e.g., initial 80–120 ms, target 60–120 ms) for smooth playback
- On `response.interrupted`: worklet immediately drops its frames and stops speaking

4) Transport
- Single WebSocket bi-directional channel for audio deltas and control events

## Event Model (selected)

- response.audio.delta: base64(Opus)
- response.audio.start: optional, indicates first packet of a response
- response.audio.end: optional, indicates natural end of a response
- response.interrupted: indicates backend barge-in detected; client should flush output buffers
- response.error: error string + optional telemetry id

## Backpressure and Flow Control

- Backend does not buffer; if the client is slow, WebSocket backpressure naturally slows the producer; on sustained slowness, cancel/stop TTS.
- Client maintains a small jitter buffer only; if underrun happens, allow momentary glitches instead of building latency.

## What functionality is lost by removing queues?

- Fine-grained server-side cadence shaping (prosody-timed pacing) — moved to frontend jitter buffer; acceptable for real-time voice.
- Large server-side buffer enabling temporary network outages — now outages will cause audible glitches sooner; mitigate with 150–250 ms frontend buffer.
- Server-side late drop-in of interruptions — with no backend buffer and ≤ 120 ms frontend jitter, worst-case overrun before stop ≈ frontend jitter (e.g., ~120 ms), which is generally acceptable for barge-in UX.

## Why not send raw PCM straight from Orpheus to frontend?

- Bandwidth: Opus saves 10–20x vs raw PCM and lowers stutter risk.
- Browser handling: Existing decoder worker and worklet already expect Opus. Keeping Opus maintains compatibility and reduces code changes.
- Framing: Opus frames give natural pacing without extra queues.

## Would frontend-only playback with VAD interrupt be sufficient?

Yes, if we:
- Use a small frontend jitter buffer (≤ 120 ms)
- Deliver immediate deltas as soon as encoded (no cadence queue)
- On interrupt, send `response.interrupted` promptly and flush the frontend buffer
- Cancel/pause Orpheus stream on interrupt to stop wasting GPU

This preserves: fast start, low-latency barge-in, simple control flow.

## Concrete Implementation Plan

Phase 1: Backend tightening
- Replace the current multi-queue pipeline in `unmute/tts/text_to_speech.py` with a single producer that:
  - Reads Orpheus PCM stream iteratively
  - Encodes each Orpheus-provided chunk immediately and emits `response.audio.delta`
- Add explicit interrupt hook to cancel Orpheus task and stop emitting audio

Phase 2: Frontend adjustments
- Ensure `audio-output-processor.js` initial buffer ≤ 120 ms; target buffer ~80–100 ms
- Support explicit `response.interrupted` control message to clear `frames` and reset state
- Keep existing decoder worker; no change to base64 transport

Phase 3: Control and telemetry
- Emit `response.audio.start` and `response.audio.end` markers
- Log end-to-end timestamps at: model first token, first delta sent, first frame played (client)
- Add counters for dropped frames on interrupt and underruns

Phase 4: Clean-up
- Remove legacy cadence/holding queues once parity confirmed
- Keep watchdogs (message flow, encoder health) with simplified metrics

## API Contract (sketch)

- Interrupt
  - Client receives: `{ type: "response.interrupted", reason?: string }`
  - Client action: `decoder.reset(); worklet.flush();` and stop speaking immediately
- Start/End markers (optional)
  - `{ type: "response.audio.start", response_id }`
  - `{ type: "response.audio.end", response_id }`

## Tuning Defaults

- Sample rate: Keep 24 kHz end-to-end (Orpheus native), resample to AudioContext rate on frontend as today
- Packetization: unchanged; encode per Orpheus chunk (no additional batching)
- Frontend jitter: 80–120 ms initial; maintain ~100 ms when possible

## Risks and Mitigations

- Increased sensitivity to network jitter
  - Mitigation: Slightly larger frontend jitter buffer; auto-adjust within 60–150 ms window
- Audible cuts on interrupt (desired behavior but noticeable)
  - Mitigation: Optional short cross-fade (client-side), but default to hard stop for clarity
- Model cancellation delay
  - Mitigation: Cooperative cancel with Orpheus stream; ignore late-arriving audio after interrupt id
- Opus encoder threading may become a hotspot
  - Mitigation: Keep it in a background thread or process; adjust encoder complexity if needed

## Migration Steps

1) Implement encode-per-chunk path in `unmute/tts/text_to_speech.py` (no backend buffering)
2) Add `response.interrupted` emission and plumb interrupt handling end-to-end (backend → frontend)
3) Update frontend worklet to support `flush` command and reduce initial buffer to ≤ 120 ms
4) Test end-to-end locally and in staging; capture latency and barge-in metrics
5) Remove legacy queues and timing shapers after parity and stability are confirmed

## Answering the original questions succinctly

- Point of the queues: smooth bursty production, shape cadence, enable fast interrupt, and isolate components.
- Can we just send all audio to frontend and rely on VAD to interrupt? Yes, if we keep minimal buffers and Opus framing, and implement immediate flush on interrupt. That removes most backend complexity without losing critical behavior.
- Trade-offs: Lose large server-side buffering and fine pacing; gain lower latency and simpler code. With small jitter buffers, UX remains strong for real-time voice.