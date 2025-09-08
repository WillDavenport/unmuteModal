#!/usr/bin/env python3
"""
Test script to demonstrate the audio debugging system.

This creates sample log entries that simulate the audio flow through the pipeline
to show how the debug_audio_flow.py script works.
"""

import sys
from debug_audio_flow import analyze_audio_flow, print_audio_flow_report

# Sample log entries simulating a complete audio flow
SAMPLE_LOGS = """
2024-01-15 10:30:00 === ORPHEUS_GENERATION_START ===
2024-01-15 10:30:00 [ORPHEUS] Starting speech generation for text: 'Hello, this is a test message for debugging the audio pipeline...' (voice: tara)
2024-01-15 10:30:00 [ORPHEUS] Text length: 78 characters
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #1: 4800 bytes, 2400 samples
2024-01-15 10:30:01 [ORPHEUS] Running totals: 4800 bytes, 2400 samples
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #2: 4800 bytes, 2400 samples
2024-01-15 10:30:01 [ORPHEUS] Running totals: 9600 bytes, 4800 samples
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #3: 4800 bytes, 2400 samples
2024-01-15 10:30:01 [ORPHEUS] Running totals: 14400 bytes, 7200 samples
2024-01-15 10:30:01 === ORPHEUS_GENERATION_COMPLETE ===
2024-01-15 10:30:01 [ORPHEUS] Generation completed successfully
2024-01-15 10:30:01 [ORPHEUS] Total chunks generated: 3
2024-01-15 10:30:01 [ORPHEUS] Total bytes: 14400
2024-01-15 10:30:01 [ORPHEUS] Total samples: 7200
2024-01-15 10:30:01 [ORPHEUS] Audio duration: 0.30s

2024-01-15 10:30:01 === BACKEND_TTS_RAW_CHUNK_RECEIVED ===
2024-01-15 10:30:01 [BACKEND_TTS] Received raw chunk #1: 4800 bytes
2024-01-15 10:30:01 === BACKEND_TTS_CHUNK_QUEUED ===
2024-01-15 10:30:01 [BACKEND_TTS] Queued audio chunk #1
2024-01-15 10:30:01 [BACKEND_TTS] Chunk samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_RAW_CHUNK_RECEIVED ===
2024-01-15 10:30:01 [BACKEND_TTS] Received raw chunk #2: 4800 bytes
2024-01-15 10:30:01 === BACKEND_TTS_CHUNK_QUEUED ===
2024-01-15 10:30:01 [BACKEND_TTS] Queued audio chunk #2
2024-01-15 10:30:01 [BACKEND_TTS] Chunk samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_RAW_CHUNK_RECEIVED ===
2024-01-15 10:30:01 [BACKEND_TTS] Received raw chunk #3: 4800 bytes
2024-01-15 10:30:01 === BACKEND_TTS_CHUNK_QUEUED ===
2024-01-15 10:30:01 [BACKEND_TTS] Queued audio chunk #3
2024-01-15 10:30:01 [BACKEND_TTS] Chunk samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_STREAM_COMPLETE ===

2024-01-15 10:30:01 === BACKEND_TTS_MESSAGE_YIELDED ===
2024-01-15 10:30:01 [BACKEND_TTS] Yielding audio message #1
2024-01-15 10:30:01 [BACKEND_TTS] Message samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_MESSAGE_YIELDED ===
2024-01-15 10:30:01 [BACKEND_TTS] Yielding audio message #2
2024-01-15 10:30:01 [BACKEND_TTS] Message samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_MESSAGE_YIELDED ===
2024-01-15 10:30:01 [BACKEND_TTS] Yielding audio message #3
2024-01-15 10:30:01 [BACKEND_TTS] Message samples: 2400

2024-01-15 10:30:01 === CONVERSATION_TTS_MESSAGE_RECEIVED ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Received TTS message #1: TTSAudioMessage
2024-01-15 10:30:01 === CONVERSATION_TTS_AUDIO_PROCESSING ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Message samples: 2400
2024-01-15 10:30:01 === CONVERSATION_TTS_TO_OUTPUT_QUEUE ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Audio samples: 2400
2024-01-15 10:30:01 === CONVERSATION_TTS_MESSAGE_RECEIVED ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Received TTS message #2: TTSAudioMessage
2024-01-15 10:30:01 === CONVERSATION_TTS_AUDIO_PROCESSING ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Message samples: 2400
2024-01-15 10:30:01 === CONVERSATION_TTS_TO_OUTPUT_QUEUE ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Audio samples: 2400

2024-01-15 10:30:01 === WEBSOCKET_AUDIO_RECEIVED ===
2024-01-15 10:30:01 [WEBSOCKET] Audio samples: 2400
2024-01-15 10:30:01 === WEBSOCKET_OPUS_ENCODED ===
2024-01-15 10:30:01 [WEBSOCKET] Opus bytes: 120
2024-01-15 10:30:01 === WEBSOCKET_SENDING_AUDIO ===
2024-01-15 10:30:01 [WEBSOCKET] Base64 delta size: 160
2024-01-15 10:30:01 === WEBSOCKET_AUDIO_RECEIVED ===
2024-01-15 10:30:01 [WEBSOCKET] Audio samples: 2400
2024-01-15 10:30:01 === WEBSOCKET_OPUS_ENCODED ===
2024-01-15 10:30:01 [WEBSOCKET] Opus bytes: 120
2024-01-15 10:30:01 === WEBSOCKET_SENDING_AUDIO ===
2024-01-15 10:30:01 [WEBSOCKET] Base64 delta size: 160

2024-01-15 10:30:01 === FRONTEND_AUDIO_RECEIVED ===
2024-01-15 10:30:01 [FRONTEND] Opus bytes after decode: 120
2024-01-15 10:30:01 === FRONTEND_SENDING_TO_DECODER ===
2024-01-15 10:30:01 === FRONTEND_DECODER_OUTPUT ===
2024-01-15 10:30:01 === FRONTEND_TO_AUDIO_WORKLET ===
2024-01-15 10:30:01 === FRONTEND_AUDIO_RECEIVED ===
2024-01-15 10:30:01 [FRONTEND] Opus bytes after decode: 120
2024-01-15 10:30:01 === FRONTEND_SENDING_TO_DECODER ===
2024-01-15 10:30:01 === FRONTEND_DECODER_OUTPUT ===
2024-01-15 10:30:01 === FRONTEND_TO_AUDIO_WORKLET ===
"""

# Sample logs with audio loss (missing some messages)
SAMPLE_LOGS_WITH_LOSS = """
2024-01-15 10:30:00 === ORPHEUS_GENERATION_START ===
2024-01-15 10:30:00 [ORPHEUS] Starting speech generation for text: 'This test shows audio loss in the pipeline...' (voice: tara)
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #1: 4800 bytes, 2400 samples
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #2: 4800 bytes, 2400 samples
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #3: 4800 bytes, 2400 samples
2024-01-15 10:30:01 === ORPHEUS_CHUNK_GENERATED ===
2024-01-15 10:30:01 [ORPHEUS] Generated chunk #4: 4800 bytes, 2400 samples
2024-01-15 10:30:01 === ORPHEUS_GENERATION_COMPLETE ===
2024-01-15 10:30:01 [ORPHEUS] Total chunks generated: 4

2024-01-15 10:30:01 === BACKEND_TTS_RAW_CHUNK_RECEIVED ===
2024-01-15 10:30:01 [BACKEND_TTS] Received raw chunk #1: 4800 bytes
2024-01-15 10:30:01 === BACKEND_TTS_CHUNK_QUEUED ===
2024-01-15 10:30:01 [BACKEND_TTS] Chunk samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_RAW_CHUNK_RECEIVED ===
2024-01-15 10:30:01 [BACKEND_TTS] Received raw chunk #2: 4800 bytes
2024-01-15 10:30:01 === BACKEND_TTS_CHUNK_QUEUED ===
2024-01-15 10:30:01 [BACKEND_TTS] Chunk samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_STREAM_COMPLETE ===

2024-01-15 10:30:01 === BACKEND_TTS_MESSAGE_YIELDED ===
2024-01-15 10:30:01 [BACKEND_TTS] Message samples: 2400
2024-01-15 10:30:01 === BACKEND_TTS_MESSAGE_YIELDED ===
2024-01-15 10:30:01 [BACKEND_TTS] Message samples: 2400

2024-01-15 10:30:01 === CONVERSATION_TTS_MESSAGE_RECEIVED ===
2024-01-15 10:30:01 === CONVERSATION_TTS_AUDIO_PROCESSING ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Message samples: 2400
2024-01-15 10:30:01 === CONVERSATION_TTS_TO_OUTPUT_QUEUE ===
2024-01-15 10:30:01 [CONVERSATION_TTS] Audio samples: 2400

2024-01-15 10:30:01 === WEBSOCKET_AUDIO_RECEIVED ===
2024-01-15 10:30:01 [WEBSOCKET] Audio samples: 2400
2024-01-15 10:30:01 === WEBSOCKET_NO_OPUS_OUTPUT ===

2024-01-15 10:30:01 === BACKEND_TTS_STREAM_ERROR ===
2024-01-15 10:30:01 [BACKEND_TTS] Error in Modal streaming generation: Connection timeout
"""


def test_perfect_flow():
    """Test with perfect audio flow (no losses)."""
    print("Testing perfect audio flow...")
    stats = analyze_audio_flow(SAMPLE_LOGS)
    print_audio_flow_report(stats)


def test_flow_with_losses():
    """Test with audio flow that has losses."""
    print("Testing audio flow with losses...")
    stats = analyze_audio_flow(SAMPLE_LOGS_WITH_LOSS)
    print_audio_flow_report(stats)


def main():
    """Run audio flow tests."""
    if len(sys.argv) > 1 and sys.argv[1] == "--with-losses":
        test_flow_with_losses()
    else:
        test_perfect_flow()


if __name__ == "__main__":
    main()