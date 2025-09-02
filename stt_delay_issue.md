# STT Audio Delay Issue Analysis

## Problem Description

After a VAD (Voice Activity Detection) interruption, the system stops sending audio to STT for approximately 10 seconds, causing a significant delay before speech recognition resumes.

## Timeline from Logs

```
2025-09-01 20:33:13,280 - STT: Sending audio to STT function: 960 samples
2025-09-01 20:33:13,329 - STT: Sending audio to STT function: 960 samples
2025-09-01 20:33:13,329 - Interruption by STT-VAD
2025-09-01 20:33:13,329 - === Clearing TTS output queue because bot was interrupted ===
2025-09-01 20:33:13,330 - === TTS _close() starting ===
2025-09-01 20:33:13,330 - TTS connection state at shutdown: connected

[10 second gap - no STT audio sending]

2025-09-01 20:33:23,332 - === TTS _close() completed successfully ===
2025-09-01 20:33:23,333 - === STT: Sending audio to STT function: 960 samples ===
[STT resumes normal operation]
```

## Root Cause Analysis

### The Blocking Chain

The issue occurs in the `interrupt_bot()` method in `unmute_handler.py`:

```python
async def interrupt_bot(self):
    # ... other code ...
    await self.quest_manager.remove("tts")  # <-- This blocks for ~10 seconds
    await self.quest_manager.remove("llm")
```

### Why This Blocks Audio Processing

1. **VAD Interruption Detected**: STT-VAD detects user speech during bot speaking
2. **interrupt_bot() Called**: Called from within the `receive()` method (line 389)
3. **TTS Quest Removal**: `quest_manager.remove("tts")` is awaited synchronously
4. **TTS Shutdown Blocks**: The removal triggers TTS websocket shutdown via `tts.shutdown()`
5. **WebSocket Close Delay**: `await self.websocket.close()` takes ~10 seconds to complete
6. **receive() Method Blocked**: Since `interrupt_bot()` is called from `receive()`, no new audio frames can be processed
7. **STT Starved**: No audio frames are sent to STT during this period

### The Call Stack

```
receive() 
  -> interrupt_bot() 
    -> quest_manager.remove("tts") 
      -> quest.remove() 
        -> _close(tts) 
          -> tts.shutdown() 
            -> websocket.close()  # <-- 10 second delay here
```

## Impact

- **User Experience**: 10-second delay before the system responds to user input after interruption
- **STT Processing**: No audio data sent to STT during shutdown period
- **Real-time Performance**: Breaks the real-time nature of the conversation system

## Why WebSocket Close Takes 10 Seconds

The TTS websocket close operation likely encounters:
- Network timeouts
- Server-side connection cleanup delays
- WebSocket close handshake timeouts
- Modal.run service connection management overhead

## Solution Approaches

### Option 1: Asynchronous Quest Removal
Make TTS quest removal non-blocking during interruption:

```python
async def interrupt_bot(self):
    # ... other code ...
    
    # Don't await TTS removal - let it happen in background
    asyncio.create_task(self.quest_manager.remove("tts"))
    await self.quest_manager.remove("llm")  # LLM removal is typically fast
```

### Option 2: Separate Audio Processing from Quest Management
Decouple the audio processing loop from quest lifecycle management to prevent blocking.

### Option 3: Timeout-Based Quest Removal
Add timeouts to quest removal operations:

```python
async def interrupt_bot(self):
    # ... other code ...
    
    try:
        await asyncio.wait_for(self.quest_manager.remove("tts"), timeout=1.0)
    except asyncio.TimeoutError:
        logger.warning("TTS removal timed out, continuing with background cleanup")
        asyncio.create_task(self.quest_manager.remove("tts"))
```

## Recommended Fix

**Option 1** is the most straightforward solution. The TTS shutdown doesn't need to be synchronous during interruption since:
- The TTS output queue is already cleared
- The websocket connection will be closed in the background
- Audio processing can resume immediately
- Any remaining TTS cleanup can happen asynchronously

## Implementation

Modify `interrupt_bot()` in `unmute_handler.py`:

```python
async def interrupt_bot(self):
    if self.chatbot.conversation_state() != "bot_speaking":
        logger.error(f"Can't interrupt bot when conversation state is {self.chatbot.conversation_state()}")
        raise RuntimeError(
            "Can't interrupt bot when conversation state is "
            f"{self.chatbot.conversation_state()}"
        )

    await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

    if self._clear_queue is not None:
        self._clear_queue()
    logger.info("=== Clearing TTS output queue because bot was interrupted ===")
    self.output_queue = asyncio.Queue()

    await self.output_queue.put(
        (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
    )
    await self.output_queue.put(ora.UnmuteInterruptedByVAD())

    # Make TTS removal non-blocking to prevent audio processing delays
    asyncio.create_task(self.quest_manager.remove("tts"))
    await self.quest_manager.remove("llm")
```

This change would eliminate the 10-second delay and allow STT to continue processing audio immediately after interruption.
