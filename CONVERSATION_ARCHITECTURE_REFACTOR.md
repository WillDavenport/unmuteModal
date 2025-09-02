# Conversation Architecture Refactor Summary

## Overview
Successfully refactored the codebase from a quest_manager-based architecture to a conversation-based architecture where each `/realtime` connected client has an ongoing Conversation object with persistent websocket connections to STT, LLM, and TTS servers.

## Key Changes

### 1. New Architecture Components

#### `unmute/conversation.py`
- **Conversation class**: Manages persistent websocket connections to STT, LLM, and TTS services
- **ConversationManager class**: Manages multiple conversations for different clients
- Each service runs on its own asyncio task (non-blocking)
- STT completion triggers LLM, LLM completion triggers TTS
- Persistent connections that are cleaned up when conversation closes

#### Architecture Flow:
```
Client connects to /realtime 
    → ConversationManager.create_conversation()
    → Conversation() created with persistent connections to:
        - STT service (running on separate task)
        - LLM service (triggered by STT)
        - TTS service (triggered by LLM)
    → When client disconnects, connections are cleaned up
```

### 2. Preserved Functionality
- ✅ **Pause detection**: Implemented in `determine_pause()` method
- ✅ **Interruptions**: Implemented in `interrupt_bot()` method  
- ✅ **STT processing**: Implemented in `_stt_loop()` method
- ✅ **LLM processing**: Implemented in `_generate_response_task()` method
- ✅ **TTS processing**: Implemented in `_tts_loop()` method
- ✅ **Session management**: Voice settings, instructions, recording preferences
- ✅ **Debug information**: Gradio updates, timing metrics, connection states
- ✅ **Error handling**: Service failures, websocket disconnections
- ✅ **Audio processing**: Frame handling, VAD, silence detection

### 3. Service Threading
Each service runs independently:
- **STT Task**: Processes incoming audio and outputs transcription
- **LLM Task**: Triggered by STT completion, generates response text
- **TTS Task**: Triggered by LLM text generation, produces audio output

### 4. Files Modified

#### Core Architecture:
- `unmute/conversation.py` - **NEW**: Main conversation architecture
- `unmute/quest_manager.py` - **REMOVED**: No longer needed
- `unmute/unmute_handler.py` - **SIMPLIFIED**: Now a backward compatibility wrapper

#### Integration Points:
- `unmute/main_websocket.py` - Updated to use ConversationManager
- `modal_app.py` - Updated to use new conversation architecture
- `unmute/main_gradio.py` - Updated to use Conversation class

### 5. Backward Compatibility
- `UnmuteHandler` class still exists as a wrapper around `Conversation`
- All existing interfaces preserved
- Gradio integration maintained
- Modal deployment compatibility preserved

## Benefits of New Architecture

1. **Cleaner separation of concerns**: Each conversation manages its own state
2. **Better resource management**: Persistent connections reduce overhead
3. **Improved scalability**: Multiple conversations can run independently
4. **Simplified debugging**: Each conversation has its own ID and logging
5. **Thread safety**: Each service runs on its own task without blocking others
6. **Easier maintenance**: Removed complex quest management system

## Migration Notes
- No breaking changes to external APIs
- All functionality preserved from original implementation
- Services maintain the same triggering logic (STT → LLM → TTS)
- Error handling and cleanup improved with proper async task management