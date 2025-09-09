# LLM Investigation: Repetitive Response and Old Message Handling Issues

## Summary

The LLM is exhibiting repetitive behavior, saying "Hello" or "Bonjour" every time, responding to old messages, and repeating responses to the 2nd user turn on the 3rd turn. This investigation identifies the root causes and provides solutions.

## Root Causes Identified

### 1. **Automatic "Hello" Message Injection**

**Location**: `unmute/llm/chatbot.py:89` and `unmute/llm/llm_utils.py:45`

**Issue**: The system automatically injects dummy "Hello" messages in two different places:

1. **In `chatbot.py`**: When there are only 1-2 messages in chat history, a `{"role": "user", "content": "Hello!"}` is added
2. **In `llm_utils.py`**: When preprocessing messages, if there's a system message followed by an assistant message (or None), another `{"role": "user", "content": "Hello."}` is inserted

**Code Evidence**:
```python
# unmute/llm/chatbot.py:85-90
messages = [
    self.chat_history[0],
    # Some models, like Gemma, don't like it when there is no user message
    # so we add one.
    {"role": "user", "content": "Hello!"},
]

# unmute/llm/llm_utils.py:42-45  
if role_at(0) == "system" and role_at(1) in [None, "assistant"]:
    # Some LLMs, like Gemma, get confused if the assistant message goes before user
    # messages, so add a dummy user message.
    output = [output[0]] + [{"role": "user", "content": "Hello."}] + output[1:]
```

### 2. **System Prompt Forces Greeting Responses**

**Location**: `unmute/llm/system_prompt.py:29-30` and `unmute/llm/system_prompt.py:118-119`

**Issue**: The system prompt explicitly instructs the LLM to respond with greetings:

```python
# Line 29-30
"As your first message, repond to the user's message with a greeting and some kind of conversation starter."

# Line 118-119 (in SMALLTALK_INSTRUCTIONS)
"Repond to the user's message with a greeting and some kind of conversation starter."
```

### 3. **Initial Response Generation Logic**

**Location**: `unmute/conversation_handler.py:119-126`

**Issue**: The system automatically generates an initial response when:
- Chat history has only 1 message (system prompt)
- Instructions have been set

```python
# Handle initial response generation
if (
    len(self.conversation.chatbot.chat_history) == 1
    # Wait until the instructions are updated. A bit hacky
    and self.conversation.chatbot.get_instructions() is not None
):
    logger.info("Generating initial response.")
    await self.conversation._generate_response()
```

This triggers before any real user input, causing the LLM to respond to the injected "Hello" messages.

### 4. **Message State Management Issues**

**Location**: `unmute/llm/chatbot.py:21-37`

**Issue**: The conversation state logic may not properly distinguish between:
- Real user messages 
- Injected dummy messages
- Empty messages used for state transitions

The `conversation_state()` method determines state based on the last message, but doesn't account for the artificial nature of some messages.

### 5. **Message Preprocessing Duplication**

**Location**: `unmute/llm/llm_utils.py:18-60`

**Issue**: The `preprocess_messages_for_llm()` function processes the entire chat history each time, potentially:
- Re-adding dummy messages
- Not properly handling message deduplication
- Processing old messages that should be ignored

## Impact Analysis

1. **"Hello/Bonjour" every time**: Caused by automatic greeting injection + system prompt instructions
2. **Responds to old messages**: Message preprocessing includes entire history without proper filtering
3. **Repeats 2nd response on 3rd turn**: State management confusion between real and dummy messages
4. **Conversation flow disruption**: Users experience unnatural, repetitive interactions

## Recommended Solutions

### 1. **Remove Automatic Greeting Injection**

**Priority**: High

- Remove or conditionally disable the automatic "Hello" message injection in both locations
- Add a flag to track whether a real user message has been received
- Only inject dummy messages when absolutely necessary for model compatibility

### 2. **Modify System Prompt**

**Priority**: High

- Remove the mandatory greeting instruction from the system prompt
- Make greeting behavior contextual rather than forced
- Allow the LLM to respond naturally to actual user input

### 3. **Improve Message State Tracking**

**Priority**: Medium

- Add metadata to distinguish between real and artificial messages
- Implement proper message filtering to exclude dummy messages from LLM context
- Track conversation state based on real user interactions only

### 4. **Fix Initial Response Logic**

**Priority**: High

- Delay initial response generation until actual user input is received
- Remove automatic response generation based on chat history length alone
- Wait for real audio/text input before triggering LLM responses

### 5. **Enhance Message Preprocessing**

**Priority**: Medium

- Implement proper message deduplication
- Add conversation turn tracking to prevent processing old messages
- Filter out artificial messages from LLM context

## Implementation Priority

1. **Immediate fixes** (High Priority):
   - Disable automatic greeting injection
   - Modify system prompt to remove forced greetings
   - Fix initial response generation logic

2. **Follow-up improvements** (Medium Priority):
   - Enhance message state tracking
   - Improve preprocessing logic
   - Add proper conversation turn management

## Testing Recommendations

1. Test conversation flow without automatic greetings
2. Verify that LLM responds appropriately to actual user input
3. Check that conversation state transitions work correctly
4. Ensure no message duplication or old message processing
5. Test multi-turn conversations for proper context management

## Files Requiring Changes

- `unmute/llm/chatbot.py` - Remove automatic "Hello!" injection
- `unmute/llm/llm_utils.py` - Fix message preprocessing logic  
- `unmute/llm/system_prompt.py` - Remove forced greeting instructions
- `unmute/conversation_handler.py` - Fix initial response generation
- `unmute/conversation.py` - Improve message state management

This investigation reveals that the repetitive LLM behavior is caused by multiple layers of artificial message injection and forced greeting behavior, rather than a single bug. The solution requires coordinated changes across the conversation handling system.