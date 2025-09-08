# Modal Audio Flow Debugging Guide

This guide explains how to debug audio cutoff issues in Modal-deployed TTS systems using the integrated debugging endpoints and logging system.

## Overview

Since Modal functions run in a serverless environment, we can't use traditional command-line logging approaches. Instead, this system provides:

1. **Built-in Debug Endpoints** - HTTP endpoints to get real-time statistics
2. **Modal Logging Integration** - Debug events appear in Modal's log viewer
3. **Persistent State Tracking** - Statistics persist across function calls
4. **Frontend Integration** - Frontend can report its own statistics

## Quick Start

### 1. Deploy with Debug Endpoints

When you deploy your Modal app, the debug endpoints are automatically included:

```python
# In your orpheus_modal.py, debug endpoints are automatically added
from modal_audio_debug import ModalAudioDebugger
debug_stats_endpoint, reset_debug_endpoint = ModalAudioDebugger.create_debug_endpoint(orpheus_tts_app)
```

### 2. Access Debug Information

Once deployed, you can access debug information via HTTP:

```bash
# Get current audio flow statistics
curl https://your-modal-app-url/audio_debug_stats

# Reset statistics (useful between test sessions)
curl -X POST https://your-modal-app-url/reset_audio_debug
```

### 3. View Modal Logs

Debug events also appear in Modal's log viewer:

1. Go to your Modal dashboard
2. Click on your app
3. View the logs for real-time debug output

## Debug Endpoint Response

The `/audio_debug_stats` endpoint returns comprehensive pipeline statistics:

```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "pipeline_efficiency": 85.5,
  "total_events": 156,
  "stage_stats": {
    "orpheus": {
      "chunks_generated": 10,
      "generation_complete": 1,
      "errors": 0
    },
    "backend": {
      "chunks_received": 10,
      "chunks_queued": 10,
      "messages_yielded": 9,
      "stream_complete": 1
    },
    "conversation": {
      "messages_received": 8,
      "loop_complete": 1
    },
    "websocket": {
      "audio_received": 8,
      "opus_encoded": 7,
      "messages_sent": 7,
      "no_opus_output": 1
    },
    "frontend": {
      "messages_received": 6,
      "sent_to_decoder": 6,
      "decoder_frames": 6
    }
  },
  "losses": [
    {
      "stage": "Backend → Conversation",
      "loss_count": 1,
      "loss_percentage": 11.1
    },
    {
      "stage": "WebSocket → Frontend", 
      "loss_count": 1,
      "loss_percentage": 14.3
    }
  ],
  "recent_events": [
    {
      "timestamp": "2024-01-15T10:30:00.123Z",
      "stage": "orpheus",
      "event": "chunk_generated",
      "data": {
        "chunk_number": 1,
        "chunk_bytes": 4800,
        "chunk_samples": 2400
      }
    }
  ]
}
```

## Debugging Workflow

### 1. Start a Debug Session

```bash
# Reset statistics before testing
curl -X POST https://your-modal-app-url/reset_audio_debug
```

### 2. Reproduce the Issue

Use your frontend to trigger TTS generation and reproduce the audio cutoff.

### 3. Check Statistics

```bash
# Get current statistics
curl https://your-modal-app-url/audio_debug_stats | jq .
```

### 4. Analyze the Results

Look for:
- **Low pipeline_efficiency** (< 90%) indicates significant losses
- **High loss_count** in specific stages shows where audio is dropped
- **Errors** in any stage indicate failures
- **Recent_events** show the last few operations

### 5. Check Modal Logs

In Modal's dashboard, look for debug output like:
```
=== AUDIO_DEBUG: ORPHEUS_CHUNK_GENERATED ===
[AUDIO_DEBUG] chunk_number: 1
[AUDIO_DEBUG] chunk_bytes: 4800
[AUDIO_DEBUG] chunk_samples: 2400
```

## Common Issues and Solutions

### 1. Audio Cutoff Mid-Sentence

**Symptoms:**
- High `orpheus.chunks_generated` but low `frontend.messages_received`
- Losses in middle stages (backend, conversation, websocket)

**Debug Steps:**
```bash
# Check if Orpheus is generating complete audio
curl https://your-modal-app-url/audio_debug_stats | jq '.stage_stats.orpheus'

# Check where the loss occurs
curl https://your-modal-app-url/audio_debug_stats | jq '.losses'
```

**Common Causes:**
- Task cancellation during interruption
- WebSocket disconnection
- Conversation loop early termination

### 2. No Audio Generation

**Symptoms:**
- Zero or very low `orpheus.chunks_generated`
- `orpheus.errors` > 0

**Debug Steps:**
```bash
# Check Orpheus errors
curl https://your-modal-app-url/audio_debug_stats | jq '.stage_stats.orpheus.errors'

# Check recent events for error details
curl https://your-modal-app-url/audio_debug_stats | jq '.recent_events[] | select(.event == "generation_error")'
```

**Common Causes:**
- Modal container scaling issues
- Model loading failures
- Authentication problems

### 3. WebSocket Issues

**Symptoms:**
- High `websocket.no_opus_output` count
- Low `websocket.messages_sent` vs `websocket.audio_received`

**Debug Steps:**
```bash
# Check WebSocket efficiency
curl https://your-modal-app-url/audio_debug_stats | jq '.stage_stats.websocket'
```

**Common Causes:**
- Opus encoding buffer issues
- Network connectivity problems
- Client disconnections

## Frontend Integration

### Add Frontend Debugging

Update your frontend to report statistics:

```typescript
// In your frontend WebSocket message handler
if (data.type === "response.audio.delta") {
  // Report to debug endpoint
  fetch(`${MODAL_APP_URL}/log_frontend_event`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event: 'message_received',
      data: {
        opus_bytes: opus.length,
        timestamp: Date.now()
      }
    })
  });
  
  // Continue with normal processing
  const opus = base64DecodeOpus(data.delta);
  ap.decoder.postMessage({
    command: "decode",
    pages: [opus]
  });
}
```

### Create Frontend Debug Endpoint

Add this to your Modal app:

```python
@app.function()
@modal.web_endpoint(method="POST")
def log_frontend_event():
    """Accept frontend debug events."""
    from modal_audio_debug import log_frontend_event
    from fastapi import Request
    
    async def endpoint(request: Request):
        data = await request.json()
        log_frontend_event(data['event'], data.get('data', {}))
        return {"status": "logged"}
    
    return endpoint
```

## Monitoring and Alerting

### Continuous Monitoring

Create a monitoring script:

```python
import requests
import time

def monitor_audio_pipeline(app_url):
    while True:
        try:
            response = requests.get(f"{app_url}/audio_debug_stats")
            stats = response.json()
            
            efficiency = stats.get('pipeline_efficiency', 0)
            if efficiency < 80:
                print(f"⚠️  Low pipeline efficiency: {efficiency:.1f}%")
                
            for loss in stats.get('losses', []):
                if loss['loss_percentage'] > 10:
                    print(f"⚠️  High loss in {loss['stage']}: {loss['loss_percentage']:.1f}%")
                    
        except Exception as e:
            print(f"Monitoring error: {e}")
            
        time.sleep(30)  # Check every 30 seconds

# Usage
monitor_audio_pipeline("https://your-modal-app-url")
```

### Performance Dashboards

Create a simple dashboard:

```python
import streamlit as st
import requests
import plotly.graph_objects as go

def create_dashboard(app_url):
    st.title("Audio Pipeline Debug Dashboard")
    
    # Get current stats
    response = requests.get(f"{app_url}/audio_debug_stats")
    stats = response.json()
    
    # Pipeline efficiency gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = stats['pipeline_efficiency'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Pipeline Efficiency %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    st.plotly_chart(fig)
    
    # Stage statistics
    st.subheader("Stage Statistics")
    for stage, stage_stats in stats['stage_stats'].items():
        st.write(f"**{stage.title()}**")
        st.json(stage_stats)
```

## Advanced Debugging

### Session Tracking

Use session tracking for complex debugging:

```python
from modal_audio_debug import AudioDebugSession

# In your TTS generation function
with AudioDebugSession("user_123_request_456") as session:
    # Your TTS generation code here
    pass
```

### Custom Events

Add custom debug events:

```python
from modal_audio_debug import log_backend_event

# Log custom events
log_backend_event("custom_checkpoint", {
    "checkpoint_name": "before_opus_encoding",
    "buffer_size": len(audio_buffer),
    "timestamp": time.time()
})
```

### Error Context

Debug events include error context:

```python
try:
    # Your code here
    pass
except Exception as e:
    log_backend_event("processing_error", {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "context": "audio_processing_stage"
    })
    raise
```

This Modal-compatible debugging system provides comprehensive visibility into your audio pipeline without requiring command-line access, making it perfect for serverless deployments.