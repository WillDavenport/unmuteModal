# Orpheus WebSocket Connection Issues

## Problem Summary

The Orpheus TTS Modal service is experiencing HTTP 403 errors when clients attempt to establish WebSocket connections, despite the service being deployed and responding to HTTP requests correctly.

## Error Details

```
=== SERVICE_DISCOVERY: [tts] Failed to connect to wss://willdavenport--orpheus-tts-orpheustts-asgi-app-dev.modal.run/ws: server rejected WebSocket connection: HTTP 403 (took 279.7ms) ===
2025-09-07 05:12:41,239 - unmute.tts.text_to_speech - ERROR - Failed to connect to Orpheus FastAPI TTS: server rejected WebSocket connection: HTTP 403
```

## Investigation Results

### ✅ Service Health Check
- HTTP endpoints work correctly
- Health check returns: `{"status":"healthy","service":"orpheus-tts-modal"}`
- Service is deployed and running

### ✅ Direct WebSocket Connection Test
When testing the WebSocket connection directly with Python:

```python
import asyncio
import websockets
import json

async def test_websocket():
    uri = 'wss://willdavenport--orpheus-tts-orpheustts-asgi-app-dev.modal.run/ws'
    async with websockets.connect(uri) as websocket:
        message = {'type': 'text', 'text': 'Hello, this is a test.', 'voice': 'tara'}
        await websocket.send(json.dumps(message))
        response = await websocket.recv()
        print(f'Received: {response}')
```

**Result**: ✅ **CONNECTION SUCCESSFUL**
- WebSocket connection established without issues
- Service responded with expected audio format message:
  ```json
  {"type": "audio_format", "sample_rate": 24000, "channels": 1, "sample_width": 2, "format": "raw-pcm"}
  ```

### Applied Fixes

1. **CORS Middleware** - Added to FastAPI app:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **OPTIONS Handler** - Added for WebSocket endpoint:
   ```python
   @app.options("/ws")
   async def websocket_options():
       return {"message": "WebSocket endpoint available"}
   ```

3. **HEAD Support** - Added for root endpoint to fix 405 errors:
   ```python
   @app.get("/")
   @app.head("/")
   async def root():
       # ...
   ```

## Current Status

### What Works
- ✅ Direct WebSocket connections from Python scripts
- ✅ HTTP REST API endpoints
- ✅ Service health checks
- ✅ CORS middleware is properly configured

### What Doesn't Work
- ❌ WebSocket connections from the client application
- ❌ Service discovery still reports HTTP 403 errors

## Potential Root Causes

### 1. Client-Side Connection Issues
The fact that direct WebSocket connections work but the client application fails suggests the issue might be:

- **Headers**: The client may be sending additional headers that cause authentication issues
- **Origin**: The client's origin might not match what the service expects
- **Protocol**: There might be a WebSocket protocol mismatch
- **Authentication**: The client might be sending authentication tokens that are rejected

### 2. Service Discovery Layer
The error logs show `unmute.service_discovery` and `unmute.tts.text_to_speech` are involved:

- The service discovery layer might be adding headers or parameters that cause issues
- There could be a connection pooling or retry mechanism interfering
- The WebSocket client implementation in `OrpheusTextToSpeech` might have configuration issues

### 3. Modal Platform Specifics
- Modal might have specific requirements for WebSocket connections that aren't documented
- There could be load balancing or proxy issues between the client and the actual service
- Authentication at the Modal platform level might be interfering

## Next Steps for Investigation

### 1. Compare Working vs Non-Working Connections
- Capture the exact headers and parameters sent by both the working Python test and the failing client
- Use network debugging tools to see the difference in WebSocket handshake requests

### 2. Check Client Implementation
- Review `unmute/unmute/tts/text_to_speech.py` - specifically the `OrpheusTextToSpeech.start_up()` method
- Look at how headers are constructed for Modal vs non-Modal services
- Check if there are authentication tokens being sent

### 3. Test Intermediate Solutions
- Try connecting to the service using the same client code but with different configurations
- Test with and without authentication headers
- Try different WebSocket client libraries

### 4. Modal Platform Investigation
- Check Modal documentation for WebSocket-specific requirements
- Look for Modal-specific authentication or authorization mechanisms
- Test with other Modal WebSocket services to see if this is a general issue

## Code Locations

### Service Implementation
- `unmute/unmute/tts/orpheus_modal.py` - Modal service with WebSocket endpoint

### Client Implementation  
- `unmute/unmute/tts/text_to_speech.py` - `OrpheusTextToSpeech` class, `start_up()` method

### Service Discovery
- Connection attempts logged by `unmute.service_discovery`
- Error handling in the service discovery layer

## Deployment Information

- **Service URL**: `wss://willdavenport--orpheus-tts-orpheustts-asgi-app-dev.modal.run/ws`
- **Deployment Status**: ✅ Successfully deployed with CORS fixes
- **Last Updated**: After adding CORS middleware and OPTIONS/HEAD handlers

## Conclusion

This is a puzzling issue where the WebSocket service works perfectly when tested directly, but fails when accessed through the application's service discovery and client implementation. The issue is likely in the client-side connection logic or additional headers/authentication being sent by the application that aren't present in our direct test.

The next step should be to examine the exact difference between the working direct connection and the failing application connection by comparing request headers, parameters, and connection logic.
