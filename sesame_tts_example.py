#!/usr/bin/env python3
"""
Sesame_TTS Modal Service Example

This example shows how to use the Sesame_TTS service that implements
the CSM (Conversational Speech Model) from SesameAI Labs.

The service provides:
1. Simple text-to-speech generation
2. Context-aware speech generation for conversations
3. Both HTTP and WebSocket interfaces

Requirements:
- Modal account and authentication
- HuggingFace token for accessing CSM models
- GPU resources (L4 recommended)
"""

import modal

def example_simple_tts():
    """Example of simple text-to-speech generation"""
    
    # Get the deployed app
    app = modal.App.lookup("voice-stack")
    
    # Get the Sesame TTS service
    sesame_tts = app.cls.SesameTTSService()
    
    # Generate speech from text
    text = "Hello! This is a demonstration of the Sesame CSM text-to-speech model."
    
    print(f"Generating speech for: {text}")
    
    audio_bytes = sesame_tts.generate_speech.remote(
        text=text,
        speaker=0,  # Speaker ID (0 or 1)
        max_audio_length_ms=15000,  # 15 seconds max
        temperature=0.9,  # Sampling temperature
        topk=50  # Top-k sampling
    )
    
    # Save the audio
    with open("example_output.wav", "wb") as f:
        f.write(audio_bytes)
    
    print(f"Audio saved to example_output.wav ({len(audio_bytes)} bytes)")


def example_conversational_tts():
    """Example of context-aware conversational TTS"""
    
    # This would require implementing audio loading and context management
    # For now, this is a placeholder showing the API structure
    
    app = modal.App.lookup("voice-stack")
    sesame_tts = app.cls.SesameTTSService()
    
    # In a real implementation, you would:
    # 1. Load previous conversation audio segments
    # 2. Convert them to the required format
    # 3. Pass them as context
    
    context_segments = [
        # Example structure - in practice you'd load actual audio tensors
        # {
        #     'text': 'Previous utterance',
        #     'speaker': 0,
        #     'audio_tensor': torch.tensor(...)
        # }
    ]
    
    audio_bytes = sesame_tts.generate_speech_with_context.remote(
        text="This response takes into account our previous conversation.",
        speaker=1,
        context_segments=context_segments,
        max_audio_length_ms=12000,
        temperature=0.8,
        topk=45
    )
    
    with open("conversational_output.wav", "wb") as f:
        f.write(audio_bytes)
    
    print(f"Conversational audio saved to conversational_output.wav")


if __name__ == "__main__":
    print("=== Sesame_TTS Modal Service Examples ===\n")
    
    print("1. Simple TTS Example:")
    try:
        example_simple_tts()
        print("✓ Simple TTS example completed\n")
    except Exception as e:
        print(f"✗ Simple TTS example failed: {e}\n")
    
    print("2. Conversational TTS Example:")
    try:
        example_conversational_tts()
        print("✓ Conversational TTS example completed\n")
    except Exception as e:
        print(f"✗ Conversational TTS example failed: {e}\n")
    
    print("=== Examples completed ===")
    print("\nTo deploy the service:")
    print("1. Ensure you have Modal authentication set up")
    print("2. Set up HuggingFace token in Modal secrets")
    print("3. Run: modal deploy modal_app.py")
    print("4. The Sesame_TTS service will be available at:")
    print("   https://your-username--voice-stack-sesamettservice-web.modal.run")