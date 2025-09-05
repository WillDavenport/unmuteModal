#!/usr/bin/env python3
"""
Deploy the Orpheus FastAPI and llama.cpp Modal services

This script deploys the new Modal-based Orpheus TTS services that replace
the old implementation with the orpheus_fast_api based solution.
"""

import subprocess
import sys
import os


def deploy_service(service_file: str, app_name: str):
    """Deploy a Modal service"""
    print(f"\n{'='*60}")
    print(f"Deploying {app_name}...")
    print(f"{'='*60}")
    
    try:
        # Deploy the Modal app
        result = subprocess.run(
            ["modal", "deploy", service_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Info:", result.stderr)
            
        print(f"✅ {app_name} deployed successfully!")
        
        # Extract the URL from the output
        for line in result.stdout.split('\n'):
            if 'https://' in line and '.modal.run' in line:
                url = line.strip()
                if '→' in url:
                    url = url.split('→')[1].strip()
                print(f"📍 Service URL: {url}")
                return url
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to deploy {app_name}")
        print(f"Error: {e.stderr}")
        sys.exit(1)
        
    return None


def main():
    """Main deployment function"""
    print("🚀 Deploying Orpheus Modal Services")
    print("="*60)
    
    # Check if Modal is installed and authenticated
    try:
        result = subprocess.run(
            ["modal", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Modal CLI version: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ Modal CLI not found. Please install it with: pip install modal")
        sys.exit(1)
    
    # Check authentication
    try:
        result = subprocess.run(
            ["modal", "token", "info"],
            capture_output=True,
            text=True,
            check=True
        )
        if "No token found" in result.stdout or "No token found" in result.stderr:
            print("❌ Not authenticated with Modal. Please run: modal setup")
            sys.exit(1)
        print("✅ Modal authentication verified")
    except subprocess.CalledProcessError:
        print("⚠️ Could not verify Modal authentication. You may need to run: modal setup")
    
    # Deploy the llama.cpp server first
    print("\n" + "="*60)
    print("Step 1: Deploying Orpheus llama.cpp server")
    print("="*60)
    
    llama_url = deploy_service(
        "unmute/tts/orpheus_llama_modal.py",
        "Orpheus llama.cpp Server"
    )
    
    # Deploy the Orpheus FastAPI service
    print("\n" + "="*60)
    print("Step 2: Deploying Orpheus FastAPI TTS service")
    print("="*60)
    
    # Set the llama endpoint environment variable if we got a URL
    if llama_url:
        # Convert https:// to the proper endpoint format
        llama_endpoint = llama_url.replace("https://", "http://") + "/v1/completions"
        os.environ["ORPHEUS_LLAMA_ENDPOINT"] = llama_endpoint
        print(f"📍 Using llama.cpp endpoint: {llama_endpoint}")
    
    fastapi_url = deploy_service(
        "unmute/tts/orpheus_fastapi_modal.py",
        "Orpheus FastAPI TTS Service"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 Deployment Complete!")
    print("="*60)
    
    if llama_url:
        print(f"\n📍 Orpheus llama.cpp Server:")
        print(f"   URL: {llama_url}")
        print(f"   Endpoint: {llama_url}/v1/completions")
        
    if fastapi_url:
        print(f"\n📍 Orpheus FastAPI TTS Service:")
        print(f"   URL: {fastapi_url}")
        print(f"   OpenAI-compatible endpoint: {fastapi_url}/v1/audio/speech")
        print(f"   WebSocket streaming: {fastapi_url.replace('https://', 'wss://')}/v1/audio/speech/stream/ws")
    
    print("\n📝 Configuration:")
    print("   Add these to your environment variables or .env file:")
    print(f"   ORPHEUS_FASTAPI_URL={fastapi_url}")
    if llama_url:
        print(f"   ORPHEUS_LLAMA_ENDPOINT={llama_url}/v1/completions")
    
    print("\n✨ To test the services:")
    print("   1. Test llama.cpp:")
    if llama_url:
        print(f"      curl {llama_url}/health")
    print("   2. Test Orpheus FastAPI:")
    if fastapi_url:
        print(f"      curl {fastapi_url}/health")
    print("\n   3. Generate speech:")
    if fastapi_url:
        print(f'      curl -X POST {fastapi_url}/v1/audio/speech \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"input": "Hello world!", "voice": "tara"}\' \\')
        print('        --output test.wav')
    
    print("\n🎯 Next steps:")
    print("   1. Update your main Modal app to use these services")
    print("   2. Test the WebSocket streaming endpoints")
    print("   3. Configure any custom voices or parameters as needed")
    

if __name__ == "__main__":
    main()