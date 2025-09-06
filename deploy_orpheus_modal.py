#!/usr/bin/env python3
"""
Deployment script for Orpheus TTS Modal services
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_modal_auth():
    """Check if Modal is authenticated"""
    try:
        result = subprocess.run(["modal", "token", "show"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Modal authentication verified")
            return True
        else:
            print("❌ Modal not authenticated. Please run: modal token new")
            return False
    except FileNotFoundError:
        print("❌ Modal CLI not found. Please install with: pip install modal")
        return False

def main():
    print("🎯 Orpheus TTS Modal Deployment Script")
    print("This script will deploy the Orpheus TTS services to Modal")
    
    # Check Modal authentication
    if not check_modal_auth():
        sys.exit(1)
    
    # Change to the workspace directory
    workspace_dir = "/workspace"
    if os.path.exists(workspace_dir):
        os.chdir(workspace_dir)
        print(f"📁 Changed to workspace directory: {workspace_dir}")
    else:
        print(f"❌ Workspace directory not found: {workspace_dir}")
        sys.exit(1)
    
    # Deploy the llama.cpp server first (TTS depends on it)
    print("\n" + "="*60)
    print("🦙 Deploying Orpheus llama.cpp server...")
    print("="*60)
    
    if not run_command(
        ["modal", "deploy", "unmute/tts/orpheus_modal.py::llama_app"],
        "Deploying Orpheus llama.cpp server"
    ):
        print("❌ Failed to deploy llama.cpp server")
        sys.exit(1)
    
    # Wait a moment for the first service to fully deploy
    print("⏳ Waiting 10 seconds for llama.cpp server to fully deploy...")
    time.sleep(10)
    
    # Deploy the TTS service
    print("\n" + "="*60)
    print("🗣️ Deploying Orpheus TTS service...")
    print("="*60)
    
    if not run_command(
        ["modal", "deploy", "unmute/tts/orpheus_modal.py::app"],
        "Deploying Orpheus TTS service"
    ):
        print("❌ Failed to deploy TTS service")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("="*80)
    
    print("\n📋 Next Steps:")
    print("1. Update your environment variables to point to the deployed services:")
    print("   - ORPHEUS_TTS_URL: https://willdavenport--orpheus-tts-orpheustts-asgi-app.modal.run")
    print("   - ORPHEUS_LLAMA_ENDPOINT: https://willdavenport--orpheus-llama-server-orpheusllamaserver-asgi-app.modal.run/v1/completions")
    print("\n2. Test the services:")
    print("   curl https://willdavenport--orpheus-tts-orpheustts-asgi-app.modal.run/health")
    print("   curl https://willdavenport--orpheus-llama-server-orpheusllamaserver-asgi-app.modal.run/health")
    print("\n3. Test TTS generation:")
    print('   curl -X POST https://willdavenport--orpheus-tts-orpheustts-asgi-app.modal.run/v1/audio/speech \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"input": "Hello world! This is a test.", "voice": "tara"}\' \\')
    print('     --output test.wav')
    
    print("\n4. The services are now ready to be used by your orchestrator!")

if __name__ == "__main__":
    main()