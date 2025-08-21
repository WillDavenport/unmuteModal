#!/usr/bin/env python3
"""
Deployment script for the Modal voice stack application.

This script helps deploy the voice stack to Modal with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_modal_cli():
    """Check if Modal CLI is installed and authenticated."""
    try:
        result = subprocess.run(["modal", "--help"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Modal CLI not found. Please install it:")
            print("pip install modal")
            return False
        
        # Check authentication
        result = subprocess.run(["modal", "token", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Modal not authenticated. Please run:")
            print("modal token new")
            return False
        
        print("✅ Modal CLI is installed and authenticated")
        return True
    except FileNotFoundError:
        print("❌ Modal CLI not found. Please install it:")
        print("pip install modal")
        return False


def create_modal_secrets():
    """Create required Modal secrets."""
    print("📝 Setting up Modal secrets...")
    
    # Create voice-auth secret for inter-service authentication
    try:
        subprocess.run([
            "modal", "secret", "create", "voice-auth",
            "MODAL_AUTH_TOKEN=public_token",
            "KYUTAI_API_KEY=public_token"
        ], check=True)
        print("✅ Created voice-auth secret")
    except subprocess.CalledProcessError:
        print("⚠️  voice-auth secret may already exist")


def deploy_app():
    """Deploy the Modal app."""
    print("🚀 Deploying Modal app...")
    
    try:
        subprocess.run(["modal", "deploy", "modal_app.py"], check=True)
        print("✅ Modal app deployed successfully!")
        
        print("\n🔗 Your services are now available at:")
        print("- Orchestrator: https://kyutai-labs--voice-stack-orchestratorservice-web.modal.run/ws")
        print("- STT Service: https://kyutai-labs--voice-stack-sttservice-web.modal.run/ws")
        print("- TTS Service: https://kyutai-labs--voice-stack-ttsservice-web.modal.run/ws") 
        print("- LLM Service: https://kyutai-labs--voice-stack-llmservice-web.modal.run/ws")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        return False
    
    return True


def serve_locally():
    """Serve the app locally for development."""
    print("🔧 Starting local development server...")
    
    try:
        subprocess.run(["modal", "serve", "modal_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Local server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Local serve failed: {e}")


def main():
    """Main deployment function."""
    print("🎤 Modal Voice Stack Deployment")
    print("=" * 40)
    
    # Check prerequisites
    if not check_modal_cli():
        sys.exit(1)
    
    # Check if unmute directory exists
    if not Path("unmute").exists():
        print("❌ unmute directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Get deployment mode
    mode = "deploy"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if mode not in ["deploy", "serve"]:
        print("Usage: python deploy_modal.py [deploy|serve]")
        print("  deploy: Deploy to Modal cloud")
        print("  serve:  Run locally for development")
        sys.exit(1)
    
    # Create secrets
    create_modal_secrets()
    
    if mode == "deploy":
        success = deploy_app()
        if success:
            print("\n🎉 Deployment complete!")
            print("You can now use the orchestrator endpoint with your frontend.")
    else:
        serve_locally()


if __name__ == "__main__":
    main()