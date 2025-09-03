#!/usr/bin/env python3
"""
Deployment script for Sesame_TTS Modal service

This script helps deploy the Sesame_TTS service to Modal with proper configuration.
"""

import os
import subprocess
import sys


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("=== Checking Prerequisites ===")
    
    # Check if modal is installed
    try:
        import modal
        print("✓ Modal is installed")
    except ImportError:
        print("✗ Modal is not installed. Install with: pip install modal")
        return False
    
    # Check if user is authenticated with Modal
    try:
        result = subprocess.run(["modal", "token", "show"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Modal authentication is configured")
        else:
            print("✗ Modal authentication is not configured. Run: modal token new")
            return False
    except Exception as e:
        print(f"✗ Could not check Modal authentication: {e}")
        return False
    
    # Check if HuggingFace secret is configured
    try:
        result = subprocess.run(["modal", "secret", "list"], 
                              capture_output=True, text=True, timeout=10)
        if "huggingface-secret" in result.stdout:
            print("✓ HuggingFace secret is configured")
        else:
            print("⚠ HuggingFace secret not found. You may need to configure it:")
            print("  modal secret create huggingface-secret HF_TOKEN=your_token_here")
            print("  This is required for accessing the CSM models.")
    except Exception as e:
        print(f"⚠ Could not check HuggingFace secret: {e}")
    
    return True


def deploy_service():
    """Deploy the Sesame_TTS service to Modal"""
    print("\n=== Deploying Sesame_TTS Service ===")
    
    try:
        # Deploy the modal app
        result = subprocess.run(
            ["modal", "deploy", "modal_app.py"],
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            print("✓ Deployment successful!")
            print("\nDeployment output:")
            print(result.stdout)
            
            # Extract and display service URLs
            lines = result.stdout.split('\n')
            for line in lines:
                if "https://" in line and "sesametts" in line.lower():
                    print(f"\n🚀 Sesame_TTS Service URL: {line.strip()}")
            
        else:
            print("✗ Deployment failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Deployment timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Deployment error: {e}")
        return False
    
    return True


def test_deployment():
    """Test the deployed service"""
    print("\n=== Testing Deployment ===")
    
    try:
        # Import and test the service
        import modal
        
        app = modal.App.lookup("voice-stack", create_if_missing=False)
        if app is None:
            print("✗ Could not find deployed app 'voice-stack'")
            return False
        
        print("✓ Found deployed app")
        
        # Try to get the service class
        try:
            sesame_tts = app.cls.SesameTTSService()
            print("✓ Sesame_TTS service class found")
            
            # Test simple generation
            print("Testing speech generation...")
            audio_bytes = sesame_tts.generate_speech.remote(
                text="Hello from Sesame TTS deployment test!",
                speaker=0,
                max_audio_length_ms=5000
            )
            
            if audio_bytes and len(audio_bytes) > 0:
                print(f"✓ Speech generation successful ({len(audio_bytes)} bytes)")
                
                # Save test audio
                with open("deployment_test.wav", "wb") as f:
                    f.write(audio_bytes)
                print("✓ Test audio saved to deployment_test.wav")
                
                return True
            else:
                print("✗ Speech generation returned empty result")
                return False
                
        except Exception as e:
            print(f"✗ Service test failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Deployment test error: {e}")
        return False


def main():
    """Main deployment workflow"""
    print("Sesame_TTS Modal Deployment Script")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Deploy the service
    if not deploy_service():
        print("\n❌ Deployment failed. Check the error messages above.")
        sys.exit(1)
    
    # Test the deployment
    if test_deployment():
        print("\n🎉 Sesame_TTS service deployed and tested successfully!")
        print("\nNext steps:")
        print("1. Update your client code to use the new service URLs")
        print("2. Test with your own text and audio inputs")
        print("3. Monitor the service performance in the Modal dashboard")
    else:
        print("\n⚠ Deployment succeeded but testing failed.")
        print("The service may still be starting up. Try testing again in a few minutes.")


if __name__ == "__main__":
    main()