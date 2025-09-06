#!/usr/bin/env python3
"""
Verify that the Modal code syntax is correct without actually deploying
"""

import ast
import sys

def verify_python_syntax(file_path):
    """Verify Python syntax of a file"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        print(f"✅ {file_path}: Python syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax error - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error - {e}")
        return False

def main():
    print("🔍 Verifying Modal code syntax...")
    
    files_to_check = [
        "unmute/tts/orpheus_modal.py",
        "unmute/modal_app.py",
        "deploy_orpheus_modal.py"
    ]
    
    all_valid = True
    for file_path in files_to_check:
        if not verify_python_syntax(file_path):
            all_valid = False
    
    if all_valid:
        print("\n🎉 All files have valid Python syntax!")
        print("The Modal code is ready for deployment.")
        print("\nTo deploy, run from your local environment:")
        print("  modal deploy unmute/tts/orpheus_modal.py::llama_app")
        print("  modal deploy unmute/tts/orpheus_modal.py::app")
    else:
        print("\n❌ Some files have syntax errors. Please fix them before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()