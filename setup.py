#!/usr/bin/env python3
"""
Setup script for Automated Essay Scoring System
This script helps users set up the environment and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_spacy_model():
    """Download SpaCy English model."""
    print("\n🌍 Downloading SpaCy English model...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ SpaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not download SpaCy model: {e}")
        print("   Some text analysis features may not work properly")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["data", "models", "logs", "static/css", "static/js", "templates"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def create_env_file():
    """Create .env file from template."""
    print("\n⚙️  Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  Warning: Could not create .env file")

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import transformers
        import fastapi
        import pandas
        import sklearn
        import numpy
        
        print("✅ All core packages imported successfully")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (CPU-only mode)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_demo():
    """Run a quick demo to verify everything works."""
    print("\n🚀 Running quick demo...")
    
    try:
        result = subprocess.run([sys.executable, "0198.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Demo completed successfully")
            return True
        else:
            print(f"❌ Demo failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Demo timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def main():
    """Main setup function."""
    print("🧠 Automated Essay Scoring System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Download SpaCy model
    download_spacy_model()
    
    # Create directories
    create_directories()
    
    # Create environment file
    create_env_file()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed")
        sys.exit(1)
    
    # Run demo
    run_demo()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("   1. Run the web interface: python 0198.py --web")
    print("   2. Train the model: python 0198.py --train")
    print("   3. Run evaluation: python 0198.py --evaluate")
    print("   4. Read the README.md for more information")
    print("\n🌐 Web interface will be available at: http://localhost:8000")

if __name__ == "__main__":
    main()
