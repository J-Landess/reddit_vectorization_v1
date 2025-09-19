#!/usr/bin/env python3
"""
Setup script for Reddit Data Analysis Pipeline.
"""
import os
import sys
import subprocess
import logging

def setup_environment():
    """Set up the environment for the Reddit analysis pipeline."""
    print("🚀 Setting up Reddit Data Analysis Pipeline")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ['data', 'outputs', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check for environment variables
    print("\n🔑 Checking environment variables...")
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables:")
        print("export REDDIT_CLIENT_ID='your_client_id_here'")
        print("export REDDIT_CLIENT_SECRET='your_client_secret_here'")
        print("\nOr create a .env file with these values.")
    else:
        print("✅ Environment variables found")
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Set your Reddit API credentials (if not already done)")
    print("2. Run: python main.py")
    print("3. Check the outputs/ directory for results")

if __name__ == "__main__":
    setup_environment()
