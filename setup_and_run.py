"""
Setup and run script for PerfectExam
This script helps set up the environment and run the server
"""
import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has API keys"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("⚠️  .env file not found!")
        print("\nCreating .env file template...")
        
        env_content = """# PerfectExam Configuration
# Add your API keys below

# Google Gemini API Key (for OCR)
# Get it from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Anthropic Claude API Key (for parsing and generation)
# Get it from: https://console.anthropic.com/
CLAUDE_API_KEY=your_claude_api_key_here
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Created .env file")
        print("\n⚠️  IMPORTANT: Please edit .env file and add your API keys!")
        print("   - GEMINI_API_KEY: https://makersuite.google.com/app/apikey")
        print("   - CLAUDE_API_KEY: https://console.anthropic.com/")
        return False
    
    # Check if API keys are set
    with open(".env", "r") as f:
        content = f.read()
        
    if "your_gemini_api_key_here" in content or "your_claude_api_key_here" in content:
        print("⚠️  .env file exists but API keys are not configured!")
        print("   Please edit .env file and add your actual API keys.")
        return False
    
    return True

def main():
    """Main setup and run function"""
    print("=" * 60)
    print("PerfectExam - Setup and Run")
    print("=" * 60)
    print()
    
    # Check environment file
    env_ok = check_env_file()
    
    if not env_ok:
        print("\n" + "=" * 60)
        print("Please configure your API keys in .env file first!")
        print("=" * 60)
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nExiting. Please configure .env file and try again.")
            sys.exit(1)
    
    # Try to import and test config
    try:
        from config import AccuracyConfig
        config = AccuracyConfig()
        print("\n✅ Configuration loaded successfully")
    except Exception as e:
        print(f"\n❌ Error loading configuration: {str(e)}")
        print("   Make sure your .env file has valid API keys.")
        sys.exit(1)
    
    # Start the server
    print("\n" + "=" * 60)
    print("Starting PerfectExam Server...")
    print("=" * 60)
    print("\nServer will be available at:")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

