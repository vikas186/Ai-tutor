"""
Quick start script for PerfectExam
"""
import uvicorn
import sys
from pathlib import Path

# Check if .env exists
if not Path(".env").exists():
    print("⚠️  Warning: .env file not found!")
    print("Please create .env file with your API keys:")
    print("  GEMINI_API_KEY=your_key_here")
    print("  CLAUDE_API_KEY=your_key_here")
    print("\nYou can copy .env.example to .env and fill in your keys.")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("PerfectExam - 100% Accuracy Generator")
    print("=" * 60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("API documentation: http://0.0.0.0:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

