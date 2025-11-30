"""
Test script to verify API keys are working
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_key():
    """Test Gemini API key"""
    print("=" * 60)
    print("Testing Gemini API Key")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[X] GEMINI_API_KEY is not set or using placeholder value")
        return False
    
    print(f"[OK] GEMINI_API_KEY found (length: {len(api_key)} characters)")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try different model names (from available models list)
        model_names = [
            "models/gemini-2.0-flash",  # Stable version
            "models/gemini-flash-latest",  # Latest flash
            "models/gemini-2.5-flash",  # Newer version
            "models/gemini-2.0-flash-001",  # Specific version
            "gemini-2.0-flash",  # Without models/ prefix
        ]
        
        print("Testing API connection...")
        model = None
        working_model = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test with a simple call
                response = model.generate_content("Say 'API key is working'")
                working_model = model_name
                break
            except Exception as e:
                continue
        
        if not model or not working_model:
            # Try listing available models
            try:
                models = genai.list_models()
                print(f"   Available models: {[m.name for m in models]}")
            except:
                pass
            raise Exception("Could not find a working Gemini model")
        
        print(f"[OK] Gemini API Key is WORKING!")
        print(f"   Working model: {working_model}")
        print(f"   Response: {response.text[:100]}")
        return True
        
    except Exception as e:
        print(f"[X] Gemini API Key test FAILED")
        print(f"   Error: {str(e)}")
        return False


def test_claude_key():
    """Test Claude API key"""
    print("\n" + "=" * 60)
    print("Testing Claude API Key")
    print("=" * 60)
    
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key or api_key == "your_claude_api_key_here":
        print("[X] CLAUDE_API_KEY is not set or using placeholder value")
        return False
    
    print(f"[OK] CLAUDE_API_KEY found (length: {len(api_key)} characters)")
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        # Make a simple test call
        print("Testing API connection...")
        # Try different model names (Sonnet 4.5 first since user has purchased it)
        model_names = [
            "claude-3-5-sonnet-20241022",  # Claude Sonnet 4.5 (latest)
            "claude-3-5-sonnet-20240620",  # Previous version
            "claude-3-5-sonnet",           # Without date
            "claude-3-5-haiku-20241022",   # Fallback
        ]
        message = None
        last_error = None
        working_model = None
        
        for model_name in model_names:
            try:
                message = client.messages.create(
                    model=model_name,
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": "Say 'API key is working'"}
                    ]
                )
                working_model = model_name
                break
            except Exception as e:
                last_error = e
                continue
        
        if not message:
            raise last_error
        
        if working_model:
            print(f"   Working model: {working_model}")
        
        response_text = message.content[0].text
        print(f"[OK] Claude API Key is WORKING!")
        print(f"   Response: {response_text}")
        return True
        
    except Exception as e:
        print(f"[X] Claude API Key test FAILED")
        print(f"   Error: {str(e)}")
        return False


def check_env_file():
    """Check if .env file exists and show status"""
    print("=" * 60)
    print("Checking .env File")
    print("=" * 60)
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print("[X] .env file NOT FOUND!")
        print("\nPlease create a .env file with:")
        print("  GEMINI_API_KEY=your_key_here")
        print("  CLAUDE_API_KEY=your_key_here")
        return False
    
    print("[OK] .env file exists")
    
    # Check file content (without exposing keys)
    with open(".env", "r") as f:
        lines = f.readlines()
        gemini_found = False
        claude_found = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                gemini_found = True
                value = line.split("=", 1)[1]
                if value and value != "your_gemini_api_key_here":
                    print(f"[OK] GEMINI_API_KEY is set (not empty)")
                else:
                    print(f"[!] GEMINI_API_KEY is set but using placeholder")
            elif line.startswith("CLAUDE_API_KEY="):
                claude_found = True
                value = line.split("=", 1)[1]
                if value and value != "your_claude_api_key_here":
                    print(f"[OK] CLAUDE_API_KEY is set (not empty)")
                else:
                    print(f"[!] CLAUDE_API_KEY is set but using placeholder")
        
        if not gemini_found:
            print("[X] GEMINI_API_KEY not found in .env")
        if not claude_found:
            print("[X] CLAUDE_API_KEY not found in .env")
    
    return True


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("API Keys Verification Test")
    print("=" * 60 + "\n")
    
    # Check .env file
    env_ok = check_env_file()
    
    if not env_ok:
        print("\n[X] Please fix .env file issues first!")
        return
    
    print("\n")
    
    # Test keys
    gemini_ok = test_gemini_key()
    claude_ok = test_claude_key()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if gemini_ok and claude_ok:
        print("[OK] ALL API KEYS ARE WORKING!")
        print("\nYou can now run the server:")
        print("  python run.py")
    else:
        print("[X] SOME API KEYS ARE NOT WORKING")
        if not gemini_ok:
            print("   - Gemini API key needs to be fixed")
        if not claude_ok:
            print("   - Claude API key needs to be fixed")
        print("\nPlease check:")
        print("  1. API keys are correct in .env file")
        print("  2. API keys have sufficient credits/quota")
        print("  3. Internet connection is working")
        print("  4. API services are accessible from your location")


if __name__ == "__main__":
    main()

