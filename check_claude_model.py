"""
Quick script to check available Claude models
"""
from anthropic import Anthropic
from config import AccuracyConfig
import os

config = AccuracyConfig()

# Try different model names
models_to_try = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620", 
    "claude-3-5-sonnet",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022"
]

client = Anthropic(api_key=config.claude_api_key)

print("Testing Claude models...")
print("=" * 60)

for model in models_to_try:
    try:
        print(f"\nTesting: {model}")
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ SUCCESS: {model} is available!")
        print(f"   Response: {response.content[0].text[:50]}")
        break
    except Exception as e:
        error_msg = str(e)
        if "not_found" in error_msg.lower():
            print(f"❌ Not found: {model}")
        else:
            print(f"⚠️  Error with {model}: {error_msg[:100]}")

print("\n" + "=" * 60)
print("Model check complete!")

