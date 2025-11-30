"""
Check which Claude models are available with the API key
"""
import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

api_key = os.getenv("CLAUDE_API_KEY")
if not api_key:
    print("❌ CLAUDE_API_KEY not found")
    sys.exit(1)

client = Anthropic(api_key=api_key)

# List of possible model names
models_to_test = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620", 
    "claude-3-5-sonnet",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-20240620",
    "claude-3-5-haiku",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

print("=" * 60)
print("Testing Claude Models")
print("=" * 60)

working_models = []

for model_name in models_to_test:
    try:
        print(f"\nTesting: {model_name}...", end=" ")
        message = client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hi"}]
        )
        print("✅ WORKS!")
        working_models.append(model_name)
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not_found" in error_msg.lower():
            print("❌ Not found")
        else:
            print(f"❌ Error: {error_msg[:50]}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if working_models:
    print(f"\n✅ Working models ({len(working_models)}):")
    for model in working_models:
        print(f"   - {model}")
    
    # Recommend Sonnet if available
    sonnet_models = [m for m in working_models if "sonnet" in m.lower()]
    if sonnet_models:
        recommended = sonnet_models[0]
        print(f"\n✅ Recommended (Sonnet): {recommended}")
        print(f"\nUpdate config.py:")
        print(f'   claude_model = "{recommended}"')
    else:
        print(f"\n⚠️  No Sonnet models available. Using: {working_models[0]}")
        print(f"\nUpdate config.py:")
        print(f'   claude_model = "{working_models[0]}"')
else:
    print("\n❌ No working models found!")
    print("Please check your API key and account access.")


