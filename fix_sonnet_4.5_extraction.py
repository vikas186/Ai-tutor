"""
Diagnostic script to check why not all questions are being extracted
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import json

load_dotenv()

def test_sonnet_4_5_extraction():
    """Test Claude Sonnet 4.5 extraction with sample text"""
    
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("❌ CLAUDE_API_KEY not found in .env")
        return
    
    client = Anthropic(api_key=api_key)
    
    # Test with a sample that has multiple questions
    sample_text = """
    Test 1
    
    Section A:
    1. What is 2 + 2? Answer: 4
    2. What is 3 + 3? Answer: 6
    3. What is 4 + 4? Answer: 8
    
    Section B:
    1. What is 5 + 5? Answer: 10
    2. What is 6 + 6? Answer: 12
    3. What is 7 + 7? Answer: 14
    
    Test 2
    
    Section A:
    1. What is 8 + 8? Answer: 16
    2. What is 9 + 9? Answer: 18
    3. What is 10 + 10? Answer: 20
    """
    
    print("=" * 60)
    print("Testing Claude Sonnet 4.5 Extraction")
    print("=" * 60)
    
    # Try different model names
    model_names = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet",
    ]
    
    for model_name in model_names:
        print(f"\nTesting model: {model_name}")
        try:
            prompt = f"""
Extract ALL questions from the following text. There are MULTIPLE tests and sections.

CRITICAL: Extract EVERY question from ALL tests and ALL sections.
Do NOT stop after one question. Extract ALL 9 questions.

Text:
{sample_text}

Return JSON with ALL questions in the "questions" array.
"""
            
            message = client.messages.create(
                model=model_name,
                max_tokens=4096,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = message.content[0].text
            print(f"✅ Model {model_name} works!")
            print(f"   Response length: {len(response)} chars")
            print(f"   Stop reason: {message.stop_reason}")
            
            # Try to extract JSON
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    num_questions = len(data.get("questions", []))
                    print(f"   Extracted questions: {num_questions}")
                    
                    if num_questions == 9:
                        print(f"   ✅ PERFECT! All 9 questions extracted!")
                        return model_name
                    else:
                        print(f"   ⚠️  Only {num_questions}/9 questions extracted")
                else:
                    print(f"   ⚠️  Could not find JSON in response")
            except Exception as e:
                print(f"   ⚠️  JSON parsing error: {str(e)}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    return None

if __name__ == "__main__":
    working_model = test_sonnet_4_5_extraction()
    if working_model:
        print(f"\n✅ Recommended model: {working_model}")
        print(f"Update config.py: claude_model = '{working_model}'")
    else:
        print("\n❌ Could not find working Sonnet 4.5 model")


