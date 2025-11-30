"""
Complete example script to extract all questions from a document
"""
import requests
import json
from pathlib import Path
import sys

def extract_all_questions(pdf_path, subject="Mathematics", topic="General"):
    """
    Extract all questions from a PDF/image file
    
    Args:
        pdf_path: Path to PDF or image file
        subject: Subject name
        topic: Topic name
    """
    print("=" * 60)
    print("PerfectExam - Question Extraction")
    print("=" * 60)
    
    # Step 1: Check server is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("❌ Server is not running. Start it with: python run.py")
            return None
        print("✅ Server is running")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Start it with: python run.py")
        print("   Make sure the server is running on http://localhost:8000")
        return None
    
    # Step 2: Check file exists
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    # Step 3: Upload and extract
    print(f"\n📄 Processing: {pdf_path}")
    print(f"   Subject: {subject}")
    print(f"   Topic: {topic}")
    
    url = "http://localhost:8000/extract-perfect-questions"
    
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": f}
            data = {"subject": subject, "topic": topic}
            
            print("\n⏳ Extracting questions... (this may take 1-3 minutes)")
            print("   - OCR extraction...")
            print("   - Question parsing...")
            print("   - Validation...")
            
            response = requests.post(url, files=files, data=data, timeout=300)  # 5 min timeout
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"❌ Error uploading file: {str(e)}")
        return None
    
    # Step 4: Process results
    if response.status_code == 200:
        result = response.json()
        num_questions = len(result["questions"])
        confidence = result["confidence"]
        
        print(f"\n✅ SUCCESS!")
        print(f"   Extracted: {num_questions} questions")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Review Flagged: {result['requires_human_review']}")
        
        if result.get("parsing_errors"):
            print(f"   Parsing Errors: {len(result['parsing_errors'])}")
        
        # Step 5: Save results
        output_file = Path(pdf_path).stem + "_questions.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_file}")
        
        # Step 6: Show summary
        print("\n" + "=" * 60)
        print("QUESTION SUMMARY")
        print("=" * 60)
        
        by_type = {}
        by_difficulty = {}
        
        for q in result["questions"]:
            q_type = q["question_type"]
            diff = q["difficulty"]
            
            by_type[q_type] = by_type.get(q_type, 0) + 1
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        
        print("\nBy Type:")
        for q_type, count in sorted(by_type.items()):
            print(f"   {q_type}: {count}")
        
        print("\nBy Difficulty:")
        for diff, count in sorted(by_difficulty.items()):
            print(f"   {diff}: {count}")
        
        # Show first 5 questions
        print("\n" + "=" * 60)
        print("SAMPLE QUESTIONS (First 5)")
        print("=" * 60)
        
        for i, q in enumerate(result["questions"][:5], 1):
            print(f"\n{i}. {q['question_text'][:150]}...")
            print(f"   Type: {q['question_type']} | Difficulty: {q['difficulty']}")
            print(f"   Answer: {q['correct_answer']}")
            if q.get('options'):
                print(f"   Options: {', '.join(q['options'])}")
        
        if num_questions > 5:
            print(f"\n... and {num_questions - 5} more questions")
        
        return result
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        try:
            error_detail = response.json()
            print(f"   Detail: {error_detail.get('detail', 'Unknown error')}")
        except:
            print(f"   Response: {response.text[:200]}")
        return None


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        subject = sys.argv[2] if len(sys.argv) > 2 else "Mathematics"
        topic = sys.argv[3] if len(sys.argv) > 3 else "General"
    else:
        print("Usage: python extract_questions_example.py <file_path> [subject] [topic]")
        print("\nExample:")
        print("  python extract_questions_example.py arithmetic_test.pdf Mathematics 'Mental Arithmetic'")
        sys.exit(1)
    
    result = extract_all_questions(pdf_path, subject, topic)
    
    if result:
        print("\n" + "=" * 60)
        print("✅ Extraction Complete!")
        print("=" * 60)
        print(f"\nAll {len(result['questions'])} questions saved to JSON file.")
        print("You can now use this data in your application!")
    else:
        print("\n" + "=" * 60)
        print("❌ Extraction Failed")
        print("=" * 60)
        sys.exit(1)

