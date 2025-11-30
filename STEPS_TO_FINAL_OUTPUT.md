# Steps to Get Final Output - Complete Guide

## Quick Steps Overview

1. **Start the Server**
2. **Upload Your Document** (PDF or Image)
3. **Get Extracted Questions** (JSON response)
4. **Review Results**

---

## Detailed Step-by-Step Instructions

### Step 1: Start the Server

**Option A: Using run.py (Recommended)**
```bash
python run.py
```

**Option B: Using uvicorn directly**
```bash
uvicorn main:app --reload
```

**Option C: Using setup script**
```bash
python setup_and_run.py
```

**Expected Output:**
```
============================================================
PerfectExam - 100% Accuracy Generator
============================================================

Starting server on http://0.0.0.0:8000
API documentation: http://0.0.0.0:8000/docs

Press Ctrl+C to stop
```

**Verify Server is Running:**
- Open browser: http://localhost:8000/health
- Should show: `{"status":"healthy",...}`

---

### Step 2: Access API Documentation

**Open in Browser:**
```
http://localhost:8000/docs
```

This opens the **Swagger UI** - an interactive interface to test the API.

---

### Step 3: Extract Questions from Document

#### Method 1: Using Swagger UI (Easiest)

1. **In Swagger UI, find the endpoint:**
   - Look for `POST /extract-perfect-questions`
   - Click on it to expand

2. **Click "Try it out" button**

3. **Fill in the form:**
   - **file:** Click "Choose File" and select your PDF/image
   - **subject:** Enter subject name (e.g., "Mathematics", "Arithmetic")
   - **topic:** Enter topic name (e.g., "Mental Arithmetic", "Basic Math")

4. **Click "Execute"**

5. **View Results:**
   - Scroll down to see the response
   - You'll see all extracted questions in JSON format

#### Method 2: Using Python Script

Create a file `extract_questions.py`:

```python
import requests

# API endpoint
url = "http://localhost:8000/extract-perfect-questions"

# Your file
file_path = "your_document.pdf"  # or .png, .jpg, etc.

# Prepare the request
files = {"file": open(file_path, "rb")}
data = {
    "subject": "Mathematics",
    "topic": "Mental Arithmetic"
}

# Send request
response = requests.post(url, files=files, data=data)

# Check response
if response.status_code == 200:
    result = response.json()
    
    print(f"✅ Success! Extracted {len(result['questions'])} questions")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Requires Review: {result['requires_human_review']}")
    
    # Save to file
    import json
    with open("extracted_questions.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Questions saved to extracted_questions.json")
    
    # Print first few questions
    print("\nFirst 3 questions:")
    for i, q in enumerate(result['questions'][:3], 1):
        print(f"\n{i}. {q['question_text'][:100]}...")
        print(f"   Type: {q['question_type']}")
        print(f"   Answer: {q['correct_answer']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.json())
```

**Run it:**
```bash
python extract_questions.py
```

#### Method 3: Using curl (Command Line)

```bash
curl -X POST "http://localhost:8000/extract-perfect-questions" \
  -F "file=@your_document.pdf" \
  -F "subject=Mathematics" \
  -F "topic=Mental Arithmetic" \
  -o extracted_questions.json
```

---

### Step 4: Understand the Output

The response will be a JSON object like this:

```json
{
  "questions": [
    {
      "id": "uuid-here",
      "question_text": "What is 2 + 2?",
      "question_type": "short_answer",
      "options": null,
      "correct_answer": "4",
      "explanation": null,
      "difficulty": "easy",
      "tags": ["arithmetic", "addition"],
      "subject": "Mathematics",
      "topic": "Mental Arithmetic",
      "confidence_score": 0.98,
      "validation_checks": [...]
    },
    // ... more questions
  ],
  "confidence": 0.9743,
  "validation_checks": [...],
  "parsing_errors": [],
  "requires_human_review": false
}
```

**Key Fields:**
- **questions:** Array of all extracted questions
- **confidence:** Overall confidence score (0.0-1.0)
- **requires_human_review:** `true` if flagged for review
- **parsing_errors:** Any errors encountered

---

### Step 5: Process the Results

#### Save to JSON File

```python
import json

# After getting response
with open("all_questions.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
```

#### Export to CSV

```python
import csv
import json

# Load questions
with open("extracted_questions.json", "r") as f:
    data = json.load(f)

# Write to CSV
with open("questions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question", "Type", "Answer", "Difficulty", "Subject", "Topic"])
    
    for q in data["questions"]:
        writer.writerow([
            q["question_text"],
            q["question_type"],
            q["correct_answer"],
            q["difficulty"],
            q["subject"],
            q["topic"]
        ])
```

#### Filter Questions

```python
# Filter by difficulty
easy_questions = [q for q in result["questions"] if q["difficulty"] == "easy"]

# Filter by type
mcq_questions = [q for q in result["questions"] if q["question_type"] == "mcq"]

# Filter by confidence
high_confidence = [q for q in result["questions"] if q["confidence_score"] >= 0.98]
```

---

## Complete Workflow Example

### Full Python Script

```python
import requests
import json
from pathlib import Path

def extract_all_questions(pdf_path, subject="Mathematics", topic="General"):
    """
    Extract all questions from a PDF/image file
    """
    print("=" * 60)
    print("PerfectExam - Question Extraction")
    print("=" * 60)
    
    # Step 1: Check server is running
    try:
        health = requests.get("http://localhost:8000/health")
        if health.status_code != 200:
            print("❌ Server is not running. Start it with: python run.py")
            return
        print("✅ Server is running")
    except:
        print("❌ Cannot connect to server. Start it with: python run.py")
        return
    
    # Step 2: Upload and extract
    print(f"\n📄 Processing: {pdf_path}")
    print(f"   Subject: {subject}")
    print(f"   Topic: {topic}")
    
    url = "http://localhost:8000/extract-perfect-questions"
    
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        data = {"subject": subject, "topic": topic}
        
        print("\n⏳ Extracting questions... (this may take a minute)")
        response = requests.post(url, files=files, data=data)
    
    # Step 3: Process results
    if response.status_code == 200:
        result = response.json()
        num_questions = len(result["questions"])
        confidence = result["confidence"]
        
        print(f"\n✅ SUCCESS!")
        print(f"   Extracted: {num_questions} questions")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Review Flagged: {result['requires_human_review']}")
        
        # Step 4: Save results
        output_file = Path(pdf_path).stem + "_questions.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {output_file}")
        
        # Step 5: Show summary
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
        for q_type, count in by_type.items():
            print(f"   {q_type}: {count}")
        
        print("\nBy Difficulty:")
        for diff, count in by_difficulty.items():
            print(f"   {diff}: {count}")
        
        # Show first 3 questions
        print("\n" + "=" * 60)
        print("SAMPLE QUESTIONS (First 3)")
        print("=" * 60)
        
        for i, q in enumerate(result["questions"][:3], 1):
            print(f"\n{i}. {q['question_text']}")
            print(f"   Type: {q['question_type']} | Difficulty: {q['difficulty']}")
            print(f"   Answer: {q['correct_answer']}")
        
        return result
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        print(response.json())
        return None

if __name__ == "__main__":
    # Example usage
    result = extract_all_questions(
        pdf_path="your_arithmetic_test.pdf",
        subject="Mathematics",
        topic="Mental Arithmetic"
    )
```

---

## Quick Reference

### 1. Start Server
```bash
python run.py
```

### 2. Open API Docs
```
http://localhost:8000/docs
```

### 3. Upload File
- Use Swagger UI or Python script
- Upload PDF/image
- Set subject and topic

### 4. Get Results
- JSON response with all questions
- Save to file for further processing

### 5. Process Results
- Filter by type/difficulty
- Export to CSV
- Use in your application

---

## Troubleshooting

### Server Not Starting?
- Check if port 8000 is in use
- Verify API keys in `.env` file
- Check Python version (3.11+)

### Only 1 Question Extracted?
- Check server logs for warnings
- Verify OCR extracted full text
- Check if document has clear question separators

### Slow Processing?
- Large documents take time
- Multiple OCR passes for accuracy
- Check server logs for progress

### Errors?
- Check server logs for details
- Verify file format is supported
- Check API keys are valid

---

## Expected Processing Time

- **Small document (1-2 pages):** 10-30 seconds
- **Medium document (5-10 pages):** 30-60 seconds
- **Large document (20+ pages):** 1-3 minutes

The system does multiple passes for accuracy, so it takes time but ensures quality.

---

## Next Steps After Extraction

1. **Review Questions:** Check `requires_human_review` flag
2. **Validate:** Use `/validate-accuracy` endpoint
3. **Generate Variations:** Use `/generate-perfect-variations` endpoint
4. **Export:** Save to CSV/JSON for your use case

---

## Complete Example Script

Save this as `extract_questions.py`:

```python
import requests
import json

url = "http://localhost:8000/extract-perfect-questions"
files = {"file": open("your_document.pdf", "rb")}
data = {"subject": "Mathematics", "topic": "Mental Arithmetic"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Extracted {len(result['questions'])} questions")
with open("output.json", "w") as f:
    json.dump(result, f, indent=2)
```

Run: `python extract_questions.py`

---

That's it! Follow these steps to get all your questions extracted! 🎉

