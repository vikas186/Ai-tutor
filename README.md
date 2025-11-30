# PerfectExam - 100% Accuracy Generator

A Python-based exam question generation system that guarantees maximum accuracy using Google Gemini 2.0 Flash for OCR and Claude 3.5 Sonnet for parsing and generation.

## 🎯 Core Philosophy

**100% Accuracy or Human Review** - This system is designed with the mindset that 99% accuracy is unacceptable. Every component is built, implemented, and tested for perfect accuracy.

## ✨ Features

- **Multi-Pass OCR**: Triple-pass extraction using Google Gemini 2.0 Flash with different strategies
- **Advanced Image Enhancement**: Pre-processing pipeline with contrast enhancement, noise reduction, deskewing, and resolution optimization
- **Multi-Step Parsing**: 4-step verification process using Claude 3.5 Sonnet
- **Triple-Generation**: Generate, verify, and refine question variations
- **Comprehensive Validation**: Multi-layer validation system (OCR, Structure, Content, Consistency)
- **Strict Data Models**: Pydantic models with minimum 95% confidence requirement
- **Quality Gates**: Configurable thresholds for each processing stage

## 🏗️ Architecture

```
INPUT PROCESSING:
High-Quality Document → Enhanced Pre-processing → Gemini OCR → 
Accuracy Validation → Claude Parsing → Quality Gate → 
Claude Generation → Final Validation → Output
```

## 📋 Requirements

- Python 3.11+
- Google Gemini API Key
- Anthropic Claude API Key

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-me
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. Run the application:
```bash
python main.py
```

Or using uvicorn:
```bash
uvicorn main:app --reload
```

## 📡 API Endpoints

### 1. Extract Perfect Questions
**POST** `/extract-perfect-questions`

Extract questions from a document (PDF/Image) with 100% accuracy guarantee.

**Parameters:**
- `file`: Upload file (PDF or image)
- `subject`: Subject name (default: "General")
- `topic`: Topic name (default: "General")

**Response:** `ParsingResult` with extracted questions and validation checks

### 2. Generate Perfect Variations
**POST** `/generate-perfect-variations`

Generate question variations with 100% conceptual accuracy.

**Parameters:**
- `source_question`: Source question as JSON
- `num_variations`: Number of variations to generate (default: 3)

**Response:** `GenerationResult` with generated questions

### 3. Validate Accuracy
**POST** `/validate-accuracy`

Validate accuracy of question data.

**Parameters:**
- `question_data`: Question or questions as JSON

**Response:** `ValidationResult` with validation report

### 4. Extract OCR Only
**POST** `/extract-ocr-only`

Extract text using OCR only (useful for debugging).

**Parameters:**
- `file`: Upload file

**Response:** `OCRResult` with extracted text

### 5. Health Check
**GET** `/health`

Check system health and component status.

### 6. Configuration
**GET** `/config`

Get current configuration (without sensitive keys).

## 🔧 Configuration

Configuration is managed through environment variables and `config.py`:

```python
# Quality Gates
ocr_confidence: 0.98  # 98% minimum confidence
parsing_completeness: 1.0  # 100% complete parsing
generation_accuracy: 1.0  # 100% concept accuracy
structural_validity: 1.0  # 100% valid structure

# Validation Thresholds
min_confidence: 0.98  # Minimum overall confidence
human_review_threshold: 0.95  # Flag for human review below this
max_retries: 5  # Maximum retry attempts
```

## 📊 Data Models

### PerfectQuestion
```python
{
    "id": "UUID",
    "question_text": "string (min 10 chars)",
    "question_type": "mcq|true_false|fill_blank|short_answer|long_answer",
    "options": ["option1", "option2", ...],  # For MCQs
    "correct_answer": "string",
    "explanation": "string (optional)",
    "difficulty": "easy|medium|hard",
    "tags": ["tag1", "tag2"],
    "subject": "string",
    "topic": "string",
    "confidence_score": 0.95-1.0,  # Minimum 95%
    "validation_checks": [...]
}
```

## 🔍 Validation System

The system includes multiple validation layers:

1. **OCR Validator**: Checks extraction quality, text completeness, question indicators
2. **Structure Validator**: Validates required fields, MCQ options, confidence scores
3. **Content Validator**: Checks question text quality, answer presence, tags
4. **Consistency Validator**: Validates consistency across multiple questions

## 🛡️ Error Handling

- **No Silent Failures**: All errors are explicitly raised
- **Automatic Retry**: Retry with enhanced parameters on failure
- **Human Review Flags**: Borderline cases are flagged for human review
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 📈 Accuracy Monitoring

The system tracks:
- Confidence scores at each stage
- Validation check results
- Human review flags
- Error rates (target: 0%)

## 🧪 Testing

To test the system:

1. **OCR Testing**: Use `/extract-ocr-only` with sample documents
2. **Full Pipeline**: Use `/extract-perfect-questions` with exam documents
3. **Generation**: Use `/generate-perfect-variations` with sample questions
4. **Validation**: Use `/validate-accuracy` to check question quality

## ⚠️ Important Notes

1. **API Keys**: Ensure you have valid API keys for both Gemini and Claude
2. **Resource Requirements**: System requires sufficient memory for large document processing
3. **Internet Connection**: Reliable high-speed internet required for API calls
4. **Accuracy Guarantee**: System flags results for human review if confidence < 95%

## 🔐 Security

- API keys are stored in `.env` file (not committed to version control)
- Sensitive information is not exposed in configuration endpoints
- File uploads are processed in temporary files and cleaned up

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Support

[Add support contact information here]

---

**Built with the understanding that any accuracy less than 100% is a critical failure.**

