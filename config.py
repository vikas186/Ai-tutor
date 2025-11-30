"""
Configuration for PerfectExam - 100% Accuracy Generator
"""
from pydantic_settings import BaseSettings
from typing import Literal


class AccuracyConfig(BaseSettings):
    """Configuration for maximum accuracy"""
    
    # Gemini OCR Configuration
    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-2.0-flash"  # Stable version (available models)
    gemini_temperature: float = 0.1
    gemini_max_output_tokens: int = 8192
    
    # Claude LLM Configuration
    claude_api_key: str = ""
    claude_model: str = "claude-3-5-haiku-20241022"  # Using Claude 3.5 Haiku (Sonnet requires upgraded API access)
    claude_temperature: float = 0.1
    claude_max_tokens: int = 8192  # Max output tokens for Claude
    
    # Validation Thresholds
    min_confidence: float = 0.98
    max_retries: int = 5
    human_review_threshold: float = 0.95
    
    # Quality Gates
    ocr_confidence: float = 0.98
    parsing_completeness: float = 1.0
    generation_accuracy: float = 1.0
    structural_validity: float = 1.0
    
    # Image Processing
    image_enhancement_enabled: bool = True
    ocr_passes: int = 3
    
    # Multi-LLM Ensemble Settings
    enable_multi_llm: bool = True
    consensus_threshold: float = 0.8  # Agreement threshold for consensus
    use_ensemble_for_ocr: bool = True
    use_ensemble_for_parsing: bool = True
    use_ensemble_for_generation: bool = True
    conflict_resolution_strategy: str = "hybrid"  # voting, longest, claude_priority, gemini_priority, hybrid
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Perfection prompts for LLM interactions
PERFECTION_PROMPTS = {
    "parsing": '''
You are an absolute perfectionist in extracting exam questions. 
You MUST achieve 100% accuracy with zero errors.

CRITICAL RULES:
1. Extract EVERY question completely - DO NOT SKIP ANY QUESTIONS
2. Preserve EXACT wording and formatting
3. Identify ALL options for MCQs
4. Extract CORRECT answers precisely - if answer is in document, extract it; if it's a math problem, calculate it; if not available, use "N/A"
5. ALWAYS provide tags - at minimum use [subject, topic, question_type] - tags are REQUIRED
6. Flag ANY ambiguities for human review
7. Extract ALL questions in the document, not just the first few
8. If the document has many questions, extract ALL of them
9. Do not stop after extracting one or a few questions - continue until ALL are extracted
10. If there are multiple tests/sections, extract questions from ALL tests/sections
11. Count ALL questions first, then extract each one completely

CRITICAL: The document may contain 50, 100, 200+ questions. You MUST extract ALL of them.
DO NOT stop after 1 question. DO NOT stop after 10 questions. Extract EVERY SINGLE QUESTION.

For arithmetic tests with multiple sections (A, B, C), extract ALL questions from ALL sections.
For documents with multiple pages/tests, extract ALL questions from ALL pages/tests.

Return ONLY perfectly structured JSON. If you cannot achieve 100% confidence, 
return a validation error instead of partial data.

Output format (JSON):
{
    "questions": [
        {
            "question_text": "exact question text",
            "question_type": "mcq|true_false|fill_blank|short_answer|long_answer",
            "options": ["option1", "option2", ...],  # For MCQs only
            "correct_answer": "exact correct answer if available in document, otherwise calculate it or use 'N/A'",
            "explanation": "explanation if available",
            "difficulty": "easy|medium|hard",
            "subject": "subject name",
            "topic": "topic name",
            "tags": ["tag1", "tag2", "subject", "topic"]  # REQUIRED: At least one tag
        }
    ],
    "confidence": 0.0-1.0,
    "validation_errors": []  # Empty if 100% confident
}

CRITICAL: Every question MUST have:
- correct_answer: Extract from document if available, calculate if it's a math problem, or use "N/A" if not available
- tags: ALWAYS provide at least one tag (can use subject, topic, question_type, difficulty)
''',
    
    "generation": '''
You generate exam questions with 100% conceptual accuracy.

NON-NEGOTIABLE REQUIREMENTS:
1. Maintain EXACT same concept as source
2. Use PRECISE same difficulty level
3. Follow IDENTICAL question pattern
4. Ensure 100% factual correctness
5. Zero hallucinations or creative deviations

Every generated question must be verifiably perfect.

Generate variations that test the SAME concept but with different wording or examples.
Maintain the same structure, difficulty, and educational value.
'''
}

