"""
Example usage of PerfectExam system
"""
import asyncio
import json
from pathlib import Path
from config import AccuracyConfig
from perfect_ocr import PerfectGeminiOCR
from perfect_parser import PerfectParser
from perfect_generator import PerfectGenerator
from accuracy_validator import AccuracyPipeline
from models import PerfectQuestion, QuestionType, Difficulty


async def example_ocr_extraction():
    """Example: Extract text from an image using OCR"""
    print("=" * 60)
    print("Example 1: OCR Extraction")
    print("=" * 60)
    
    config = AccuracyConfig()
    ocr_engine = PerfectGeminiOCR(config)
    
    # Replace with your image path
    image_path = "sample_exam.png"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    try:
        result = ocr_engine.extract_perfect_text(image_path)
        print(f"\nExtracted Text (confidence: {result.confidence:.2%}):")
        print("-" * 60)
        print(result.extracted_text[:500])  # First 500 chars
        print("-" * 60)
        print(f"\nValidation Checks:")
        for check in result.validation_checks:
            status = "✓" if check.passed else "✗"
            print(f"  {status} {check.check_type}: {check.details}")
        
        if result.requires_human_review:
            print("\n⚠️  Result flagged for human review")
    except Exception as e:
        print(f"Error: {str(e)}")


async def example_question_parsing():
    """Example: Parse questions from extracted text"""
    print("\n" + "=" * 60)
    print("Example 2: Question Parsing")
    print("=" * 60)
    
    config = AccuracyConfig()
    parser = PerfectParser(config)
    
    # Sample extracted text (replace with actual OCR output)
    sample_text = """
    Question 1: What is the capital of France?
    a) London
    b) Berlin
    c) Paris
    d) Madrid
    Correct Answer: c) Paris
    
    Question 2: True or False: Python is a compiled language.
    Correct Answer: False
    """
    
    try:
        result = parser.parse_perfect_questions(
            sample_text,
            subject="Computer Science",
            topic="Programming Languages"
        )
        
        print(f"\nParsed {len(result.questions)} questions")
        print(f"Confidence: {result.confidence:.2%}")
        
        for i, question in enumerate(result.questions, 1):
            print(f"\nQuestion {i}:")
            print(f"  Text: {question.question_text[:100]}...")
            print(f"  Type: {question.question_type.value}")
            print(f"  Difficulty: {question.difficulty.value}")
            print(f"  Confidence: {question.confidence_score:.2%}")
        
        if result.requires_human_review:
            print("\n⚠️  Result flagged for human review")
            
    except Exception as e:
        print(f"Error: {str(e)}")


async def example_question_generation():
    """Example: Generate question variations"""
    print("\n" + "=" * 60)
    print("Example 3: Question Generation")
    print("=" * 60)
    
    config = AccuracyConfig()
    generator = PerfectGenerator(config)
    
    # Create a sample source question
    source_question = PerfectQuestion(
        question_text="What is the time complexity of binary search?",
        question_type=QuestionType.MCQ,
        options=[
            "O(n)",
            "O(log n)",
            "O(n log n)",
            "O(n²)"
        ],
        correct_answer="O(log n)",
        explanation="Binary search eliminates half of the search space in each iteration.",
        difficulty=Difficulty.MEDIUM,
        tags=["algorithms", "complexity", "search"],
        subject="Computer Science",
        topic="Algorithms"
    )
    
    try:
        result = generator.generate_perfect_variations(
            source_question=source_question,
            num_variations=2
        )
        
        print(f"\nGenerated {len(result.generated_questions)} variations")
        print(f"Confidence: {result.confidence:.2%}")
        
        for i, question in enumerate(result.generated_questions, 1):
            print(f"\nVariation {i}:")
            print(f"  Text: {question.question_text}")
            if question.options:
                print(f"  Options: {', '.join(question.options)}")
            print(f"  Answer: {question.correct_answer}")
        
        print(f"\nValidation Checks:")
        for check in result.validation_checks:
            status = "✓" if check.passed else "✗"
            print(f"  {status} {check.check_type}: {check.details}")
        
        if result.requires_human_review:
            print("\n⚠️  Result flagged for human review")
            
    except Exception as e:
        print(f"Error: {str(e)}")


async def example_validation():
    """Example: Validate question accuracy"""
    print("\n" + "=" * 60)
    print("Example 4: Accuracy Validation")
    print("=" * 60)
    
    config = AccuracyConfig()
    validator = AccuracyPipeline(config)
    
    # Create sample questions
    questions = [
        PerfectQuestion(
            question_text="What is 2 + 2?",
            question_type=QuestionType.SHORT_ANSWER,
            correct_answer="4",
            difficulty=Difficulty.EASY,
            tags=["math", "arithmetic"],
            subject="Mathematics",
            topic="Basic Arithmetic"
        ),
        PerfectQuestion(
            question_text="What is the derivative of x²?",
            question_type=QuestionType.SHORT_ANSWER,
            correct_answer="2x",
            difficulty=Difficulty.MEDIUM,
            tags=["math", "calculus"],
            subject="Mathematics",
            topic="Calculus"
        )
    ]
    
    try:
        result = validator.validate_100_percent(questions, "questions")
        
        print(f"\nValidation Result:")
        print(f"  Passed: {result.passed}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Requires Review: {result.requires_human_review}")
        
        print(f"\nPassed Validations: {len(result.passed_validations)}")
        print(f"Failed Validations: {len(result.failed_validations)}")
        
    except Exception as e:
        print(f"Validation Error: {str(e)}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("PerfectExam - Example Usage")
    print("=" * 60)
    print("\nNote: These examples require valid API keys in .env file")
    print("Make sure GEMINI_API_KEY and CLAUDE_API_KEY are set\n")
    
    # Run examples
    asyncio.run(example_ocr_extraction())
    asyncio.run(example_question_parsing())
    asyncio.run(example_question_generation())
    asyncio.run(example_validation())
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

