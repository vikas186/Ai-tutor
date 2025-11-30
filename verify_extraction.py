"""
Verification script for extracted questions
"""
import json
from typing import List, Dict, Any

def verify_extraction(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify the quality and completeness of extracted questions
    """
    questions = results.get("questions", [])
    
    verification_report = {
        "total_questions": len(questions),
        "expected_count": 25,
        "count_match": len(questions) == 25,
        "issues": [],
        "warnings": [],
        "strengths": [],
        "question_analysis": []
    }
    
    # Check count
    if len(questions) != 25:
        verification_report["issues"].append(
            f"Expected 25 questions but found {len(questions)}"
        )
    else:
        verification_report["strengths"].append("Correct count: 25 questions extracted")
    
    # Check question numbering
    question_numbers = []
    for i, q in enumerate(questions, 1):
        q_text = q.get("question_text", "")
        q_num = None
        
        # Try to extract question number from text
        import re
        # Look for number at start of text
        match = re.match(r'^(\d+)\s*\n', q_text)
        if match:
            q_num = int(match.group(1))
        else:
            # Look for "Question X" pattern
            match = re.search(r'Question\s+(\d+)', q_text, re.IGNORECASE)
            if match:
                q_num = int(match.group(1))
        
        question_numbers.append(q_num)
        
        # Analyze each question
        q_analysis = {
            "index": i,
            "extracted_number": q_num,
            "expected_number": i,
            "number_match": q_num == i if q_num else None,
            "has_sub_questions": bool(re.search(r'\([a-z]\)', q_text, re.IGNORECASE)),
            "text_length": len(q_text),
            "question_type": q.get("question_type"),
            "has_answer": q.get("correct_answer") != "N/A",
            "issues": [],
            "warnings": []
        }
        
        # Check for issues
        if q_num and q_num != i:
            q_analysis["issues"].append(f"Question number mismatch: expected {i}, found {q_num}")
        
        if len(q_text) < 20:
            q_analysis["warnings"].append("Question text is very short")
        
        if len(q_text) > 2000:
            q_analysis["warnings"].append("Question text is very long")
        
        if not q.get("correct_answer") or q.get("correct_answer") == "N/A":
            q_analysis["warnings"].append("No answer provided (expected if not in PDF)")
        
        verification_report["question_analysis"].append(q_analysis)
    
    # Check for missing numbers
    missing_numbers = []
    for i in range(1, 26):
        if i not in question_numbers:
            missing_numbers.append(i)
    
    if missing_numbers:
        verification_report["issues"].append(
            f"Missing question numbers: {missing_numbers}"
        )
    
    # Check for duplicates
    question_texts = [q.get("question_text", "").strip() for q in questions]
    duplicates = []
    seen = set()
    for i, text in enumerate(question_texts):
        normalized = text.lower().strip()
        if normalized in seen:
            duplicates.append(i + 1)
        seen.add(normalized)
    
    if duplicates:
        verification_report["issues"].append(f"Possible duplicate questions: {duplicates}")
    
    # Check structure
    required_fields = ["question_text", "question_type", "correct_answer", "tags", "subject", "topic"]
    missing_fields = []
    for i, q in enumerate(questions, 1):
        for field in required_fields:
            if field not in q or q[field] is None:
                missing_fields.append(f"Question {i} missing {field}")
    
    if missing_fields:
        verification_report["issues"].extend(missing_fields)
    else:
        verification_report["strengths"].append("All questions have required fields")
    
    # Check confidence
    confidences = [q.get("confidence_score", 0) for q in questions]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    verification_report["average_confidence"] = avg_confidence
    
    if avg_confidence >= 0.95:
        verification_report["strengths"].append(f"High average confidence: {avg_confidence:.2%}")
    
    # Overall assessment
    verification_report["overall_status"] = "PASS" if not verification_report["issues"] else "ISSUES_FOUND"
    verification_report["quality_score"] = calculate_quality_score(verification_report)
    
    return verification_report

def calculate_quality_score(report: Dict[str, Any]) -> float:
    """Calculate overall quality score (0-1)"""
    score = 1.0
    
    # Deduct for issues
    score -= len(report["issues"]) * 0.1
    
    # Deduct for warnings
    score -= len(report["warnings"]) * 0.05
    
    # Deduct for count mismatch
    if not report["count_match"]:
        score -= 0.2
    
    # Bonus for strengths
    score += len(report["strengths"]) * 0.05
    
    return max(0.0, min(1.0, score))

if __name__ == "__main__":
    # Example usage - paste your JSON results here
    sample_results = {
        "questions": [],
        "confidence": 0.0
    }
    
    print("Paste your extraction results JSON and run this script to verify quality")
    print("Or import this module and call verify_extraction(results_dict)")

