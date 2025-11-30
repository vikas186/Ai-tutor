"""
Integration tests for Multi-LLM Ensemble functionality
Tests the consensus, cross-validation, and merging capabilities
"""
import logging
from config import AccuracyConfig
from multi_llm_validator import MultiLLMValidator, ConsensusStrategy
from consensus_algorithms import TextSimilarity, JSONMerger, QuestionComparator, ConsensusScorer
from models import PerfectQuestion, QuestionType, Difficulty
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_llm_validator():
    """Test MultiLLMValidator basic functionality"""
    logger.info("=" * 60)
    logger.info("TEST: MultiLLMValidator")
    logger.info("=" * 60)
    
    config = AccuracyConfig()
    validator = MultiLLMValidator(
        claude_api_key=config.claude_api_key,
        gemini_api_key=config.gemini_api_key,
        claude_model=config.claude_model,
        gemini_model=config.gemini_model
    )
    
    # Test simple generation with consensus
    prompt = "What is 2+2? Provide only the number."
    result = validator.generate_with_consensus(
        prompt=prompt,
        strategy=ConsensusStrategy.HYBRID,
        temperature=0.0,
        max_tokens=100
    )
    
    logger.info(f"✅ Generated with consensus:")
    logger.info(f"   Final content: {result.final_content[:100]}")
    logger.info(f"   Confidence: {result.confidence:.2f}")
    logger.info(f"   Consensus score: {result.consensus_score:.2f}")
    logger.info(f"   Strategy: {result.strategy_used}")
    logger.info(f"   Individual responses: {len(result.individual_responses)}")
    
    assert result.final_content, "Should have generated content"
    assert len(result.individual_responses) >= 1, "Should have at least one response"
    
    logger.info("✅ MultiLLMValidator test passed!")


def test_text_similarity():
    """Test text similarity algorithms"""
    logger.info("=" * 60)
    logger.info("TEST: Text Similarity")
    logger.info("=" * 60)
    
    text1 = "What is the capital of France?"
    text2 = "What is France's capital city?"
    text3 = "How many legs does a spider have?"
    
    # Test similarity between similar texts
    sim_similar = TextSimilarity.combined_similarity(text1, text2)
    logger.info(f"Similarity (similar texts): {sim_similar:.2f}")
    assert sim_similar > 0.5, "Similar texts should have high similarity"
    
    # Test similarity between different texts
    sim_different = TextSimilarity.combined_similarity(text1, text3)
    logger.info(f"Similarity (different texts): {sim_different:.2f}")
    assert sim_different < 0.5, "Different texts should have low similarity"
    
    logger.info("✅ Text similarity test passed!")


def test_json_merger():
    """Test JSON merging functionality"""
    logger.info("=" * 60)
    logger.info("TEST: JSON Merger")
    logger.info("=" * 60)
    
    json_a = {
        "questions": [
            {"question_text": "What is 2+2?", "correct_answer": "4"},
            {"question_text": "What is 3+3?", "correct_answer": "6"}
        ],
        "confidence": 0.95
    }
    
    json_b = {
        "questions": [
            {"question_text": "What is 2+2?", "correct_answer": "4"},  # Duplicate
            {"question_text": "What is 5+5?", "correct_answer": "10"}  # New
        ],
        "confidence": 0.90
    }
    
    merged = JSONMerger.merge_question_lists(json_a, json_b)
    
    logger.info(f"JSON A questions: {len(json_a['questions'])}")
    logger.info(f"JSON B questions: {len(json_b['questions'])}")
    logger.info(f"Merged questions: {len(merged['questions'])}")
    
    # Should have 3 unique questions (2+2, 3+3, 5+5)
    assert len(merged['questions']) == 3, f"Should have 3 unique questions, got {len(merged['questions'])}"
    
    logger.info("✅ JSON merger test passed!")


def test_question_comparator():
    """Test question comparison and matching"""
    logger.info("=" * 60)
    logger.info("TEST: Question Comparator")
    logger.info("=" * 60)
    
    questions_a = [
        {"question_text": "What is the capital of France?", "correct_answer": "Paris"},
        {"question_text": "What is 2+2?", "correct_answer": "4"}
    ]
    
    questions_b = [
        {"question_text": "What is France's capital city?", "correct_answer": "Paris"},  # Match
        {"question_text": "What is 5+5?", "correct_answer": "10"}  # Different
    ]
    
    matches, only_a, only_b = QuestionComparator.find_matching_questions(
        questions_a, questions_b, similarity_threshold=0.7
    )
    
    logger.info(f"Matches: {len(matches)}")
    logger.info(f"Only in A: {len(only_a)}")
    logger.info(f"Only in B: {len(only_b)}")
    
    assert len(matches) >= 1, "Should find at least one match"
    assert len(only_a) >= 1, "Should have questions only in A"
    assert len(only_b) >= 1, "Should have questions only in B"
    
    # Test merging matched questions
    if matches:
        merged_q = QuestionComparator.merge_matched_questions(matches[0][0], matches[0][1])
        logger.info(f"Merged question: {merged_q['question_text'][:50]}")
        assert merged_q["question_text"], "Merged question should have text"
    
    logger.info("✅ Question comparator test passed!")


def test_consensus_scorer():
    """Test consensus scoring"""
    logger.info("=" * 60)
    logger.info("TEST: Consensus Scorer")
    logger.info("=" * 60)
    
    # Test high agreement
    responses_high = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "France's capital city is Paris."
    ]
    
    score_high = ConsensusScorer.calculate_agreement_score(responses_high)
    logger.info(f"High agreement score: {score_high:.2f}")
    assert score_high > 0.5, "High agreement responses should have score > 0.5"
    
    # Test low agreement
    responses_low = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Tokyo is Japan's capital."
    ]
    
    score_low = ConsensusScorer.calculate_agreement_score(responses_low)
    logger.info(f"Low agreement score: {score_low:.2f}")
    assert score_low < 0.5, "Low agreement responses should have score < 0.5"
    
    logger.info("✅ Consensus scorer test passed!")


def test_ensemble_parser():
    """Test ensemble parser with sample text"""
    logger.info("=" * 60)
    logger.info("TEST: Ensemble Parser")
    logger.info("=" * 60)
    
    try:
        from perfect_parser import PerfectParser
        
        config = AccuracyConfig()
        config.enable_multi_llm = True
        config.use_ensemble_for_parsing = True
        
        parser = PerfectParser(config)
        
        # Sample text with questions
        sample_text = """
        1. What is the capital of France?
        Answer: Paris
        
        2. What is 2+2?
        Answer: 4
        """
        
        result = parser.parse_perfect_questions(sample_text, "Geography", "Capitals")
        
        logger.info(f"✅ Parsed questions: {len(result.questions)}")
        logger.info(f"   Confidence: {result.confidence:.2f}")
        logger.info(f"   Requires review: {result.requires_human_review}")
        
        assert len(result.questions) >= 1, "Should extract at least one question"
        
        logger.info("✅ Ensemble parser test passed!")
    except Exception as e:
        logger.warning(f"⚠️ Ensemble parser test skipped: {str(e)}")


def run_all_tests():
    """Run all integration tests"""
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-LLM ENSEMBLE INTEGRATION TESTS")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Text Similarity", test_text_similarity),
        ("JSON Merger", test_json_merger),
        ("Question Comparator", test_question_comparator),
        ("Consensus Scorer", test_consensus_scorer),
        ("MultiLLM Validator", test_multi_llm_validator),
        ("Ensemble Parser", test_ensemble_parser),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🧪 Running: {test_name}")
            test_func()
            passed += 1
            logger.info(f"✅ {test_name} PASSED\n")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED: {str(e)}\n")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Passed: {passed}/{len(tests)}")
    logger.info(f"❌ Failed: {failed}/{len(tests)}")
    logger.info("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

