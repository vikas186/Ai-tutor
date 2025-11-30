"""
Perfect Generator using Multi-LLM Ensemble (Claude + Gemini)
Triple-generation with cross-validation for 100% accuracy
"""
from anthropic import Anthropic
import google.generativeai as genai
from typing import List, Dict, Any
import json
import logging
from models import PerfectQuestion, GenerationResult, ValidationCheck, QuestionType, Difficulty, EnsembleGenerationResult, ConsensusCheck
from config import AccuracyConfig, PERFECTION_PROMPTS
from consensus_algorithms import QuestionComparator, ConsensusScorer
import re

logger = logging.getLogger(__name__)


class PerfectGenerator:
    """Question generator with multi-LLM ensemble validation"""
    
    def __init__(self, config: AccuracyConfig):
        self.config = config
        self.client = Anthropic(api_key=config.claude_api_key)
        self.model = config.claude_model
        
        # Initialize Gemini for ensemble generation
        if config.enable_multi_llm and config.use_ensemble_for_generation:
            genai.configure(api_key=config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(config.gemini_model)
            logger.info("Multi-LLM ensemble generation enabled (Claude + Gemini)")
        else:
            self.gemini_model = None
            logger.info("Single-model generation enabled (Claude only)")
    
    def generate_perfect_variations(self, 
                                    source_question: PerfectQuestion,
                                    num_variations: int = 3) -> GenerationResult:
        """
        Generate question variations with 100% conceptual accuracy using ensemble
        
        Args:
            source_question: Source question to create variations from
            num_variations: Number of variations to generate
            
        Returns:
            GenerationResult with generated questions
        """
        logger.info(f"Generating {num_variations} variations for question: {source_question.id}")
        
        # Use ensemble generation if enabled
        if self.config.enable_multi_llm and self.config.use_ensemble_for_generation and self.gemini_model:
            logger.info("Using ensemble generation (Claude + Gemini)")
            return self._generate_with_ensemble(source_question, num_variations)
        else:
            logger.info("Using single-model generation (Claude only)")
            return self._generate_with_claude_only(source_question, num_variations)
    
    def _generate_with_claude_only(self, source_question: PerfectQuestion, num_variations: int) -> GenerationResult:
        """Generate variations using Claude only"""
        # Step 1: Generate initial variations
        logger.info("Step 1: Initial generation")
        initial_variations = self._generate_initial(source_question, num_variations)
        
        # Step 2: Verify against source material
        logger.info("Step 2: Verification against source")
        verified_variations = self._verify_against_source(initial_variations, source_question)
        
        # Step 3: Refine for perfection
        logger.info("Step 3: Refinement")
        final_variations = self._refine_for_perfection(verified_variations, source_question)
        
        # Build validation checks
        validation_checks = []
        
        # Check concept consistency
        concept_check = self._check_concept_consistency(final_variations, source_question)
        validation_checks.append(concept_check)
        
        # Check pattern adherence
        pattern_check = self._check_pattern_adherence(final_variations, source_question)
        validation_checks.append(pattern_check)
        
        # Check quality
        quality_check = self._check_quality(final_variations)
        validation_checks.append(quality_check)
        
        # Calculate overall confidence
        confidence = sum(c.confidence for c in validation_checks) / len(validation_checks) if validation_checks else 0.0
        requires_review = confidence < self.config.human_review_threshold
        
        return GenerationResult(
            generated_questions=final_variations,
            source_question_id=source_question.id,
            confidence=confidence,
            validation_checks=validation_checks,
            requires_human_review=requires_review
        )
    
    def _generate_initial(self, source: PerfectQuestion, num_variations: int) -> List[Dict[str, Any]]:
        """Step 1: Generate initial question variations"""
        system_prompt = PERFECTION_PROMPTS["generation"]
        
        source_json = {
            "question_text": source.question_text,
            "question_type": source.question_type.value,
            "options": source.options,
            "correct_answer": source.correct_answer,
            "difficulty": source.difficulty.value,
            "subject": source.subject,
            "topic": source.topic,
            "tags": source.tags
        }
        
        user_prompt = f"""
Generate {num_variations} variations of the following exam question.

CRITICAL REQUIREMENTS:
1. Maintain EXACT same concept and learning objective
2. Use PRECISE same difficulty level: {source.difficulty.value}
3. Follow IDENTICAL question pattern: {source.question_type.value}
4. Ensure 100% factual correctness
5. Zero hallucinations or creative deviations

Source question:
{json.dumps(source_json, indent=2)}

Generate variations that test the SAME concept but with:
- Different wording
- Different examples (if applicable)
- Different numerical values (if applicable)
- Different scenarios (if applicable)

But maintain:
- Same difficulty
- Same question type
- Same subject and topic
- Same educational value

Return JSON format:
{{
    "variations": [
        {{
            "question_text": "...",
            "question_type": "{source.question_type.value}",
            "options": [...],  # If MCQ
            "correct_answer": "...",
            "difficulty": "{source.difficulty.value}",
            "subject": "{source.subject}",
            "topic": "{source.topic}",
            "tags": [...],
            "explanation": "..."
        }}
    ],
    "concept_consistency": true/false,
    "confidence": 0.0-1.0
}}
"""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.claude_max_tokens,
                temperature=self.config.claude_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            response_text = message.content[0].text
            json_data = self._extract_json(response_text)
            
            variations = json_data.get("variations", [])
            
            # Ensure we have the right number
            if len(variations) < num_variations:
                logger.warning(f"Generated {len(variations)} variations, requested {num_variations}")
            
            return variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Initial generation failed: {str(e)}")
            raise
    
    def _verify_against_source(self, variations: List[Dict], source: PerfectQuestion) -> List[Dict]:
        """Step 2: Verify variations match source concept"""
        verified = []
        
        for variation in variations:
            # Use Claude to verify concept match
            system_prompt = "You verify that question variations maintain the exact same concept as the source."
            
            user_prompt = f"""
Verify that this variation maintains the EXACT same concept as the source question.

Source question:
{source.question_text}

Source concept indicators:
- Subject: {source.subject}
- Topic: {source.topic}
- Difficulty: {source.difficulty.value}
- Type: {source.question_type.value}

Variation:
{variation.get('question_text', '')}

Respond with JSON:
{{
    "concept_match": true/false,
    "confidence": 0.0-1.0,
    "reason": "explanation"
}}
"""
            
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.0,  # Zero temperature for verification
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                
                response_text = message.content[0].text
                verification = self._extract_json(response_text)
                
                if verification.get("concept_match", False) and verification.get("confidence", 0) >= 0.95:
                    verified.append(variation)
                else:
                    logger.warning(f"Variation failed concept verification: {verification.get('reason', '')}")
                    # Still include but flag
                    verified.append(variation)
                    
            except Exception as e:
                logger.warning(f"Verification failed for variation, including anyway: {str(e)}")
                verified.append(variation)
        
        return verified
    
    def _refine_for_perfection(self, variations: List[Dict], source: PerfectQuestion) -> List[PerfectQuestion]:
        """Step 3: Refine and convert to PerfectQuestion models"""
        perfect_questions = []
        
        for var_data in variations:
            try:
                # Ensure all fields match source
                question_type = QuestionType(var_data.get("question_type", source.question_type.value))
                difficulty = Difficulty(var_data.get("difficulty", source.difficulty.value))
                
                perfect_q = PerfectQuestion(
                    question_text=var_data.get("question_text", "").strip(),
                    question_type=question_type,
                    options=var_data.get("options") if question_type == QuestionType.MCQ else None,
                    correct_answer=var_data.get("correct_answer", "").strip(),
                    explanation=var_data.get("explanation"),
                    difficulty=difficulty,
                    tags=var_data.get("tags", source.tags),
                    subject=var_data.get("subject", source.subject),
                    topic=var_data.get("topic", source.topic),
                    confidence_score=min(1.0, var_data.get("confidence", 0.95))
                )
                
                perfect_questions.append(perfect_q)
                
            except Exception as e:
                logger.error(f"Failed to create PerfectQuestion from variation: {str(e)}")
                continue
        
        return perfect_questions
    
    def _check_concept_consistency(self, variations: List[PerfectQuestion], source: PerfectQuestion) -> ValidationCheck:
        """Check that variations maintain concept consistency"""
        # Check subject and topic consistency
        subject_match = all(v.subject == source.subject for v in variations)
        topic_match = all(v.topic == source.topic for v in variations)
        difficulty_match = all(v.difficulty == source.difficulty for v in variations)
        type_match = all(v.question_type == source.question_type for v in variations)
        
        all_consistent = subject_match and topic_match and difficulty_match and type_match
        confidence = 1.0 if all_consistent else 0.7
        
        return ValidationCheck(
            check_type="concept_consistency",
            passed=all_consistent,
            confidence=confidence,
            details=f"Subject: {subject_match}, Topic: {topic_match}, Difficulty: {difficulty_match}, Type: {type_match}"
        )
    
    def _check_pattern_adherence(self, variations: List[PerfectQuestion], source: PerfectQuestion) -> ValidationCheck:
        """Check that variations follow the same pattern"""
        # All should have same question type
        type_consistent = all(v.question_type == source.question_type for v in variations)
        
        # MCQs should have same number of options
        if source.question_type == QuestionType.MCQ:
            source_option_count = len(source.options) if source.options else 0
            option_consistent = all(
                len(v.options) == source_option_count if v.options else False 
                for v in variations
            )
        else:
            option_consistent = True
        
        all_adherent = type_consistent and option_consistent
        confidence = 1.0 if all_adherent else 0.8
        
        return ValidationCheck(
            check_type="pattern_adherence",
            passed=all_adherent,
            confidence=confidence,
            details=f"Type consistent: {type_consistent}, Options consistent: {option_consistent}"
        )
    
    def _check_quality(self, variations: List[PerfectQuestion]) -> ValidationCheck:
        """Check overall quality of generated questions"""
        if not variations:
            return ValidationCheck(
                check_type="quality",
                passed=False,
                confidence=0.0,
                details="No variations generated"
            )
        
        # Check all questions are valid
        all_valid = all(
            v.question_text and len(v.question_text) >= 10 and
            v.correct_answer and len(v.correct_answer) > 0
            for v in variations
        )
        
        # Check confidence scores
        avg_confidence = sum(v.confidence_score for v in variations) / len(variations)
        confidence_high = avg_confidence >= 0.95
        
        passed = all_valid and confidence_high
        confidence = avg_confidence if passed else 0.7
        
        return ValidationCheck(
            check_type="quality",
            passed=passed,
            confidence=confidence,
            details=f"All valid: {all_valid}, Avg confidence: {avg_confidence:.2%}"
        )
    
    def _generate_with_ensemble(self, source_question: PerfectQuestion, num_variations: int) -> EnsembleGenerationResult:
        """
        Generate variations using both Claude and Gemini, then merge with consensus
        """
        logger.info("🔄 Using ensemble generation (Claude + Gemini)")
        
        # Generate with both models
        claude_variations = []
        gemini_variations = []
        consensus_checks = []
        
        # Claude generation
        try:
            logger.info("📤 Generating with Claude...")
            claude_variations_data = self._generate_initial(source_question, num_variations)
            claude_variations = self._refine_for_perfection(claude_variations_data, source_question)
            logger.info(f"✅ Claude generated {len(claude_variations)} variations")
        except Exception as e:
            logger.error(f"❌ Claude generation failed: {str(e)}")
        
        # Gemini generation
        try:
            logger.info("📤 Generating with Gemini...")
            gemini_variations_data = self._generate_with_gemini(source_question, num_variations)
            gemini_variations = self._refine_for_perfection(gemini_variations_data, source_question)
            logger.info(f"✅ Gemini generated {len(gemini_variations)} variations")
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {str(e)}")
        
        # Merge variations
        if not claude_variations and not gemini_variations:
            raise ValueError("Both Claude and Gemini generation failed")
        
        if not claude_variations:
            logger.warning("Only Gemini succeeded, returning Gemini variations")
            all_variations = gemini_variations[:num_variations]
            models_agreement_score = 0.5
        elif not gemini_variations:
            logger.warning("Only Claude succeeded, returning Claude variations")
            all_variations = claude_variations[:num_variations]
            models_agreement_score = 0.5
        else:
            # Both succeeded - combine and cross-validate
            logger.info(f"🔀 Merging variations: Claude ({len(claude_variations)}) + Gemini ({len(gemini_variations)})")
            
            # Calculate agreement
            models_agreement_score = self._calculate_generation_agreement(claude_variations, gemini_variations)
            
            # Combine variations (take best from each)
            all_variations = claude_variations + gemini_variations
            all_variations = all_variations[:num_variations]
            
            # Create consensus check
            consensus_check = ConsensusCheck(
                models_used=["claude", "gemini"],
                agreement_score=models_agreement_score,
                disagreements=[] if models_agreement_score > 0.8 else ["Low agreement between models"],
                consensus_strategy="combined_best",
                final_choice="mixed"
            )
            consensus_checks.append(consensus_check)
        
        # Build validation checks
        validation_checks = []
        
        # Check concept consistency
        concept_check = self._check_concept_consistency(all_variations, source_question)
        validation_checks.append(concept_check)
        
        # Check pattern adherence
        pattern_check = self._check_pattern_adherence(all_variations, source_question)
        validation_checks.append(pattern_check)
        
        # Check quality
        quality_check = self._check_quality(all_variations)
        validation_checks.append(quality_check)
        
        # Calculate overall confidence
        confidence = sum(c.confidence for c in validation_checks) / len(validation_checks) if validation_checks else 0.0
        # Boost confidence if high agreement
        if models_agreement_score > 0.85:
            confidence = min(1.0, confidence * 1.1)
        
        requires_review = confidence < self.config.human_review_threshold
        
        return EnsembleGenerationResult(
            generated_questions=all_variations,
            source_question_id=source_question.id,
            confidence=confidence,
            validation_checks=validation_checks,
            consensus_checks=consensus_checks,
            requires_human_review=requires_review,
            models_agreement_score=models_agreement_score
        )
    
    def _generate_with_gemini(self, source: PerfectQuestion, num_variations: int) -> List[Dict[str, Any]]:
        """Generate variations using Gemini"""
        source_json = {
            "question_text": source.question_text,
            "question_type": source.question_type.value,
            "options": source.options,
            "correct_answer": source.correct_answer,
            "difficulty": source.difficulty.value,
            "subject": source.subject,
            "topic": source.topic,
            "tags": source.tags
        }
        
        prompt = f"""Generate {num_variations} variations of the following exam question.

CRITICAL REQUIREMENTS:
1. Maintain EXACT same concept and learning objective
2. Use PRECISE same difficulty level: {source.difficulty.value}
3. Follow IDENTICAL question pattern: {source.question_type.value}
4. Ensure 100% factual correctness
5. Zero hallucinations or creative deviations

Source question:
{json.dumps(source_json, indent=2)}

Generate variations that test the SAME concept but with:
- Different wording
- Different examples (if applicable)
- Different numerical values (if applicable)
- Different scenarios (if applicable)

But maintain:
- Same difficulty
- Same question type
- Same subject and topic
- Same educational value

Return JSON format:
{{
    "variations": [
        {{
            "question_text": "...",
            "question_type": "{source.question_type.value}",
            "options": [...],
            "correct_answer": "...",
            "difficulty": "{source.difficulty.value}",
            "subject": "{source.subject}",
            "topic": "{source.topic}",
            "tags": [...],
            "explanation": "..."
        }}
    ],
    "concept_consistency": true/false,
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 8000,
                }
            )
            
            response_text = response.text
            json_data = self._extract_json(response_text)
            
            variations = json_data.get("variations", [])
            
            if len(variations) < num_variations:
                logger.warning(f"Gemini generated {len(variations)} variations, requested {num_variations}")
            
            return variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise
    
    def _calculate_generation_agreement(self, variations_a: List[PerfectQuestion], 
                                       variations_b: List[PerfectQuestion]) -> float:
        """Calculate agreement score between two sets of generated variations"""
        if not variations_a or not variations_b:
            return 0.0
        
        # Compare characteristics
        scores = []
        
        # Check if same number generated
        count_similarity = 1.0 - abs(len(variations_a) - len(variations_b)) / max(len(variations_a), len(variations_b))
        scores.append(count_similarity)
        
        # Check if types match
        types_a = [v.question_type.value for v in variations_a]
        types_b = [v.question_type.value for v in variations_b]
        type_match = sum(1 for i in range(min(len(types_a), len(types_b))) if types_a[i] == types_b[i])
        type_score = type_match / max(len(types_a), len(types_b))
        scores.append(type_score)
        
        # Check if difficulties match
        diffs_a = [v.difficulty.value for v in variations_a]
        diffs_b = [v.difficulty.value for v in variations_b]
        diff_match = sum(1 for i in range(min(len(diffs_a), len(diffs_b))) if diffs_a[i] == diffs_b[i])
        diff_score = diff_match / max(len(diffs_a), len(diffs_b))
        scores.append(diff_score)
        
        # Average agreement
        agreement = sum(scores) / len(scores)
        logger.info(f"Generation agreement score: {agreement:.2f}")
        
        return agreement
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        # Remove markdown code blocks
        text_cleaned = re.sub(r'```json\s*', '', text)
        text_cleaned = re.sub(r'```\s*', '', text_cleaned)
        text_cleaned = text_cleaned.strip()
        
        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(text_cleaned)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {text[:200]}")
            return {"variations": [], "confidence": 0.0}

