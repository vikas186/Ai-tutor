"""
Multi-LLM Validator Module
Coordinates multiple LLM calls for ensemble validation and consensus
"""
from typing import List, Dict, Any, Optional, Callable, Tuple
from anthropic import Anthropic
import google.generativeai as genai
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    GEMINI = "gemini"


class ConsensusStrategy(str, Enum):
    """Consensus strategies for merging results"""
    VOTING = "voting"
    LONGEST = "longest"
    CLAUDE_PRIORITY = "claude_priority"
    GEMINI_PRIORITY = "gemini_priority"
    HYBRID = "hybrid"


@dataclass
class LLMResponse:
    """Single LLM response"""
    provider: LLMProvider
    content: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class EnsembleResult:
    """Result from ensemble of multiple LLMs"""
    final_content: str
    confidence: float
    individual_responses: List[LLMResponse]
    consensus_score: float
    strategy_used: ConsensusStrategy
    disagreements: List[str]


class MultiLLMValidator:
    """
    Coordinates multiple LLM calls for improved accuracy through ensemble validation
    """
    
    def __init__(self, claude_api_key: str, gemini_api_key: str, 
                 claude_model: str = "claude-3-5-sonnet-20240620",
                 gemini_model: str = "models/gemini-2.0-flash",
                 consensus_threshold: float = 0.8):
        """
        Initialize multi-LLM validator
        
        Args:
            claude_api_key: Anthropic API key
            gemini_api_key: Google API key
            claude_model: Claude model to use
            gemini_model: Gemini model to use
            consensus_threshold: Threshold for considering responses in consensus
        """
        self.claude_client = Anthropic(api_key=claude_api_key)
        self.claude_model = claude_model
        
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(gemini_model)
        
        self.consensus_threshold = consensus_threshold
        
        logger.info(f"Multi-LLM Validator initialized with Claude ({claude_model}) and Gemini ({gemini_model})")
    
    def generate_with_consensus(self, 
                               prompt: str,
                               system_prompt: Optional[str] = None,
                               strategy: ConsensusStrategy = ConsensusStrategy.HYBRID,
                               temperature: float = 0.1,
                               max_tokens: int = 8192) -> EnsembleResult:
        """
        Generate response using multiple LLMs and merge with consensus
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            strategy: Consensus strategy to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            EnsembleResult with merged response
        """
        logger.info(f"Generating with consensus using strategy: {strategy}")
        
        # Generate responses from both models
        responses = []
        
        # Claude response
        try:
            claude_response = self._generate_claude(prompt, system_prompt, temperature, max_tokens)
            responses.append(claude_response)
            logger.info(f"Claude response: {len(claude_response.content)} chars")
        except Exception as e:
            logger.error(f"Claude generation failed: {str(e)}")
        
        # Gemini response
        try:
            gemini_response = self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
            responses.append(gemini_response)
            logger.info(f"Gemini response: {len(gemini_response.content)} chars")
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
        
        # If both failed, raise error
        if not responses:
            raise ValueError("All LLM providers failed to generate response")
        
        # If only one succeeded, return it
        if len(responses) == 1:
            logger.warning(f"Only {responses[0].provider} succeeded, returning single response")
            return EnsembleResult(
                final_content=responses[0].content,
                confidence=responses[0].confidence * 0.8,  # Reduced confidence for single model
                individual_responses=responses,
                consensus_score=0.5,
                strategy_used=strategy,
                disagreements=["Only one model succeeded"]
            )
        
        # Merge responses using consensus strategy
        return self._merge_responses(responses, strategy)
    
    def cross_validate(self, 
                      content: str,
                      validation_prompt: str,
                      system_prompt: Optional[str] = None) -> Tuple[bool, float, List[str]]:
        """
        Cross-validate content using multiple LLMs
        
        Args:
            content: Content to validate
            validation_prompt: Validation prompt
            system_prompt: System prompt for validation
            
        Returns:
            Tuple of (is_valid, confidence, issues_found)
        """
        logger.info("Cross-validating with multiple LLMs")
        
        validations = []
        
        # Claude validation
        try:
            claude_valid, claude_conf, claude_issues = self._validate_with_claude(
                content, validation_prompt, system_prompt
            )
            validations.append((LLMProvider.CLAUDE, claude_valid, claude_conf, claude_issues))
        except Exception as e:
            logger.error(f"Claude validation failed: {str(e)}")
        
        # Gemini validation
        try:
            gemini_valid, gemini_conf, gemini_issues = self._validate_with_gemini(
                content, validation_prompt, system_prompt
            )
            validations.append((LLMProvider.GEMINI, gemini_valid, gemini_conf, gemini_issues))
        except Exception as e:
            logger.error(f"Gemini validation failed: {str(e)}")
        
        if not validations:
            return False, 0.0, ["All validators failed"]
        
        # Compute consensus validation
        valid_count = sum(1 for _, valid, _, _ in validations if valid)
        avg_confidence = sum(conf for _, _, conf, _ in validations) / len(validations)
        all_issues = []
        for _, _, _, issues in validations:
            all_issues.extend(issues)
        
        # Consensus: both agree it's valid
        is_valid = valid_count >= len(validations) * self.consensus_threshold
        
        logger.info(f"Cross-validation result: {is_valid} (confidence: {avg_confidence:.2f})")
        
        return is_valid, avg_confidence, all_issues
    
    def compare_outputs(self, 
                       output_a: Any,
                       output_b: Any,
                       comparison_metric: Callable[[Any, Any], float]) -> float:
        """
        Compare two outputs using a custom metric
        
        Args:
            output_a: First output
            output_b: Second output
            comparison_metric: Function that returns similarity score (0-1)
            
        Returns:
            Similarity score between 0 and 1
        """
        return comparison_metric(output_a, output_b)
    
    def _generate_claude(self, prompt: str, system_prompt: Optional[str], 
                        temperature: float, max_tokens: int) -> LLMResponse:
        """Generate response using Claude"""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.claude_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.claude_client.messages.create(**kwargs)
        
        content = response.content[0].text
        
        return LLMResponse(
            provider=LLMProvider.CLAUDE,
            content=content,
            confidence=0.95,  # Base confidence for Claude
            metadata={
                "model": self.claude_model,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        )
    
    def _generate_gemini(self, prompt: str, system_prompt: Optional[str],
                        temperature: float, max_tokens: int) -> LLMResponse:
        """Generate response using Gemini"""
        # Combine system prompt and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = self.gemini_model.generate_content(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        
        content = response.text
        
        return LLMResponse(
            provider=LLMProvider.GEMINI,
            content=content,
            confidence=0.95,  # Base confidence for Gemini
            metadata={
                "model": "gemini-2.0-flash",
                "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
            }
        )
    
    def _merge_responses(self, responses: List[LLMResponse], 
                        strategy: ConsensusStrategy) -> EnsembleResult:
        """
        Merge multiple responses using specified strategy
        
        Args:
            responses: List of LLM responses
            strategy: Consensus strategy
            
        Returns:
            EnsembleResult with merged response
        """
        logger.info(f"Merging {len(responses)} responses with strategy: {strategy}")
        
        # Calculate consensus score (similarity between responses)
        consensus_score = self._calculate_consensus_score(responses)
        
        # Detect disagreements
        disagreements = self._detect_disagreements(responses)
        
        # Apply strategy
        if strategy == ConsensusStrategy.VOTING:
            final_content, confidence = self._voting_strategy(responses)
        elif strategy == ConsensusStrategy.LONGEST:
            final_content, confidence = self._longest_strategy(responses)
        elif strategy == ConsensusStrategy.CLAUDE_PRIORITY:
            final_content, confidence = self._priority_strategy(responses, LLMProvider.CLAUDE)
        elif strategy == ConsensusStrategy.GEMINI_PRIORITY:
            final_content, confidence = self._priority_strategy(responses, LLMProvider.GEMINI)
        elif strategy == ConsensusStrategy.HYBRID:
            final_content, confidence = self._hybrid_strategy(responses, consensus_score)
        else:
            # Default to hybrid
            final_content, confidence = self._hybrid_strategy(responses, consensus_score)
        
        # Boost confidence if high consensus
        if consensus_score > 0.9:
            confidence = min(1.0, confidence * 1.1)
        elif consensus_score < 0.5:
            confidence *= 0.8
        
        return EnsembleResult(
            final_content=final_content,
            confidence=confidence,
            individual_responses=responses,
            consensus_score=consensus_score,
            strategy_used=strategy,
            disagreements=disagreements
        )
    
    def _calculate_consensus_score(self, responses: List[LLMResponse]) -> float:
        """Calculate similarity/consensus between responses"""
        if len(responses) < 2:
            return 1.0
        
        # Simple consensus: compare lengths and basic similarity
        lengths = [len(r.content) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        
        # Length similarity
        length_variance = max(abs(l - avg_length) / avg_length for l in lengths if avg_length > 0)
        length_similarity = max(0, 1.0 - length_variance)
        
        # Basic text similarity (could be enhanced with better algorithms)
        # For now, check if responses contain similar keywords
        words_sets = [set(r.content.lower().split()) for r in responses]
        if len(words_sets) >= 2:
            intersection = words_sets[0].intersection(words_sets[1])
            union = words_sets[0].union(words_sets[1])
            text_similarity = len(intersection) / len(union) if union else 0
        else:
            text_similarity = 1.0
        
        # Combined score
        consensus = (length_similarity * 0.3 + text_similarity * 0.7)
        
        logger.info(f"Consensus score: {consensus:.2f} (length: {length_similarity:.2f}, text: {text_similarity:.2f})")
        
        return consensus
    
    def _detect_disagreements(self, responses: List[LLMResponse]) -> List[str]:
        """Detect significant disagreements between responses"""
        disagreements = []
        
        if len(responses) < 2:
            return disagreements
        
        # Check length differences
        lengths = [len(r.content) for r in responses]
        if max(lengths) > min(lengths) * 2:
            disagreements.append(f"Significant length difference: {min(lengths)} vs {max(lengths)}")
        
        # Check if both contain similar structure (JSON, etc.)
        has_json = ['{' in r.content and '}' in r.content for r in responses]
        if not all(has_json) and any(has_json):
            disagreements.append("Structure mismatch: one response has JSON, other doesn't")
        
        return disagreements
    
    def _voting_strategy(self, responses: List[LLMResponse]) -> Tuple[str, float]:
        """Use voting strategy (currently returns highest confidence)"""
        best = max(responses, key=lambda r: r.confidence)
        return best.content, best.confidence
    
    def _longest_strategy(self, responses: List[LLMResponse]) -> Tuple[str, float]:
        """Use longest response (often most complete)"""
        longest = max(responses, key=lambda r: len(r.content))
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        return longest.content, avg_confidence
    
    def _priority_strategy(self, responses: List[LLMResponse], 
                          priority_provider: LLMProvider) -> Tuple[str, float]:
        """Use response from priority provider"""
        priority_response = next((r for r in responses if r.provider == priority_provider), None)
        if priority_response:
            return priority_response.content, priority_response.confidence
        else:
            # Fallback to first response
            return responses[0].content, responses[0].confidence
    
    def _hybrid_strategy(self, responses: List[LLMResponse], 
                        consensus_score: float) -> Tuple[str, float]:
        """
        Hybrid strategy: Use Claude if high consensus, otherwise use longest
        """
        # If high consensus, prefer Claude (more reliable)
        if consensus_score > 0.8:
            claude_response = next((r for r in responses if r.provider == LLMProvider.CLAUDE), None)
            if claude_response:
                logger.info("High consensus: using Claude response")
                return claude_response.content, claude_response.confidence
        
        # Otherwise, use longest (most complete)
        logger.info("Low consensus or no Claude: using longest response")
        return self._longest_strategy(responses)
    
    def _validate_with_claude(self, content: str, validation_prompt: str,
                             system_prompt: Optional[str]) -> Tuple[bool, float, List[str]]:
        """Validate content using Claude"""
        full_prompt = f"{validation_prompt}\n\nContent to validate:\n{content}"
        
        response = self._generate_claude(full_prompt, system_prompt, 0.0, 1000)
        
        # Parse validation response (expecting JSON with validation results)
        content_lower = response.content.lower()
        is_valid = 'true' in content_lower or 'valid' in content_lower or 'passed' in content_lower
        
        # Extract confidence if present
        confidence = 0.95 if is_valid else 0.5
        
        # Extract issues
        issues = []
        if not is_valid:
            issues.append("Claude validation failed")
        
        return is_valid, confidence, issues
    
    def _validate_with_gemini(self, content: str, validation_prompt: str,
                              system_prompt: Optional[str]) -> Tuple[bool, float, List[str]]:
        """Validate content using Gemini"""
        full_prompt = f"{validation_prompt}\n\nContent to validate:\n{content}"
        
        response = self._generate_gemini(full_prompt, system_prompt, 0.0, 1000)
        
        # Parse validation response
        content_lower = response.content.lower()
        is_valid = 'true' in content_lower or 'valid' in content_lower or 'passed' in content_lower
        
        confidence = 0.95 if is_valid else 0.5
        
        issues = []
        if not is_valid:
            issues.append("Gemini validation failed")
        
        return is_valid, confidence, issues

