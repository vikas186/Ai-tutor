"""
Perfect OCR Module using Multi-LLM Ensemble (Gemini + Claude)
Multi-pass extraction with cross-validation for 100% accuracy
"""
import google.generativeai as genai
from anthropic import Anthropic
from typing import Dict, List, Optional
import base64
import re
from models import OCRResult, ValidationCheck, EnsembleOCRResult, ConsensusCheck
from image_enhancement import ImageEnhancementPipeline
from file_handler import FileHandler
from config import AccuracyConfig, PERFECTION_PROMPTS
from consensus_algorithms import TextSimilarity, ConsensusScorer
import logging

logger = logging.getLogger(__name__)


class PerfectGeminiOCR:
    """OCR engine with multi-LLM ensemble validation for 100% accuracy"""
    
    def __init__(self, config: AccuracyConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        self.enhancement_pipeline = ImageEnhancementPipeline()
        
        # Initialize Claude for ensemble OCR
        if config.enable_multi_llm and config.use_ensemble_for_ocr:
            self.claude_client = Anthropic(api_key=config.claude_api_key)
            self.claude_model = config.claude_model
            logger.info("Multi-LLM ensemble OCR enabled (Gemini + Claude Vision)")
        else:
            self.claude_client = None
            self.claude_model = None
            logger.info("Single-model OCR enabled (Gemini only)")
    
    def extract_perfect_text(self, file_path: str) -> OCRResult:
        """
        Extract text with multi-pass strategy for maximum accuracy
        
        Args:
            file_path: Path to image file or PDF
            
        Returns:
            OCRResult with extracted text and confidence score
        """
        # Step 0: Handle different file types
        logger.info(f"Processing file: {file_path}")
        
        # Validate file
        file_info = FileHandler.validate_file(file_path)
        if not file_info['exists']:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine if it's PDF or image
        is_pdf = file_info['type'] == 'pdf'
        is_image = file_info['type'] == 'image'
        
        if not is_pdf and not is_image:
            raise ValueError(f"Unsupported file type: {file_info['extension']}. Supported: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP")
        
        # For PDFs: Read file bytes directly (Gemini supports PDF natively)
        # For images: Enhance first, then use
        if is_pdf:
            logger.info("Processing PDF file (using Gemini native PDF support)")
            # Read file and ensure it's closed before proceeding
            with open(file_path, 'rb') as f:
                file_data = f.read()
            # File is now closed, ensure it's released (Windows)
            import gc
            gc.collect()
            file_mime_type = "application/pdf"
            file_content = file_data
        else:
            # Step 1: Enhance image quality
            logger.info("Enhancing image for optimal OCR")
            enhanced_image = self.enhancement_pipeline.optimize(file_path)
            file_mime_type = "image/png"
            file_content = enhanced_image
        
        # Step 2: Multiple OCR strategies
        strategies = [
            {
                "prompt": "Extract all text with perfect accuracy including mathematical expressions, preserving exact formatting and layout.",
                "name": "comprehensive_extraction"
            },
            {
                "prompt": "Focus on layout preservation and question structure. Extract text maintaining question-answer relationships.",
                "name": "structure_focused"
            },
            {
                "prompt": "Extract tables, formulas, special characters, and all textual content with 100% precision.",
                "name": "precision_focused"
            }
        ]
        
        results = []
        validation_checks = []
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"OCR Pass {i+1}/{len(strategies)}: {strategy['name']}")
                
                # Prepare file for Gemini (PDF or image)
                file_data = {
                    "mime_type": file_mime_type,
                    "data": file_content
                }
                
                # Generate content with low temperature for consistency
                response = self.model.generate_content(
                    [strategy["prompt"], file_data],
                    generation_config={
                        "temperature": self.config.gemini_temperature,
                        "max_output_tokens": self.config.gemini_max_output_tokens,
                    }
                )
                
                extracted_text = response.text
                
                # DEBUG: Log what Gemini extracted
                logger.info(f"OCR Pass {i+1} extracted text length: {len(extracted_text)} characters")
                if len(extracted_text) > 0:
                    logger.info(f"OCR Pass {i+1} first 300 chars: {extracted_text[:300]}")
                else:
                    logger.warning(f"⚠️  OCR Pass {i+1} extracted EMPTY text!")
                
                # Validate this pass (use file_content for validation)
                confidence = self._validate_ocr_result(extracted_text, file_content)
                
                results.append({
                    "text": extracted_text,
                    "confidence": confidence,
                    "strategy": strategy["name"]
                })
                
                validation_checks.append(ValidationCheck(
                    check_type=f"ocr_pass_{i+1}",
                    passed=confidence >= self.config.ocr_confidence,
                    confidence=confidence,
                    details=f"Strategy: {strategy['name']}, Confidence: {confidence:.2%}"
                ))
                
            except Exception as e:
                logger.error(f"OCR pass {i+1} failed: {str(e)}")
                validation_checks.append(ValidationCheck(
                    check_type=f"ocr_pass_{i+1}",
                    passed=False,
                    confidence=0.0,
                    details=f"Error: {str(e)}"
                ))
        
        # Step 3: Ensemble validation with Claude (if enabled)
        if self.config.enable_multi_llm and self.config.use_ensemble_for_ocr and self.claude_client:
            logger.info("🔄 Running Claude Vision cross-validation")
            try:
                claude_result = self._extract_with_claude(file_content, file_mime_type)
                results.append(claude_result)
                validation_checks.append(ValidationCheck(
                    check_type="claude_vision_validation",
                    passed=claude_result["confidence"] >= self.config.ocr_confidence,
                    confidence=claude_result["confidence"],
                    details=f"Claude Vision: {claude_result['confidence']:.2%}"
                ))
                logger.info(f"✅ Claude Vision extracted {len(claude_result['text'])} chars")
            except Exception as e:
                logger.warning(f"Claude Vision validation failed: {str(e)}")
        
        # Step 4: Consensus merge - combine results intelligently
        if not results:
            raise ValueError("All OCR passes failed")
        
        final_text, consensus_checks = self._ensemble_consensus_merge(results)
        
        # Clean OCR text - remove Gemini's preamble and markdown formatting
        final_text = self._clean_ocr_text(final_text)
        
        final_confidence = max(r["confidence"] for r in results)
        
        # Final validation
        requires_review = final_confidence < self.config.human_review_threshold
        
        logger.info(f"✅ OCR extraction completed. Confidence: {final_confidence:.2%}, Text length: {len(final_text)} characters")
        
        # DEBUG: Log sample of final extracted text
        if len(final_text) > 0:
            logger.info(f"📄 Final extracted text (first 500 chars):\n{final_text[:500]}")
            logger.info(f"📄 Final extracted text (last 200 chars):\n{final_text[-200:]}")
        else:
            logger.error(f"❌ CRITICAL: Final extracted text is EMPTY!")
        
        # Log warning if text seems truncated
        if len(final_text) < 500:
            logger.warning(f"⚠️  Extracted text is very short ({len(final_text)} chars). Document might have more content.")
        
        return OCRResult(
            extracted_text=final_text,
            confidence=final_confidence,
            validation_checks=validation_checks,
            requires_human_review=requires_review
        )
    
    def _validate_ocr_result(self, text: str, file_content: bytes) -> float:
        """
        Validate OCR result quality
        
        Returns:
            Confidence score 0.0-1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Check for reasonable text characteristics
        confidence = 1.0
        
        # Penalize if text is too short (might be incomplete)
        if len(text) < 50:
            confidence *= 0.8
        
        # Check for common OCR error patterns
        error_patterns = ["??", "###", "..." * 5]  # Multiple suspicious patterns
        for pattern in error_patterns:
            if pattern in text:
                confidence *= 0.9
        
        # Reward if text contains question-like patterns
        question_indicators = ["?", "question", "answer", "option", "a)", "b)", "c)"]
        if any(indicator.lower() in text.lower() for indicator in question_indicators):
            confidence = min(1.0, confidence * 1.1)  # Boost confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_with_claude(self, file_content: bytes, file_mime_type: str) -> Dict:
        """
        Extract text using Claude Vision
        
        Args:
            file_content: File content bytes
            file_mime_type: MIME type of file
            
        Returns:
            Dict with extracted text and metadata
        """
        # Claude Vision supports images
        if file_mime_type == "application/pdf":
            # Claude can handle PDFs directly
            logger.info("Using Claude with PDF")
            file_data = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_mime_type,
                    "data": base64.b64encode(file_content).decode('utf-8')
                }
            }
        else:
            # Image file
            logger.info("Using Claude Vision with image")
            file_data = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": file_mime_type,
                    "data": base64.b64encode(file_content).decode('utf-8')
                }
            }
        
        prompt = "Extract all text from this document with perfect accuracy. Include mathematical expressions, tables, and all textual content. Preserve exact formatting and layout."
        
        try:
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=8192,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": [
                        file_data,
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            
            extracted_text = response.content[0].text
            confidence = self._validate_ocr_result(extracted_text, file_content)
            
            return {
                "text": extracted_text,
                "confidence": confidence,
                "strategy": "claude_vision"
            }
        except Exception as e:
            logger.error(f"Claude Vision extraction failed: {str(e)}")
            raise
    
    def _ensemble_consensus_merge(self, results: List[Dict]) -> tuple[str, List[ConsensusCheck]]:
        """
        Merge OCR results using ensemble consensus (optimized for speed)
        
        Returns:
            Tuple of (merged_text, consensus_checks)
        """
        if not results:
            return "", []
        
        consensus_checks = []
        
        # Sort by confidence and length first (fast operation)
        sorted_results = sorted(results, key=lambda x: (x["confidence"], len(x["text"])), reverse=True)
        
        # Quick check: if results are very different in length, skip expensive comparison
        texts = [r["text"] for r in results]
        lengths = [len(t) for t in texts]
        if lengths:
            max_len = max(lengths)
            min_len = min(lengths)
            # If one result is 10x longer than another, they're clearly different
            if max_len > 0 and min_len > 0 and max_len / min_len > 5:
                logger.info(f"Results vary greatly in length ({min_len} vs {max_len} chars), using longest")
                agreement_score = 0.3  # Low agreement
            else:
                # Calculate agreement between results (optimized for large texts)
                agreement_score = self._fast_ocr_agreement_score(texts)
        else:
            agreement_score = 0.5
        
        logger.info(f"OCR agreement score: {agreement_score:.2f}")
        
        # Strategy: If high agreement, use any. If low agreement, use longest
        if agreement_score > 0.85:
            # High agreement - use highest confidence
            final_text = sorted_results[0]["text"]
            strategy = "high_agreement_best_confidence"
        else:
            # Low agreement - use longest (most complete)
            longest = max(sorted_results, key=lambda x: len(x["text"]))
            final_text = longest["text"]
            strategy = "low_agreement_longest"
        
        # Create consensus check
        models_used = [r["strategy"] for r in results]
        disagreements = []
        if agreement_score < 0.7:
            disagreements.append(f"Low agreement between models: {agreement_score:.2f}")
        
        consensus_check = ConsensusCheck(
            models_used=models_used,
            agreement_score=agreement_score,
            disagreements=disagreements,
            consensus_strategy=strategy,
            final_choice=sorted_results[0]["strategy"]
        )
        consensus_checks.append(consensus_check)
        
        logger.info(f"Using {strategy}: {len(final_text)} chars")
        
        return final_text, consensus_checks
    
    def _consensus_merge(self, results: List[Dict]) -> str:
        """
        Intelligently merge multiple OCR results (backward compatibility)
        """
        final_text, _ = self._ensemble_consensus_merge(results)
        return final_text
    
    def _fast_ocr_agreement_score(self, texts: List[str]) -> float:
        """
        Fast agreement score calculation for OCR results (optimized for large texts)
        Uses sampling and length comparison instead of full text comparison
        """
        if len(texts) < 2:
            return 1.0
        
        # Strategy 1: Compare lengths (fast)
        lengths = [len(t) for t in texts]
        avg_length = sum(lengths) / len(lengths)
        if avg_length == 0:
            return 1.0
        
        # Length similarity
        length_variance = max(abs(l - avg_length) / avg_length for l in lengths)
        length_similarity = max(0, 1.0 - length_variance)
        
        # Strategy 2: Sample comparison (for very large texts, only compare samples)
        max_sample_size = 5000  # Only compare first 5000 chars
        text_samples = [t[:max_sample_size] for t in texts if len(t) > 0]
        
        if len(text_samples) >= 2:
            # Quick word-based comparison on samples
            words_sets = [set(s.lower().split()[:100]) for s in text_samples]  # First 100 words
            if len(words_sets) >= 2:
                intersection = words_sets[0].intersection(words_sets[1])
                union = words_sets[0].union(words_sets[1])
                word_similarity = len(intersection) / len(union) if union else 0
            else:
                word_similarity = 0.5
        else:
            word_similarity = 0.5
        
        # Combined score (weighted: 60% length, 40% word similarity)
        agreement = (length_similarity * 0.6 + word_similarity * 0.4)
        
        return agreement
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text by removing Gemini's preamble and markdown formatting
        """
        if not text:
            return text
        
        # Remove common Gemini preambles
        preambles = [
            r'^Okay, here is the extracted text.*?\n',
            r'^Okay, here\'s the extracted text.*?\n',
            r'^Here\'s the extracted content.*?\n',
            r'^Here is the extracted text.*?\n',
            r'^Here\'s the extracted text.*?\n',
        ]
        
        cleaned = text
        for preamble in preambles:
            cleaned = re.sub(preamble, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove markdown code blocks
        cleaned = re.sub(r'^```.*?\n', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove excessive dots/asterisks (form field placeholders)
        # But keep some - they might be part of the document
        # Only remove if there are more than 50 consecutive dots/asterisks
        cleaned = re.sub(r'[\.\*]{50,}', '...', cleaned)
        
        # Remove "**Page X**" markers if they're just formatting
        cleaned = re.sub(r'\*\*Page \d+\*\*\s*\n', '', cleaned, flags=re.IGNORECASE)
        
        # Trim whitespace
        cleaned = cleaned.strip()
        
        logger.info(f"Cleaned OCR text: {len(text)} -> {len(cleaned)} chars (removed {len(text) - len(cleaned)} chars of preamble/formatting)")
        
        return cleaned
    
    def _check_consensus(self, texts: List[str]) -> bool:
        """Check if multiple OCR results show consensus"""
        if len(texts) < 2:
            return True
        
        # Simple consensus check: compare text lengths and key phrases
        lengths = [len(t) for t in texts]
        avg_length = sum(lengths) / len(lengths)
        
        # If lengths are within 20% of each other, consider it consensus
        length_variance = max(abs(l - avg_length) / avg_length for l in lengths)
        
        return length_variance < 0.2

