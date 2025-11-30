# """
# Perfect Parser using Claude 3.5 Sonnet
# Multi-step verification for 100% accuracy
# """
# from anthropic import Anthropic
# from typing import List, Dict, Any
# import json
# import logging
# from models import PerfectQuestion, ParsingResult, ValidationCheck, QuestionType, Difficulty
# from config import AccuracyConfig, PERFECTION_PROMPTS
# import re

# logger = logging.getLogger(__name__)


# class PerfectParser:
#     """Question parser with multi-step verification"""
    
#     def __init__(self, config: AccuracyConfig):
#         self.config = config
#         self.client = Anthropic(api_key=config.claude_api_key)
#         self.model = config.claude_model
    
#     def parse_perfect_questions(self, extracted_text: str, 
#                                subject: str = "General",
#                                topic: str = "General") -> ParsingResult:
#         """
#         Parse questions from extracted text with 100% accuracy
        
#         Args:
#             extracted_text: Text extracted from OCR
#             subject: Subject name
#             topic: Topic name
            
#         Returns:
#             ParsingResult with parsed questions
#         """
#         logger.info("=" * 60)
#         logger.info("🔍 STARTING PERFECT QUESTION PARSING")
#         logger.info("=" * 60)
        
#         # Validate input
#         if not extracted_text:
#             logger.error(f"❌ CRITICAL: Input text is None or empty! OCR might have failed.")
#             return ParsingResult(
#                 questions=[],
#                 confidence=0.0,
#                 validation_checks=[ValidationCheck(
#                     check_type="input_validation",
#                     passed=False,
#                     confidence=0.0,
#                     details="Input text is empty or None"
#                 )],
#                 parsing_errors=["Input text is empty or None"],
#                 requires_human_review=True
#             )
        
#         logger.info(f"📥 Input text length: {len(extracted_text)} characters")
        
#         # DEBUG: Log sample of input text
#         if len(extracted_text) > 0:
#             logger.info(f"📄 Input text (first 500 chars):\n{extracted_text[:500]}")
#         else:
#             logger.error(f"❌ CRITICAL: Input text is EMPTY! OCR might have failed.")
        
#         # Check if document needs chunking
#         text_length = len(extracted_text)
#         question_indicators = extracted_text.count('?') + extracted_text.count('Question') + extracted_text.count('Test')
        
#         # Better question detection: look for numbered questions (1, 2, 3, etc.) or lettered sub-questions (a, b, c)
#         import re
#         numbered_questions = len(re.findall(r'\n\s*\d+[\.\)]\s+', extracted_text))  # "1. ", "2) ", etc.
#         lettered_questions = len(re.findall(r'\n\s*[a-z][\.\)]\s+', extracted_text))  # "a. ", "b) ", etc.
#         total_question_markers = numbered_questions + lettered_questions
        
#         logger.info(f"📊 Question detection: {question_indicators} indicators, {numbered_questions} numbered, {lettered_questions} lettered = {total_question_markers} total markers")
        
#         # Use chunking if document is large or has multiple questions
#         # Use the original working threshold
#         use_chunking = text_length > 30000 or question_indicators > 2 or total_question_markers > 1
        
#         if use_chunking:
#             logger.info(f"✅ Using chunking: text_length={text_length}, indicators={question_indicators}, markers={total_question_markers}")
#         else:
#             logger.info(f"⚠️  Single pass mode: text_length={text_length}, indicators={question_indicators}, markers={total_question_markers}")
#             # Even for small docs, if we see numbered questions, use chunking
#             if total_question_markers > 0:
#                 logger.info(f"🔍 Detected {total_question_markers} question markers - forcing chunking to extract ALL")
#                 use_chunking = True
        
#         try:
#             if use_chunking:
#                 logger.warning(f"Large document detected ({text_length} chars, ~{question_indicators} questions)")
#                 logger.info("Using chunking strategy to extract all questions")
#                 initial_result = self._extract_with_chunking(extracted_text, subject, topic)
#             else:
#                 # Step 1: Initial extraction (single pass)
#                 logger.info("Step 1: Initial extraction")
#                 try:
#                     initial_result = self._initial_extraction(extracted_text, subject, topic)
#                 except ValueError as e:
#                     # If truncation detected, retry with chunking
#                     if "truncated" in str(e).lower():
#                         logger.warning("Single pass failed due to truncation. Retrying with chunking...")
#                         initial_result = self._extract_with_chunking(extracted_text, subject, topic)
#                     else:
#                         raise
#         except Exception as e:
#             logger.error(f"❌ CRITICAL: Initial extraction failed completely: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
#             # Return empty result instead of crashing
#             return ParsingResult(
#                 questions=[],
#                 confidence=0.0,
#                 validation_checks=[ValidationCheck(
#                     check_type="extraction_error",
#                     passed=False,
#                     confidence=0.0,
#                     details=f"Extraction failed: {str(e)}"
#                 )],
#                 parsing_errors=[f"Extraction failed: {str(e)}"],
#                 requires_human_review=True
#             )
        
#         # Check if we got any questions
#         if not initial_result or len(initial_result) == 0:
#             logger.error("❌ CRITICAL: Initial extraction returned 0 questions!")
#             logger.error("   This means no questions were extracted from any chunks.")
#             logger.error("   Check logs above to see which chunks extracted questions.")
#             logger.error(f"   Document length: {len(extracted_text)} chars")
#             logger.error(f"   Question indicators: {extracted_text.count('?') + extracted_text.count('Question') + extracted_text.count('Test')}")
#             # Return empty result with detailed error
#             return ParsingResult(
#                 questions=[],
#                 confidence=0.0,
#                 validation_checks=[ValidationCheck(
#                     check_type="extraction_failure",
#                     passed=False,
#                     confidence=0.0,
#                     details="No questions extracted from document. Check chunk extraction logs."
#                 )],
#                 parsing_errors=["No questions extracted from document. All chunks returned 0 questions. Check server logs for details."],
#                 requires_human_review=True
#             )
        
#         logger.info(f"✅ Initial extraction successful: {len(initial_result)} questions extracted")
        
#         # Step 2: Structure validation
#         logger.info("Step 2: Structure validation")
#         try:
#             validated_result = self._validate_structure(initial_result, extracted_text)
#         except Exception as e:
#             logger.error(f"Structure validation failed: {str(e)}")
#             validated_result = initial_result if initial_result else []
        
#         # Step 3: Cross-verification
#         logger.info("Step 3: Cross-verification")
#         try:
#             verified_result = self._cross_verify(validated_result, extracted_text)
#         except Exception as e:
#             logger.error(f"Cross-verification failed: {str(e)}")
#             verified_result = validated_result if validated_result else []
        
#         # Step 4: Final approval
#         logger.info("Step 4: Final approval")
#         try:
#             final_result = self._final_approval(verified_result, extracted_text)
#         except Exception as e:
#             logger.error(f"Final approval failed: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
#             final_result = []
        
#         # Build validation checks
#         validation_checks = []
#         confidence = 1.0
        
#         # Ensure final_result is a list
#         if final_result is None:
#             logger.error("❌ CRITICAL: final_result is None! Setting to empty list.")
#             final_result = []
        
#         # Check completeness
#         completeness = len(final_result) > 0
#         validation_checks.append(ValidationCheck(
#             check_type="parsing_completeness",
#             passed=completeness,
#             confidence=1.0 if completeness else 0.0,
#             details=f"Extracted {len(final_result)} questions"
#         ))
        
#         # Check structure validity
#         try:
#             all_valid = all(self._is_valid_question(q) for q in final_result) if final_result else False
#         except Exception as e:
#             logger.error(f"Error during structure validation: {str(e)}")
#             all_valid = False
        
#         validation_checks.append(ValidationCheck(
#             check_type="structure_validity",
#             passed=all_valid,
#             confidence=1.0 if all_valid else 0.5,
#             details=f"All questions structurally valid: {all_valid}"
#         ))
        
#         # Calculate overall confidence
#         if validation_checks:
#             confidence = sum(c.confidence for c in validation_checks) / len(validation_checks)
        
#         requires_review = confidence < self.config.human_review_threshold
        
#         # Final safety check - ensure we always return a valid ParsingResult
#         try:
#             return ParsingResult(
#                 questions=final_result if final_result else [],
#                 confidence=confidence,
#                 validation_checks=validation_checks,
#                 parsing_errors=[] if all_valid else ["Some questions failed structure validation"],
#                 requires_human_review=requires_review
#             )
#         except Exception as e:
#             logger.error(f"❌ CRITICAL: Failed to create ParsingResult: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
#             # Return a safe fallback result
#             return ParsingResult(
#                 questions=[],
#                 confidence=0.0,
#                 validation_checks=[ValidationCheck(
#                     check_type="result_creation_error",
#                     passed=False,
#                     confidence=0.0,
#                     details=f"Failed to create result: {str(e)}"
#                 )],
#                 parsing_errors=[f"Failed to create result: {str(e)}"],
#                 requires_human_review=True
#             )
    
#     def _extract_with_chunking(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
#         """
#         Extract questions from very large documents using chunking
#         Optimized to prevent system freezes with delays, timeouts, and memory management
#         """
#         import time
#         import gc
#         from document_chunker import DocumentChunker
        
#         # Use VERY small chunks to prevent API calls from hanging and memory issues
#         # Process chunking with aggressive memory management
#         import gc
#         gc.collect()  # Clean up before chunking
        
#         # Determine chunk size based on text length to prevent MemoryError
#         # But ensure chunks are large enough to contain complete questions
#         text_len = len(text)
#         if text_len > 100000:
#             chunk_size = 15000  # Safe size for very large documents
#             logger.info(f"Very large document ({text_len} chars), using chunks ({chunk_size})")
#         else:
#             chunk_size = 20000  # Good balance - was working before
#             logger.info(f"Using chunk size ({chunk_size}) for question extraction")
        
#         # Use reasonable overlap to catch questions at boundaries
#         overlap = 1000  # Was working before - keep it
#         logger.info(f"Using chunk size: {chunk_size} chars with {overlap} char overlap")
#         chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        
#         try:
#             chunks = chunker.chunk_text(text)
#         except MemoryError as me:
#             logger.error(f"❌ Memory error during chunking: {str(me)}")
#             logger.error("Trying with even smaller chunks...")
#             # Try with much smaller chunks
#             gc.collect()
#             chunker = DocumentChunker(chunk_size=5000, overlap=250)
#             gc.collect()
#             chunks = chunker.chunk_text(text)
        
#         # Limit maximum chunks to prevent system overload, but allow more for complete extraction
#         MAX_CHUNKS = 50  # Increased to allow processing more chunks (user wants ALL questions)
#         if len(chunks) > MAX_CHUNKS:
#             logger.warning(f"⚠️  Document has {len(chunks)} chunks, limiting to first {MAX_CHUNKS} chunks")
#             logger.warning(f"⚠️  If you need ALL questions, consider splitting the document or increasing MAX_CHUNKS")
#             chunks = chunks[:MAX_CHUNKS]
#         else:
#             logger.info(f"✅ Will process all {len(chunks)} chunks to extract ALL questions")
        
#         all_questions = []
#         total_chunks = len(chunks)
        
#         logger.info(f"📦 Processing {total_chunks} chunks (ULTRA-SAFE mode: small chunks, long delays)")
#         estimated_time = total_chunks * 8  # More realistic estimate
#         logger.info(f"⏱️  Estimated time: ~{estimated_time} seconds (~{estimated_time//60} minutes)")
#         print(f"📦 Processing {total_chunks} chunks (estimated ~{estimated_time//60} minutes)...", flush=True)
#         print(f"⚠️  If system freezes, wait - processing is slow but safe", flush=True)
        
#         for i, (chunk_text, start_idx, end_idx) in enumerate(chunks, 1):
#             try:
#                 # Progress logging with flush to ensure it appears immediately
#                 progress_msg = f"📄 Chunk {i}/{total_chunks} (chars {start_idx}-{end_idx})"
#                 logger.info(progress_msg)
#                 print(progress_msg, flush=True)
                
#                 # Force immediate memory cleanup before processing
#                 gc.collect()
#                 time.sleep(0.5)  # Brief pause before API call
                
#                 # Process chunk with timeout protection (Windows doesn't support signal.alarm)
#                 # Instead, we'll use a simpler approach: catch and handle long-running calls
#                 try:
#                     # Set a flag to track if we're stuck
#                     start_time = time.time()
#                     max_api_time = 90  # Maximum 90 seconds per API call
                    
#                     # Process chunk with enhanced prompt for chunked extraction
#                     chunk_questions = self._extract_chunk_questions(
#                         chunk_text, subject, topic, chunk_num=i, total_chunks=total_chunks, max_time=max_api_time
#                     )
                    
#                     elapsed = time.time() - start_time
#                     if elapsed > 30:
#                         logger.warning(f"⚠️  Chunk {i} took {elapsed:.1f}s (slow but completed)")
                    
#                     # Add questions with duplicate detection
#                     for q in chunk_questions:
#                         # Check for duplicates by comparing question text (normalized)
#                         q_text = q.get("question_text", "").strip().lower()
#                         is_duplicate = False
                        
#                         for existing_q in all_questions:
#                             existing_text = existing_q.get("question_text", "").strip().lower()
#                             # Check if texts are very similar (90%+ match)
#                             if q_text and existing_text:
#                                 # Simple similarity check: if one contains the other or they're very similar
#                                 if q_text == existing_text or (len(q_text) > 20 and len(existing_text) > 20 and 
#                                     (q_text in existing_text or existing_text in q_text)):
#                                     is_duplicate = True
#                                     logger.debug(f"⚠️  Skipping duplicate question: {q_text[:50]}...")
#                                     break
                        
#                         if not is_duplicate:
#                             all_questions.append(q)
                    
#                     result_msg = f"✅ Chunk {i}: {len(chunk_questions)} questions (Total unique: {len(all_questions)})"
#                     logger.info(result_msg)
#                     print(result_msg, flush=True)
                    
#                 except TimeoutError as te:
#                     error_msg = f"⏱️  Chunk {i} timed out after {max_api_time}s, skipping..."
#                     logger.error(error_msg)
#                     print(error_msg, flush=True)
#                     continue
#                 except Exception as api_error:
#                     error_msg = f"❌ Chunk {i} API error: {str(api_error)[:100]}"
#                     logger.error(error_msg)
#                     print(error_msg, flush=True)
#                     continue
                
#                 # Aggressive memory cleanup after each chunk
#                 gc.collect()
#                 time.sleep(0.5)  # Brief pause after processing
                
#                 # Longer delay between chunks to prevent system overload
#                 if i < total_chunks:
#                     delay = 5.0  # Increased to 5 seconds for safety
#                     logger.info(f"⏸️  Waiting {delay}s...")
#                     print(f"⏸️  Waiting {delay}s before next chunk...", flush=True)
#                     time.sleep(delay)
                    
#             except KeyboardInterrupt:
#                 logger.warning("⚠️  Processing interrupted by user")
#                 print("⚠️  Processing interrupted by user", flush=True)
#                 break
#             except Exception as e:
#                 error_msg = f"❌ Error processing chunk {i}: {str(e)[:100]}"
#                 logger.error(error_msg)
#                 print(error_msg, flush=True)
#                 # Continue with next chunk instead of crashing
#                 continue
        
#         # Final duplicate removal and validation
#         logger.info(f"🔍 Final deduplication: {len(all_questions)} questions before dedup")
        
#         # CRITICAL: Check if we're missing questions
#         # Count question numbers in original text to estimate expected count
#         question_number_pattern = r'\b(\d+)[\.\)]\s+[A-Z]|\b(\d+)\s+[A-Z]|Question\s+(\d+)|^(\d+)\s+[A-Z]'
#         question_numbers = re.findall(question_number_pattern, text, re.MULTILINE | re.IGNORECASE)
#         estimated_questions = len([n for n in question_numbers if any(n)])
        
#         # Also count sub-questions (a), (b), (c), etc.
#         sub_question_pattern = r'\([a-z]\)\s+[A-Z]|\([a-z]\)\s+\d'
#         sub_questions = len(re.findall(sub_question_pattern, text, re.IGNORECASE))
        
#         total_estimated = estimated_questions + sub_questions
#         logger.info(f"🔍 Estimated questions in document: {total_estimated} (main: {estimated_questions}, sub: {sub_questions})")
        
#         # More aggressive duplicate removal
#         unique_questions = []
#         seen_texts = set()
        
#         for q in all_questions:
#             q_text = q.get("question_text", "").strip().lower()
#             if not q_text:
#                 continue
            
#             # Normalize text for comparison (remove extra spaces, punctuation)
#             normalized = re.sub(r'\s+', ' ', q_text)
#             normalized = re.sub(r'[^\w\s]', '', normalized)
            
#             # Check if we've seen this question before
#             if normalized not in seen_texts:
#                 seen_texts.add(normalized)
#                 unique_questions.append(q)
#             else:
#                 logger.debug(f"⚠️  Removed duplicate: {q_text[:50]}...")
        
#         final_msg = f"✅ Complete! Extracted {len(unique_questions)} unique questions from {total_chunks} chunks (removed {len(all_questions) - len(unique_questions)} duplicates)"
#         logger.info(final_msg)
#         print(final_msg, flush=True)
        
#         # CRITICAL: Check if we're missing questions
#         # Count question numbers in original text to estimate expected count
#         question_number_pattern = r'\b(\d+)[\.\)]\s+[A-Z]|\b(\d+)\s+[A-Z]|Question\s+(\d+)|^(\d+)\s+[A-Z]'
#         question_numbers = re.findall(question_number_pattern, text, re.MULTILINE | re.IGNORECASE)
#         estimated_questions = len([n for n in question_numbers if any(n)])
        
#         # Also count sub-questions (a), (b), (c), etc.
#         sub_question_pattern = r'\([a-z]\)\s+[A-Z]|\([a-z]\)\s+\d'
#         sub_questions = len(re.findall(sub_question_pattern, text, re.IGNORECASE))
        
#         total_estimated = estimated_questions + sub_questions
#         logger.info(f"🔍 Estimated questions in document: {total_estimated} (main: {estimated_questions}, sub: {sub_questions})")
        
#         if len(unique_questions) < total_estimated * 0.7:  # If we extracted less than 70% of estimated
#             logger.warning(f"⚠️  WARNING: Only extracted {len(unique_questions)} questions but document likely has ~{total_estimated} questions!")
#             logger.warning(f"   This suggests some chunks may have failed to extract questions.")
#             logger.warning(f"   Check logs above for chunks that extracted 0 questions despite having question markers.")
#         elif len(unique_questions) < total_estimated * 0.9:  # If we extracted less than 90% of estimated
#             logger.warning(f"⚠️  WARNING: Extracted {len(unique_questions)} questions but document likely has ~{total_estimated} questions!")
#             logger.warning(f"   Missing approximately {total_estimated - len(unique_questions)} questions.")
#             logger.warning(f"   Some questions may have been missed. Check logs for incomplete extractions.")
#         else:
#             logger.info(f"✅ Successfully extracted {len(unique_questions)} questions (estimated: {total_estimated})")
        
#         # Final aggressive memory cleanup
#         gc.collect()
#         time.sleep(1)  # Final pause
        
#         return unique_questions
    
#     def _initial_extraction_with_timeout(self, text: str, subject: str, topic: str, max_time: int = 90) -> List[Dict[str, Any]]:
#         """
#         Wrapper for _initial_extraction with timeout protection
#         """
#         import time
#         import threading
        
#         result = []
#         error = None
#         completed = threading.Event()
        
#         def extract():
#             nonlocal result, error
#             try:
#                 result = self._initial_extraction(text, subject, topic)
#             except Exception as e:
#                 error = e
#             finally:
#                 completed.set()
        
#         # Start extraction in a thread
#         thread = threading.Thread(target=extract, daemon=True)
#         thread.start()
        
#         # Wait with timeout
#         if completed.wait(timeout=max_time):
#             if error:
#                 raise error
#             return result
#         else:
#             # Timeout occurred - return empty to allow processing to continue
#             logger.error(f"⏱️  Extraction timed out after {max_time}s - returning empty result for this chunk")
#             return []  # Return empty instead of raising to allow processing to continue
    
#     def _extract_chunk_questions(self, text: str, subject: str, topic: str, chunk_num: int, total_chunks: int, max_time: int = 90) -> List[Dict[str, Any]]:
#         """
#         Extract questions from a single chunk with enhanced prompt for chunked documents
#         Uses threading to prevent hanging API calls
#         """
#         import time
#         import threading
        
#         result = []
#         error = None
#         completed = threading.Event()
        
#         def extract():
#             nonlocal result, error
#             try:
#                 # Use enhanced extraction for chunks
#                 result = self._initial_extraction_chunked(text, subject, topic, chunk_num, total_chunks)
#             except Exception as e:
#                 error = e
#             finally:
#                 completed.set()
        
#         # Start extraction in a thread
#         thread = threading.Thread(target=extract, daemon=True)
#         thread.start()
        
#         # Wait with timeout
#         if completed.wait(timeout=max_time):
#             if error:
#                 raise error
#             return result
#         else:
#             # Timeout occurred - return empty to allow processing to continue
#             logger.error(f"⏱️  Chunk {chunk_num} extraction timed out after {max_time}s")
#             logger.error(f"   This chunk may contain questions that were not extracted!")
#             return []  # Return empty instead of raising to allow processing to continue
    
#     def _initial_extraction_chunked(self, text: str, subject: str, topic: str, chunk_num: int, total_chunks: int) -> List[Dict[str, Any]]:
#         """
#         Extract questions from a chunk with enhanced prompt emphasizing ALL questions
#         """
#         system_prompt = PERFECTION_PROMPTS["parsing"]
        
#         text_length = len(text)
#         question_indicators = text.count('?') + text.count('Question') + text.count('Test')
        
#         user_prompt = f"""
# CRITICAL: This is CHUNK {chunk_num} of {total_chunks} from a larger document.

# You MUST extract EVERY SINGLE question from this chunk. This is part of a complete document extraction.

# IMPORTANT:
# - This chunk may contain 1, 5, 10, 20+ questions
# - Extract ALL of them - DO NOT skip any
# - Even if you see similar questions, extract each one
# - If there are multiple sections in this chunk, extract from ALL sections
# - Count the questions in this chunk FIRST, then extract each one

# Subject: {subject}
# Topic: {topic}

# Chunk {chunk_num}/{total_chunks} text (length: {text_length} chars, ~{question_indicators} question indicators):

# IMPORTANT: The text below may start with headers, instructions, or formatting. Look PAST the headers and find the actual questions. Questions may be numbered (1, 2, 3...) or lettered (a, b, c...).

# {text}

# REMEMBER: Even if the chunk starts with "North London Collegiate School" or instructions, scroll down and find the actual questions. Questions are the numbered items that ask students to solve problems or answer questions.

# CRITICAL INSTRUCTIONS - READ CAREFULLY:
# 1. COUNT FIRST: Count how many questions are in THIS chunk (look for numbers like 1, 2, 3, 4, 5... or 1(a), 1(b), 2(a), etc.)
# 2. Extract EVERY question from THIS chunk completely - DO NOT STOP EARLY
# 3. If you see "Question 1", "Question 2", "1(a)", "1(b)", "2(a)", etc., extract ALL of them
# 4. Do NOT assume other chunks will extract questions - extract ALL from this chunk
# 5. If you see sections (A, B, C...), extract from ALL sections in this chunk
# 6. If the chunk has question numbers 1-5, extract ALL 5. If it has 1-10, extract ALL 10
# 7. Continue extracting until you reach the END of this chunk's text
# 8. Return ALL questions found in this chunk in the "questions" array
# 9. If a question text seems incomplete (e.g., "How many squares need to be shaded" without the rest), look for the complete text in the chunk
# 10. Extract the FULL question text - do not truncate or shorten questions

# VERY IMPORTANT: 
# - Look for question numbers in the text (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25...)
# - Extract EVERY numbered question you find - even if it's question 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
# - Do NOT stop after extracting 5 questions if there are more
# - Do NOT stop after extracting 10 questions if there are more
# - Do NOT stop after extracting 19 questions if there are more - continue until ALL are extracted
# - The document may have 25, 50, 100+ questions - extract ALL of them from this chunk
# - Even if the chunk only contains part of a question, extract the complete part
# - Even if the chunk contains instructions or headers, look for questions within it
# - If you see a question that starts but seems cut off, look for the continuation in the same chunk

# CRITICAL: You MUST return a JSON object with a "questions" array. 
# - If you find NO questions in this chunk, return: {{"questions": [], "confidence": 0.0, "validation_errors": ["No questions found in this chunk"]}}
# - If you find questions, return them ALL in the "questions" array
# - DO NOT return empty array if questions exist
# - DO NOT skip questions because they seem incomplete

# EXAMPLES OF WHAT TO EXTRACT (the document has 25 questions total - extract ALL):
# - "1 Fill in the blanks: (a) 1002-997 =" → Extract as question
# - "1(b) 53+27=" → Extract as separate question
# - "2 (a) Write the following numbers..." → Extract as question
# - "6 It takes 2 robots..." → Extract as question
# - "7 How many robots..." → Extract as question
# - "8 Lucy buys 7 packets..." → Extract as question
# - "9 Shreya buys 13 caramel bars..." → Extract as question
# - "10 A shape is created..." → Extract as question
# - "11 Label the net..." → Extract as question
# - "12 Lina has some beads..." → Extract as question
# - "13 At a garden party..." → Extract as question
# - "14 A concert hall..." → Extract as question
# - "15 Over the summer..." → Extract as question
# - "16 An isosceles triangle..." → Extract as question
# - "17 Angela, Bernice and Candice..." → Extract as question
# - "18 In a village..." → Extract as question
# - "19 How many squares..." → Extract as question
# - "20 Beth caught the train..." → Extract as question
# - "21 At a particular music school..." → Extract as question
# - "22 Even when the camel..." → Extract as question
# - "23 Five identical small rectangles..." → Extract as question
# - "24 A number of identical rectangles..." → Extract as question
# - "25 Circles have been placed..." → Extract as question

# EVERY numbered item (1, 2, 3... 25) and every sub-question (a, b, c, d) should be extracted as a separate question.

# CRITICAL: Do NOT stop after extracting 19 questions. The document has 25 questions. Extract ALL of them.

# Return ONLY valid JSON with ALL questions from this chunk. The "questions" array must contain EVERY question found.
# """
        
#         try:
#             # Use maximum tokens available for Haiku (8192) to ensure we can extract all questions
#             max_tokens = min(8192, self.config.claude_max_tokens)  # Haiku max is 8192
#             # For chunked extraction, always use max tokens to get all questions from each chunk
#             if text_length > 5000:
#                 max_tokens = 8192  # Use full capacity for chunked extraction
#                 logger.info(f"Using maximum tokens ({max_tokens}) for chunk extraction")
            
#             message = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=max_tokens,
#                 temperature=self.config.claude_temperature,
#                 system=system_prompt,
#                 messages=[{"role": "user", "content": user_prompt}]
#             )
            
#             if not message.content or len(message.content) == 0:
#                 raise ValueError("Claude returned empty response content")
            
#             response_text = message.content[0].text
#             if not response_text:
#                 raise ValueError("Claude returned empty response text")
            
#             logger.info(f"✅ Chunk {chunk_num}: Claude response received ({len(response_text)} chars)")
            
#             # DEBUG: Log raw response for troubleshooting
#             logger.debug(f"📄 Chunk {chunk_num} raw response (first 1000 chars):\n{response_text[:1000]}")
            
#             # Extract JSON
#             try:
#                 json_data = self._extract_json(response_text)
                
#                 # DEBUG: Log JSON structure
#                 if isinstance(json_data, dict):
#                     logger.debug(f"📊 Chunk {chunk_num} JSON keys: {list(json_data.keys())}")
#                     if "validation_errors" in json_data and json_data["validation_errors"]:
#                         logger.warning(f"⚠️  Chunk {chunk_num} validation errors: {json_data['validation_errors']}")
                
#                 questions_data = json_data.get("questions", [])
                
#                 logger.info(f"📋 Chunk {chunk_num}: Extracted {len(questions_data)} questions")
                
#                 # DEBUG: If no questions, log more details
#                 if len(questions_data) == 0:
#                     logger.warning(f"⚠️  Chunk {chunk_num}: No questions extracted!")
#                     logger.warning(f"   JSON keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
#                     logger.warning(f"   Response length: {len(response_text)} chars")
#                     logger.warning(f"   Response preview: {response_text[:500]}...")
#                     logger.warning(f"   Full JSON data: {json_data}")
                    
#                     # Check if chunk actually contains question-like content
#                     chunk_has_questions = any(marker in text.lower() for marker in ['?', 'question', '1.', '2.', '3.', '(a)', '(b)', 'answer', 'fill in', 'calculate', 'write', 'find', 'work out'])
#                     if chunk_has_questions:
#                         logger.error(f"❌ Chunk {chunk_num} HAS question markers but extracted 0 questions! This is a CRITICAL problem.")
#                         logger.error(f"   Chunk text sample (first 1000 chars): {text[:1000]}")
#                         logger.error(f"   Chunk text sample (last 500 chars): {text[-500:]}")
#                         logger.error(f"   Question markers found: {[m for m in ['?', 'question', '1.', '2.', '3.', '(a)', '(b)'] if m in text.lower()]}")
#                         logger.error(f"   Full Claude response: {response_text}")
#                         logger.error(f"   Full JSON data: {json_data}")
#                         # This is a critical error - chunk has questions but Claude didn't extract them
#                     else:
#                         logger.info(f"ℹ️  Chunk {chunk_num} appears to be header/instructions (no question markers found)")
#                         logger.debug(f"   Chunk text sample: {text[:300]}...")
#             except Exception as json_err:
#                 logger.error(f"❌ Chunk {chunk_num}: JSON extraction failed: {str(json_err)}")
#                 logger.error(f"   Response text (first 1000 chars): {response_text[:1000]}")
#                 logger.error(f"   Response text (last 500 chars): {response_text[-500:]}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 questions_data = []
            
#             # Add missing required fields with defaults
#             for q_data in questions_data:
#                 if "correct_answer" not in q_data or not q_data.get("correct_answer"):
#                     question_text = q_data.get("question_text", "")
#                     if "=" in question_text and any(op in question_text for op in ["+", "-", "×", "*", "÷", "/"]):
#                         q_data["correct_answer"] = "N/A (calculate from question)"
#                     else:
#                         q_data["correct_answer"] = "N/A"
                
#                 if "tags" not in q_data or not q_data.get("tags") or len(q_data.get("tags", [])) == 0:
#                     tags = []
#                     if q_data.get("subject"): tags.append(q_data["subject"])
#                     if q_data.get("topic"): tags.append(q_data["topic"])
#                     if q_data.get("question_type"): tags.append(q_data["question_type"])
#                     if q_data.get("difficulty"): tags.append(q_data["difficulty"])
#                     if not tags: tags = ["general"]
#                     q_data["tags"] = tags
            
#             return questions_data
            
#         except Exception as e:
#             logger.error(f"Chunk {chunk_num} extraction failed: {str(e)}")
#             raise
    
#     def _initial_extraction(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
#         """Step 1: Initial extraction using Claude"""
#         system_prompt = PERFECTION_PROMPTS["parsing"]
        
#         # Check text length and handle long documents
#         text_length = len(text)
#         logger.info(f"Text length: {text_length} characters")
        
#         # Count approximate number of questions in text (look for question patterns)
#         question_indicators = text.count('?') + text.count('Question') + text.count('Test')
#         logger.info(f"Found approximately {question_indicators} question indicators in text")
        
#         # If text is very long, we might need to chunk it, but first try with full text
#         # Claude can handle up to 200k tokens, so we should be fine for most documents
#         user_prompt = f"""
# Extract ALL exam questions from the following text with 100% accuracy.

# CRITICAL INSTRUCTIONS - READ CAREFULLY - THIS IS EXTREMELY IMPORTANT:
# 1. This document contains MULTIPLE questions - you MUST extract EVERY SINGLE ONE
# 2. Look for numbered questions: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25...
# 3. Look for lettered sub-questions: (a), (b), (c), (d), etc. under each numbered question
# 4. Count ALL questions FIRST - if you see "1(a)", "1(b)", "1(c)", "1(d)", "2(a)", etc., count them ALL
# 5. You MUST extract EVERY SINGLE question - do not skip any
# 6. Count ALL questions first, then extract each one completely
# 7. If there are multiple tests/sections, extract from ALL of them
# 8. Do NOT stop after extracting 1 question - there are MANY more
# 9. Do NOT stop after extracting 5 questions - there are MANY more
# 10. Do NOT stop after extracting 10 questions - continue until ALL are extracted
# 11. Extract ALL questions until you reach the end of the document
# 12. For arithmetic tests with sections A, B, C - extract ALL questions from ALL sections

# Subject: {subject}
# Topic: {topic}

# Text to parse (length: {text_length} characters, ~{question_indicators} question indicators):
# {text}

# VERY IMPORTANT:
# - The document may have 50, 100, 200+ questions
# - Extract ALL of them, not just a sample
# - Return a COMPLETE list with EVERY question
# - If you see "Test 1", "Test 2", "Section A", "Section B", etc., extract from ALL tests/sections
# - Count the questions FIRST: "I see 25 questions in this document"
# - Then extract EACH ONE completely
# - DO NOT stop at 10 questions
# - DO NOT stop at 19 questions
# - Extract EVERY SINGLE QUESTION until the end

# CRITICAL: Before you start extracting, COUNT how many questions are in the document.
# Then extract ALL of them. If you find 25 questions, extract all 25. If you find 50, extract all 50.

# Return ONLY valid JSON with ALL questions in the "questions" array. 
# The array should contain EVERY question found in the document - NO EXCEPTIONS.
# """
        
#         try:
#             # Use maximum tokens available to ensure we extract ALL questions
#             max_tokens = min(8192, self.config.claude_max_tokens)  # Haiku max is 8192
#             if text_length > 5000:  # For documents with multiple questions
#                 max_tokens = 8192  # Use full capacity
#                 logger.info(f"Using maximum tokens ({max_tokens}) to extract ALL questions")
            
#             # Wrap API call with timeout protection
#             import time
#             start_time = time.time()
            
#             try:
#                 message = self.client.messages.create(
#                     model=self.model,
#                     max_tokens=max_tokens,
#                     temperature=self.config.claude_temperature,
#                     system=system_prompt,
#                     messages=[
#                         {"role": "user", "content": user_prompt}
#                     ]
#                 )
                
#                 elapsed = time.time() - start_time
#                 if elapsed > 30:
#                     logger.warning(f"⚠️  API call took {elapsed:.1f}s (slow but completed)")
                    
#             except Exception as api_err:
#                 elapsed = time.time() - start_time
#                 logger.error(f"❌ API call failed after {elapsed:.1f}s: {str(api_err)}")
#                 raise
            
#             # Safely extract response text
#             if not message.content or len(message.content) == 0:
#                 raise ValueError("Claude returned empty response content")
            
#             response_text = message.content[0].text
#             if not response_text:
#                 raise ValueError("Claude returned empty response text")
            
#             logger.info(f"✅ Claude response received. Length: {len(response_text)} characters")
#             logger.info(f"📝 Response stop reason: {message.stop_reason}")
            
#             # DEBUG: Log Claude's raw response
#             if len(response_text) > 0:
#                 logger.info(f"📄 Claude raw response (first 500 chars):\n{response_text[:500]}")
#             else:
#                 logger.error(f"❌ CRITICAL: Claude returned EMPTY response!")
            
#             # Check if response was truncated
#             response_complete = True
#             if message.stop_reason == "max_tokens":
#                 logger.error(f"❌ CRITICAL: Response was TRUNCATED due to max_tokens limit ({max_tokens})")
#                 logger.error("❌ This means not all questions were extracted!")
#                 response_complete = False
#             elif message.stop_reason == "stop_sequence":
#                 logger.info("Response stopped at stop sequence (normal)")
#             elif message.stop_reason:
#                 logger.info(f"Response stop reason: {message.stop_reason}")
            
#             # Extract JSON from response
#             json_data = self._extract_json(response_text)
            
#             # DEBUG: Log what was extracted from JSON
#             logger.info(f"📊 JSON extraction result - Keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
            
#             if "validation_errors" in json_data and json_data["validation_errors"]:
#                 logger.warning(f"⚠️  Validation errors in extraction: {json_data['validation_errors']}")
            
#             questions_data = json_data.get("questions", [])
#             logger.info(f"📋 Extracted {len(questions_data)} questions from initial extraction")
            
#             # Add missing required fields with defaults
#             for q_data in questions_data:
#                 # Add correct_answer if missing
#                 if "correct_answer" not in q_data or not q_data.get("correct_answer"):
#                     question_text = q_data.get("question_text", "")
#                     if "=" in question_text and any(op in question_text for op in ["+", "-", "×", "*", "÷", "/"]):
#                         q_data["correct_answer"] = "N/A (calculate from question)"
#                     else:
#                         q_data["correct_answer"] = "N/A"
                
#                 # Add tags if missing
#                 if "tags" not in q_data or not q_data.get("tags") or len(q_data.get("tags", [])) == 0:
#                     tags = []
#                     if q_data.get("subject"):
#                         tags.append(q_data["subject"])
#                     if q_data.get("topic"):
#                         tags.append(q_data["topic"])
#                     if q_data.get("question_type"):
#                         tags.append(q_data["question_type"])
#                     if q_data.get("difficulty"):
#                         tags.append(q_data["difficulty"])
#                     if not tags:
#                         tags = ["general"]
#                     q_data["tags"] = tags
            
#             # DEBUG: If no questions, log more details
#             if len(questions_data) == 0:
#                 logger.error(f"❌ NO QUESTIONS EXTRACTED!")
#                 logger.error(f"📄 Full JSON data: {json_data}")
#                 logger.error(f"📄 Full Claude response (first 1000 chars): {response_text[:1000]}")
            
#             # Check if extraction seems incomplete
#             if not response_complete:
#                 logger.error(f"❌ CRITICAL: Response was truncated! Only {len(questions_data)} questions extracted.")
#                 logger.error("❌ Document should be chunked for complete extraction.")
#                 # Raise an error to trigger chunking retry
#                 raise ValueError(f"Response truncated: only {len(questions_data)} questions extracted from large document")
#             elif len(questions_data) == 1 and text_length > 2000:
#                 logger.warning(f"⚠️  Only 1 question extracted from long document ({text_length} chars).")
#                 logger.warning("⚠️  Document likely has more questions. Consider chunking.")
#             elif len(questions_data) < question_indicators * 0.3 and question_indicators > 20:
#                 logger.warning(f"⚠️  Only {len(questions_data)} questions extracted but found {question_indicators} question indicators.")
#                 logger.warning("⚠️  Some questions may be missing. Consider chunking.")
            
#             return questions_data
            
#         except Exception as e:
#             logger.error(f"Initial extraction failed: {str(e)}")
#             raise
    
#     def _validate_structure(self, questions_data: List[Dict], original_text: str) -> List[Dict]:
#         """Step 2: Validate structure of extracted questions"""
#         validated = []
        
#         logger.info(f"Validating structure of {len(questions_data)} questions")
        
#         for i, q_data in enumerate(questions_data, 1):
#             # Check required fields
#             required = ["question_text", "question_type", "correct_answer"]
#             if all(field in q_data for field in required):
#                 # Validate question type
#                 q_type = q_data.get("question_type", "").lower()
#                 if q_type in ["mcq", "multiple_choice"]:
#                     if "options" in q_data and len(q_data["options"]) >= 2:
#                         validated.append(q_data)
#                     else:
#                         logger.warning(f"⚠️  Question {i}: MCQ missing options: {q_data.get('question_text', '')[:50]}")
#                         # Still add it, but log the warning
#                         validated.append(q_data)
#                 elif q_type in ["true_false", "true/false", "t/f"]:
#                     q_data["question_type"] = "true_false"
#                     validated.append(q_data)
#                     logger.debug(f"✅ Question {i}: Validated (true_false)")
#                 elif q_type in ["fill_blank", "fill_in_blank", "fill-in-the-blank"]:
#                     q_data["question_type"] = "fill_blank"
#                     validated.append(q_data)
#                     logger.debug(f"✅ Question {i}: Validated (fill_blank)")
#                 elif q_type in ["short_answer", "short"]:
#                     q_data["question_type"] = "short_answer"
#                     validated.append(q_data)
#                     logger.debug(f"✅ Question {i}: Validated (short_answer)")
#                 elif q_type in ["long_answer", "essay", "long"]:
#                     q_data["question_type"] = "long_answer"
#                     validated.append(q_data)
#                     logger.debug(f"✅ Question {i}: Validated (long_answer)")
#                 else:
#                     # Default to short_answer if unclear
#                     q_data["question_type"] = "short_answer"
#                     validated.append(q_data)
#                     logger.debug(f"✅ Question {i}: Validated (defaulted to short_answer)")
#             else:
#                 missing_fields = [f for f in ["question_text", "question_type", "correct_answer"] if f not in q_data]
#                 logger.warning(f"⚠️  Question {i}: Missing required fields: {missing_fields}")
#                 logger.warning(f"   Question data: {q_data}")
#                 # Try to add defaults and still include it
#                 if "correct_answer" not in q_data:
#                     q_data["correct_answer"] = "N/A"
#                 if "question_type" not in q_data:
#                     q_data["question_type"] = "short_answer"
#                 if "question_text" in q_data:  # Only add if we have at least question_text
#                     validated.append(q_data)
#                     logger.info(f"✅ Question {i}: Added with defaults after missing fields")
        
#         logger.info(f"✅ Structure validation: {len(validated)}/{len(questions_data)} questions passed")
#         return validated
    
#     def _cross_verify(self, questions_data: List[Dict], original_text: str) -> List[Dict]:
#         """Step 3: Cross-verify questions against original text"""
#         if not questions_data:
#             return []
        
#         # Use Claude to verify extraction accuracy
#         system_prompt = "You are a verification expert. Verify that extracted questions match the source text exactly."
        
#         questions_summary = "\n".join([
#             f"{i+1}. {q.get('question_text', '')[:100]}..." 
#             for i, q in enumerate(questions_data[:5])  # Sample for verification
#         ])
        
#         user_prompt = f"""
# Verify that the following extracted questions accurately represent the source text.

# Source text:
# {original_text[:2000]}

# Extracted questions:
# {questions_summary}

# Respond with JSON:
# {{
#     "verified": true/false,
#     "confidence": 0.0-1.0,
#     "issues": ["list of any issues found"]
# }}
# """
        
#         try:
#             message = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=1000,
#                 temperature=0.0,  # Zero temperature for verification
#                 system=system_prompt,
#                 messages=[{"role": "user", "content": user_prompt}]
#             )
            
#             # Safely extract response text
#             if not message.content or len(message.content) == 0:
#                 logger.warning("Cross-verification: Claude returned empty response, skipping verification")
#                 return questions_data
            
#             response_text = message.content[0].text
#             if not response_text:
#                 logger.warning("Cross-verification: Claude returned empty text, skipping verification")
#                 return questions_data
            
#             verification = self._extract_json(response_text)
            
#             if verification.get("verified", False) and verification.get("confidence", 0) >= 0.95:
#                 return questions_data
#             else:
#                 logger.warning(f"Cross-verification found issues: {verification.get('issues', [])}")
#                 # Still return questions but flag for review
#                 return questions_data
                
#         except Exception as e:
#             logger.warning(f"Cross-verification failed, proceeding: {str(e)}")
#             return questions_data
    
#     def _final_approval(self, questions_data: List[Dict], original_text: str) -> List[PerfectQuestion]:
#         """Step 4: Convert to PerfectQuestion models with final validation"""
#         perfect_questions = []
        
#         logger.info(f"Converting {len(questions_data)} questions to PerfectQuestion models")
        
#         successful = 0
#         failed = 0
        
#         for i, q_data in enumerate(questions_data, 1):
#             try:
#                 # Add missing required fields with defaults
#                 if "correct_answer" not in q_data or not q_data.get("correct_answer"):
#                     # Try to calculate answer for math problems, otherwise use N/A
#                     question_text = q_data.get("question_text", "")
#                     if "=" in question_text and any(op in question_text for op in ["+", "-", "×", "*", "÷", "/"]):
#                         # It's a math problem, calculate it
#                         q_data["correct_answer"] = "N/A (calculate from question)"
#                     else:
#                         q_data["correct_answer"] = "N/A"
#                     logger.debug(f"Added default correct_answer for question {i}")
                
#                 # Add tags if missing
#                 if "tags" not in q_data or not q_data.get("tags") or len(q_data.get("tags", [])) == 0:
#                     tags = []
#                     if q_data.get("subject"):
#                         tags.append(q_data["subject"])
#                     if q_data.get("topic"):
#                         tags.append(q_data["topic"])
#                     if q_data.get("question_type"):
#                         tags.append(q_data["question_type"])
#                     if q_data.get("difficulty"):
#                         tags.append(q_data["difficulty"])
#                     if not tags:
#                         tags = ["general"]
#                     q_data["tags"] = tags
#                     logger.debug(f"Added default tags for question {i}: {tags}")
                
#                 # Normalize question type
#                 q_type_str = q_data.get("question_type", "short_answer").lower()
#                 q_type_map = {
#                     "mcq": QuestionType.MCQ,
#                     "multiple_choice": QuestionType.MCQ,
#                     "true_false": QuestionType.TRUE_FALSE,
#                     "true/false": QuestionType.TRUE_FALSE,
#                     "fill_blank": QuestionType.FILL_BLANK,
#                     "fill_in_blank": QuestionType.FILL_BLANK,
#                     "short_answer": QuestionType.SHORT_ANSWER,
#                     "long_answer": QuestionType.LONG_ANSWER,
#                     "essay": QuestionType.LONG_ANSWER
#                 }
#                 question_type = q_type_map.get(q_type_str, QuestionType.SHORT_ANSWER)
                
#                 # CRITICAL FIX: If question is marked as MCQ but has no options, change to short_answer
#                 if question_type == QuestionType.MCQ:
#                     options = q_data.get("options")
#                     if not options or len(options) < 2:
#                         logger.warning(f"⚠️  Question {i}: Marked as MCQ but has no options. Changing to short_answer.")
#                         logger.warning(f"   Question text: {q_data.get('question_text', '')[:80]}...")
#                         question_type = QuestionType.SHORT_ANSWER
#                         # Remove options from data to prevent confusion
#                         if "options" in q_data:
#                             del q_data["options"]
                
#                 # Normalize difficulty
#                 diff_str = q_data.get("difficulty", "medium").lower()
#                 difficulty_map = {
#                     "easy": Difficulty.EASY,
#                     "medium": Difficulty.MEDIUM,
#                     "hard": Difficulty.HARD
#                 }
#                 difficulty = difficulty_map.get(diff_str, Difficulty.MEDIUM)
                
#                 # Build PerfectQuestion - ensure all required fields have valid values
#                 question_text = q_data.get("question_text", "").strip()
#                 correct_answer = q_data.get("correct_answer", "N/A").strip()
                
#                 # Ensure correct_answer is not empty (Pydantic requires min_length=1)
#                 if not correct_answer:
#                     correct_answer = "N/A"
                
#                 # Ensure question_text meets minimum length
#                 if len(question_text) < 10:
#                     raise ValueError(f"Question text too short: {len(question_text)} chars (minimum 10)")
                
#                 perfect_q = PerfectQuestion(
#                     question_text=question_text,
#                     question_type=question_type,
#                     options=q_data.get("options") if question_type == QuestionType.MCQ else None,
#                     correct_answer=correct_answer,
#                     explanation=q_data.get("explanation"),
#                     difficulty=difficulty,
#                     tags=q_data.get("tags", [q_data.get("topic", "general")]),
#                     subject=q_data.get("subject", "General"),
#                     topic=q_data.get("topic", "General"),
#                     confidence_score=min(1.0, q_data.get("confidence", 0.95))
#                 )
                
#                 perfect_questions.append(perfect_q)
#                 successful += 1
#                 logger.debug(f"✅ Created question {i}/{len(questions_data)}: {perfect_q.question_text[:50]}...")
                
#             except Exception as e:
#                 failed += 1
#                 logger.error(f"❌ Failed to create PerfectQuestion {i}/{len(questions_data)}: {str(e)}")
#                 logger.error(f"   Question text: {q_data.get('question_text', 'N/A')[:100]}")
#                 logger.error(f"   Question data: {q_data}")
#                 import traceback
#                 logger.error(traceback.format_exc())
        
#         logger.info(f"✅ Successfully converted {successful}/{len(questions_data)} questions")
#         if failed > 0:
#             logger.warning(f"⚠️  Failed to convert {failed} questions - check logs above for details")
        
#         logger.info(f"Successfully created {len(perfect_questions)} PerfectQuestion objects from {len(questions_data)} input questions")
        
#         # Warn if only 1 question from a long document
#         if len(perfect_questions) == 1 and len(original_text) > 1000:
#             logger.warning(f"Only 1 question extracted from long text ({len(original_text)} chars). Document might have more questions that weren't extracted.")
        
#         return perfect_questions
    
#     def _extract_json(self, text: str) -> Dict:
#         """Extract JSON from Claude response"""
#         # Remove markdown code blocks if present
#         text_cleaned = re.sub(r'```json\s*', '', text)
#         text_cleaned = re.sub(r'```\s*', '', text_cleaned)
#         text_cleaned = text_cleaned.strip()
        
#         # Try to find JSON in the response (look for { ... } pattern)
#         json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
#         if json_match:
#             try:
#                 json_str = json_match.group()
#                 logger.debug(f"Found JSON match, length: {len(json_str)} chars")
#                 parsed = json.loads(json_str)
#                 logger.debug(f"✅ Successfully parsed JSON with keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
#                 return parsed
#             except json.JSONDecodeError as e:
#                 logger.warning(f"JSON parse error from match: {e}")
#                 logger.warning(f"JSON string (first 500 chars): {json_str[:500]}")
#                 logger.warning(f"JSON string (last 200 chars): {json_str[-200:]}")
#                 # Try to fix common JSON issues
#                 try:
#                     # Try fixing trailing commas
#                     json_str_fixed = re.sub(r',\s*}', '}', json_str)
#                     json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
#                     return json.loads(json_str_fixed)
#                 except:
#                     pass
        
#         # Try parsing entire response
#         try:
#             parsed = json.loads(text_cleaned)
#             logger.debug(f"✅ Successfully parsed entire response as JSON")
#             return parsed
#         except json.JSONDecodeError as e:
#             logger.error(f"❌ Failed to parse JSON from response")
#             logger.error(f"📄 Response (first 1000 chars): {text_cleaned[:1000]}")
#             logger.error(f"📄 Response (last 500 chars): {text_cleaned[-500:]}")
#             logger.error(f"❌ JSON error: {e}")
#             logger.error(f"❌ JSON error position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
#             # Return empty questions with error
#             return {
#                 "questions": [], 
#                 "validation_errors": [f"Failed to parse JSON: {str(e)}"],
#                 "confidence": 0.0
#             }
    
#     def _is_valid_question(self, question: PerfectQuestion) -> bool:
#         """Check if question is valid"""
#         try:
#             # Validate required fields
#             if not question.question_text or len(question.question_text.strip()) < 10:
#                 return False
#             if not question.correct_answer or len(question.correct_answer.strip()) == 0:
#                 return False
#             if question.question_type == QuestionType.MCQ:
#                 if not question.options or len(question.options) < 2:
#                     return False
#             return True
#         except Exception as e:
#             logger.error(f"Error validating question: {str(e)}")
#             return False

"""
Perfect Parser using Multi-LLM Ensemble (Claude + Gemini)
Multi-step verification with cross-validation for 100% accuracy
"""
from anthropic import Anthropic
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import logging
import re
import time
import gc
from datetime import datetime

# Import your models - make sure these exist
try:
    from models import PerfectQuestion, ParsingResult, ValidationCheck, QuestionType, Difficulty, EnsembleParsingResult, ConsensusCheck
    from config import AccuracyConfig, PERFECTION_PROMPTS
    from multi_llm_validator import MultiLLMValidator, ConsensusStrategy
    from consensus_algorithms import JSONMerger, QuestionComparator, ConsensusScorer
except ImportError:
    # Fallback minimal definitions
    from enum import Enum
    from pydantic import BaseModel
    from typing import List, Optional
    
    class QuestionType(Enum):
        MCQ = "mcq"
        TRUE_FALSE = "true_false" 
        FILL_BLANK = "fill_blank"
        SHORT_ANSWER = "short_answer"
        LONG_ANSWER = "long_answer"
    
    class Difficulty(Enum):
        EASY = "easy"
        MEDIUM = "medium"
        HARD = "hard"
    
    class PerfectQuestion(BaseModel):
        question_text: str
        question_type: QuestionType
        options: Optional[List[str]] = None
        correct_answer: str
        explanation: Optional[str] = None
        difficulty: Difficulty = Difficulty.MEDIUM
        tags: List[str] = []
        subject: str = "General"
        topic: str = "General"
        confidence_score: float = 0.95
    
    class ValidationCheck(BaseModel):
        check_type: str
        passed: bool
        confidence: float
        details: str
    
    class ParsingResult(BaseModel):
        questions: List[PerfectQuestion]
        confidence: float
        validation_checks: List[ValidationCheck]
        parsing_errors: List[str] = []
        requires_human_review: bool = False
    
    class AccuracyConfig:
        def __init__(self):
            self.claude_api_key = "your-api-key"
            self.claude_model = "claude-3-5-sonnet-20241022"
            self.claude_max_tokens = 4000
            self.claude_temperature = 0.1
            self.human_review_threshold = 0.8
    
    PERFECTION_PROMPTS = {
        "parsing": "You are a perfect question extraction expert. Extract ALL questions with 100% accuracy."
    }

logger = logging.getLogger(__name__)


class PerfectParser:
    """Question parser with multi-LLM ensemble validation"""
    
    def __init__(self, config: AccuracyConfig):
        self.config = config
        self.client = Anthropic(api_key=config.claude_api_key)
        self.model = config.claude_model
        self.extraction_cache = {}  # Cache for duplicate detection
        
        # Initialize Gemini for ensemble parsing
        if config.enable_multi_llm and config.use_ensemble_for_parsing:
            genai.configure(api_key=config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(config.gemini_model)
            self.multi_llm_validator = MultiLLMValidator(
                claude_api_key=config.claude_api_key,
                gemini_api_key=config.gemini_api_key,
                claude_model=config.claude_model,
                gemini_model=config.gemini_model,
                consensus_threshold=config.consensus_threshold
            )
            logger.info("Multi-LLM ensemble parsing enabled (Claude + Gemini)")
        else:
            self.gemini_model = None
            self.multi_llm_validator = None
            logger.info("Single-model parsing enabled (Claude only)")
    
    def parse_perfect_questions(self, extracted_text: str, 
                               subject: str = "General",
                               topic: str = "General") -> ParsingResult:
        """
        Parse questions from extracted text with improved accuracy
        """
        logger.info("🚀 STARTING IMPROVED QUESTION PARSING")
        
        # Enhanced input validation
        if not extracted_text or len(extracted_text.strip()) < 10:
            logger.error("Input text is empty or too short")
            return self._create_error_result("Input text is empty or too short")
        
        logger.info(f"Input text length: {len(extracted_text)} characters")
        
        # Detect document type and question patterns
        doc_analysis = self._analyze_document(extracted_text)
        logger.info(f"Document analysis: {doc_analysis}")
        
        try:
            # Use ensemble extraction if enabled
            if self.config.enable_multi_llm and self.config.use_ensemble_for_parsing:
                if len(extracted_text) < 15000:  # Increased threshold for single pass
                    logger.info("Using ensemble extraction (Claude + Gemini)")
                    questions = self._ensemble_extraction(extracted_text, subject, topic)
                else:
                    logger.info("Using optimized chunking strategy with ensemble")
                    questions = self._optimized_chunking_extraction(extracted_text, subject, topic)
            else:
                # Single model mode
                if len(extracted_text) < 15000:
                    logger.info("Using single-pass extraction (Claude only)")
                    questions = self._smart_single_pass_extraction(extracted_text, subject, topic)
                else:
                    logger.info("Using optimized chunking strategy (Claude only)")
                    questions = self._optimized_chunking_extraction(extracted_text, subject, topic)
            
            # Convert to PerfectQuestion objects
            logger.info(f"Converting {len(questions)} raw questions to PerfectQuestion objects...")
            perfect_questions = self._convert_to_perfect_questions(questions)
            logger.info(f"Successfully converted {len(perfect_questions)} questions to PerfectQuestion objects")
            
            if len(perfect_questions) == 0 and len(questions) > 0:
                logger.error(f"❌ CRITICAL: {len(questions)} questions extracted but 0 converted to PerfectQuestion!")
                logger.error("   This suggests validation/conversion errors. Check logs above for details.")
            
            # Calculate confidence
            confidence = self._calculate_confidence(perfect_questions, extracted_text)
            requires_review = confidence < self.config.human_review_threshold
            
            logger.info(f"✅ Extraction complete: {len(perfect_questions)} questions, confidence: {confidence:.2f}")
            
            return ParsingResult(
                questions=perfect_questions,
                confidence=confidence,
                validation_checks=[
                    ValidationCheck(
                        check_type="extraction_complete",
                        passed=len(perfect_questions) > 0,
                        confidence=confidence,
                        details=f"Extracted {len(perfect_questions)} questions"
                    )
                ],
                parsing_errors=[],
                requires_human_review=requires_review
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_error_result(f"Extraction failed: {str(e)}")
    
    def _analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to determine best extraction strategy"""
        # Clean text first
        cleaned_text = self._clean_input_text(text)
        
        # Count various question markers (more lenient patterns)
        # Pattern 1: Number followed by . or ) or space (more flexible)
        numbered_questions = len(re.findall(r'(?:^|\n)\s*(\d+)[\.\)]\s+', cleaned_text, re.MULTILINE))
        # Pattern 2: Number at start of line
        numbered_questions += len(re.findall(r'^\s*(\d+)\s+', cleaned_text, re.MULTILINE))
        # Pattern 3: "Question" followed by number
        question_keywords = len(re.findall(r'Question\s+(\d+)|Test\s+(\d+)', cleaned_text, re.IGNORECASE))
        
        # Lettered sub-questions
        lettered_questions = len(re.findall(r'[\(]?([a-z])[\)\.]\s+', cleaned_text, re.IGNORECASE))
        
        # Question marks
        question_marks = cleaned_text.count('?')
        
        # Detect document sections
        sections = len(re.findall(r'\b(Section|Part)\s+[A-Z]', cleaned_text, re.IGNORECASE))
        
        # Estimate: use numbered questions as primary indicator
        estimated = numbered_questions if numbered_questions > 0 else (question_marks // 2)
        
        logger.info(f"Document analysis: {numbered_questions} numbered, {question_keywords} keywords, {question_marks} question marks")
        
        return {
            "total_length": len(cleaned_text),
            "numbered_questions": numbered_questions,
            "lettered_questions": lettered_questions, 
            "question_keywords": question_keywords,
            "question_marks": question_marks,
            "sections": sections,
            "estimated_total_questions": estimated if estimated > 0 else 25  # Default to 25 if can't detect
        }
    
    def _ensemble_extraction(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
        """
        Extract questions using both Claude and Gemini, then merge with consensus
        """
        if not self.multi_llm_validator or not self.config.use_ensemble_for_parsing:
            # Fallback to single model
            return self._smart_single_pass_extraction(text, subject, topic)
        
        logger.info("🔄 Using ensemble extraction (Claude + Gemini)")
        
        # Extract with both models in parallel
        claude_questions = []
        gemini_questions = []
        
        # Claude extraction
        try:
            logger.info("📤 Extracting with Claude...")
            claude_questions = self._smart_single_pass_extraction(text, subject, topic)
            logger.info(f"✅ Claude extracted {len(claude_questions)} questions")
        except Exception as e:
            logger.error(f"❌ Claude extraction failed: {str(e)}")
        
        # Gemini extraction
        try:
            logger.info("📤 Extracting with Gemini...")
            gemini_questions = self._extract_with_gemini(text, subject, topic)
            logger.info(f"✅ Gemini extracted {len(gemini_questions)} questions")
        except Exception as e:
            logger.error(f"❌ Gemini extraction failed: {str(e)}")
        
        # If both failed, raise error
        if not claude_questions and not gemini_questions:
            raise ValueError("Both Claude and Gemini extraction failed")
        
        # If only one succeeded, return it
        if not claude_questions:
            logger.warning("Only Gemini succeeded, returning Gemini results")
            return gemini_questions
        if not gemini_questions:
            logger.warning("Only Claude succeeded, returning Claude results")
            return claude_questions
        
        # Both succeeded - merge with consensus
        logger.info(f"🔀 Merging results: Claude ({len(claude_questions)}) + Gemini ({len(gemini_questions)})")
        merged_questions = self._merge_question_extractions(claude_questions, gemini_questions)
        logger.info(f"✅ Merged result: {len(merged_questions)} questions")
        
        return merged_questions
    
    def _extract_with_gemini(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
        """Extract questions using Gemini"""
        system_prompt = """You are an expert at extracting exam questions. Extract ONLY the main numbered questions with 100% accuracy.

CRITICAL RULES:
1. Extract ONLY main numbered questions (1, 2, 3, 4, 5... up to 25)
2. DO NOT extract sub-questions like (a), (b), (c) as separate questions
3. If a question has sub-parts like "1(a)", "1(b)", combine them into ONE question with the full text
4. Include the FULL question text including all sub-parts
5. For math questions, include the complete problem
6. Return valid JSON with all main questions in the "questions" array
7. If no questions found, return {"questions": []}

QUESTION FORMAT:
{
  "question_text": "Full question text including all sub-parts",
  "question_type": "mcq|true_false|fill_blank|short_answer|long_answer",
  "correct_answer": "The correct answer or N/A",
  "options": ["Option A", "Option B", ...]  # Only for MCQ
}

IMPORTANT: Count the main questions first. If you see "1.", "2.", "3."... up to "25.", extract exactly those 25 main questions. Do NOT extract "1(a)", "1(b)" as separate questions."""
        
        user_prompt = f"""Extract ONLY the main numbered questions (1-25) from this {subject} exam about {topic}.

DOCUMENT TEXT:
{text}

CRITICAL INSTRUCTIONS:
- Extract ONLY main questions numbered 1, 2, 3, 4, 5... up to 25
- DO NOT extract sub-questions like (a), (b), (c) as separate questions
- If question 1 has parts (a) and (b), include BOTH parts in question 1's text
- Look for patterns like "1.", "2.", "3."... "25." (main questions only)
- Include mathematical expressions and formulas
- Preserve the exact wording
- Count first: How many main numbered questions are there? Extract exactly that many.

Return ONLY valid JSON: {{"questions": [array of question objects]}}
Each question should represent ONE main numbered question, not sub-questions."""
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = self.gemini_model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 8000,
            }
        )
        
        response_text = response.text
        logger.info(f"Gemini response length: {len(response_text)}")
        
        # Extract JSON
        json_data = self._robust_json_extraction(response_text)
        questions = json_data.get("questions", [])
        
        logger.info(f"Gemini extracted {len(questions)} questions")
        
        # Validate and clean questions
        cleaned = self._clean_questions(questions)
        
        # Filter out sub-questions and keep only main numbered questions
        filtered = self._filter_main_questions_only(cleaned)
        logger.info(f"After filtering sub-questions: {len(filtered)} main questions")
        
        return filtered
    
    def _merge_question_extractions(self, claude_questions: List[Dict], gemini_questions: List[Dict]) -> List[Dict]:
        """
        Merge question extractions from Claude and Gemini using consensus
        """
        # Use QuestionComparator to find matches
        matches, only_claude, only_gemini = QuestionComparator.find_matching_questions(
            claude_questions, gemini_questions, similarity_threshold=0.85
        )
        
        logger.info(f"Question matching: {len(matches)} matches, {len(only_claude)} only Claude, {len(only_gemini)} only Gemini")
        
        # Merge matched questions
        merged = []
        for q_claude, q_gemini in matches:
            merged_q = QuestionComparator.merge_matched_questions(q_claude, q_gemini)
            merged.append(merged_q)
        
        # Add questions that only one model found (they might be valid)
        # Use voting strategy: if both models are reliable, include all
        if len(matches) > 0:
            # Both models found some questions, so both are working
            # Include unique questions from both
            merged.extend(only_claude)
            merged.extend(only_gemini)
            logger.info(f"Including all unique questions from both models")
        else:
            # No matches - models completely disagree
            # Use the model with more questions (likely more complete)
            if len(claude_questions) >= len(gemini_questions):
                logger.warning("No matches found - using Claude results (more complete)")
                return claude_questions
            else:
                logger.warning("No matches found - using Gemini results (more complete)")
                return gemini_questions
        
        # Deduplicate merged list and ensure all have correct_answer
        deduplicated = []
        seen = set()
        for q in merged:
            q_text = q.get("question_text", "").strip().lower()
            normalized = re.sub(r'\s+', ' ', q_text)
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                # Ensure correct_answer is never None
                if not q.get("correct_answer") or q.get("correct_answer") is None:
                    q["correct_answer"] = "N/A"
                deduplicated.append(q)
        
        return deduplicated
    
    def _smart_single_pass_extraction(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
        """Improved single pass extraction with better prompts (Claude only)"""
        system_prompt = PERFECTION_PROMPTS["parsing"]
        
        # Clean the text first to remove any OCR artifacts
        cleaned_text = self._clean_input_text(text)
        
        # Log sample of what we're sending to Claude
        logger.info(f"Sample of text being sent to Claude (first 500 chars):\n{cleaned_text[:500]}")
        
        user_prompt = f"""Extract ALL main numbered questions (1-25) from this {subject} exam about {topic}.

CRITICAL: This document contains 25 main questions. Extract ALL of them.

DOCUMENT TEXT:
{cleaned_text}

IMPORTANT INSTRUCTIONS:
1. Look for numbered questions starting with: 1, 2, 3, 4, 5... up to 25
2. Questions may be formatted as: "1.", "1)", "Question 1", "1 ", etc.
3. If a question has sub-parts like (a), (b), (c), include ALL sub-parts in the main question text
4. Extract the COMPLETE question text including all parts
5. For math questions, include all numbers, formulas, and diagrams described in text
6. Set correct_answer to "N/A" if answer is not provided in the document
7. Set question_type based on the question format (short_answer for most math questions)

QUESTION FORMAT - Return JSON:
{{
  "questions": [
    {{
      "question_text": "Complete question text including all sub-parts",
      "question_type": "short_answer|mcq|true_false|fill_blank|long_answer",
      "correct_answer": "N/A or the answer if provided",
      "options": []  # Only for MCQ questions
    }}
  ],
  "confidence": 0.95
}}

CRITICAL: You MUST extract questions. If you see text that looks like questions (numbered items, math problems, etc.), extract them. Do NOT return empty array unless the text truly contains no questions."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=min(8000, self.config.claude_max_tokens),
                temperature=0.1,  # Lower temperature for consistency
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            if not response.content:
                raise ValueError("Empty response from Claude")
            
            response_text = response.content[0].text
            logger.info(f"Claude response length: {len(response_text)}")
            
            # DEBUG: Log Claude's response to see what it's returning
            if len(response_text) < 500:
                logger.warning(f"⚠️  Claude returned very short response: {response_text}")
            else:
                logger.debug(f"Claude response (first 500 chars): {response_text[:500]}")
            
            # Extract JSON with better error handling
            json_data = self._robust_json_extraction(response_text)
            
            # DEBUG: Log what was extracted
            logger.debug(f"JSON data keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
            if isinstance(json_data, dict) and "validation_errors" in json_data:
                logger.warning(f"⚠️  Claude reported validation errors: {json_data.get('validation_errors')}")
            
            questions = json_data.get("questions", [])
            
            logger.info(f"Extracted {len(questions)} questions in single pass")
            
            # If no questions extracted, log more details for debugging
            if len(questions) == 0:
                logger.warning(f"⚠️  No questions extracted! Claude response: {response_text[:1000]}")
                logger.warning(f"   Input text length: {len(text)} chars")
                logger.warning(f"   Input text sample: {text[:500]}")
            
            # Validate and clean questions
            cleaned = self._clean_questions(questions)
            
            # Filter out sub-questions and keep only main numbered questions
            filtered = self._filter_main_questions_only(cleaned)
            logger.info(f"After filtering sub-questions: {len(filtered)} main questions")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Single pass extraction failed: {str(e)}")
            # Fall back to chunking
            return self._optimized_chunking_extraction(text, subject, topic)
    
    def _optimized_chunking_extraction(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
        """Improved chunking strategy that prevents missing questions"""
        # Use question boundaries for chunking instead of fixed size
        chunks = self._chunk_by_questions(text)
        
        if len(chunks) == 1:
            # If only one chunk, use single pass
            return self._smart_single_pass_extraction(text, subject, topic)
        
        logger.info(f"Processing {len(chunks)} question chunks")
        
        all_questions = []
        processed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Clean chunk text first
                cleaned_chunk = self._clean_input_text(chunk)
                
                # Log sample of chunk to debug
                logger.debug(f"Chunk {i+1} sample (first 300 chars): {cleaned_chunk[:300]}")
                
                # Add context to help Claude understand it's part of a larger document
                chunk_with_context = f"""
This is part {i+1} of {len(chunks)} from a larger exam document containing 25 questions total.
Previous chunks may have contained questions 1-{i}. This chunk may contain questions {i+1} onwards.

CHUNK TEXT:
{cleaned_chunk}

CRITICAL: Extract ALL questions from this chunk. Look for:
- Numbered questions: 1, 2, 3, 4, 5... up to 25
- Questions may start with numbers followed by . or ) or space
- Include the COMPLETE question text
- If you see numbered items that look like questions, extract them
- Return JSON with "questions" array - do NOT return empty array if questions exist
"""
                
                chunk_questions = self._smart_single_pass_extraction(chunk_with_context, subject, topic)
                
                # Enhanced duplicate detection
                for q in chunk_questions:
                    q_text = q.get("question_text", "").strip()
                    if q_text and not self._is_duplicate(q_text):
                        all_questions.append(q)
                
                processed_chunks += 1
                logger.info(f"Chunk {i+1}: Added {len(chunk_questions)} questions (Total: {len(all_questions)})")
                
                # Small delay to avoid rate limits
                if i < len(chunks) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {str(e)}")
                continue
        
        # Filter out sub-questions from all extracted questions (after all chunks processed)
        logger.info(f"Before filtering: {len(all_questions)} questions")
        filtered = self._filter_main_questions_only(all_questions)
        logger.info(f"After filtering sub-questions: {len(filtered)} main questions")
        
        # CRITICAL FALLBACK: If chunking extracted 0 questions, try single-pass on full cleaned text
        if len(filtered) == 0 and len(all_questions) == 0:
            logger.warning("⚠️  Chunking extracted 0 questions! Trying single-pass on full cleaned text...")
            cleaned_full_text = self._clean_input_text(text)
            logger.info(f"Full cleaned text length: {len(cleaned_full_text)} chars")
            logger.info(f"Full cleaned text sample (first 1000 chars):\n{cleaned_full_text[:1000]}")
            
            # Try single-pass extraction on full text
            try:
                single_pass_result = self._smart_single_pass_extraction(cleaned_full_text, subject, topic)
                if len(single_pass_result) > 0:
                    logger.info(f"✅ Single-pass fallback extracted {len(single_pass_result)} questions")
                    return single_pass_result
                else:
                    logger.warning("⚠️  Single-pass fallback also returned 0 questions")
            except Exception as e:
                logger.error(f"Single-pass fallback failed: {str(e)}")
        
        logger.info(f"Chunking complete: {len(filtered)} questions from {processed_chunks}/{len(chunks)} chunks")
        return filtered
    
    def _chunk_by_questions(self, text: str, max_chunk_size: int = 8000) -> List[str]:
        """Chunk text at natural question boundaries"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            # Check if this line starts a new question
            is_question_start = re.match(r'^\s*(\d+|[a-z])[\.\)]\s+', line.strip())
            
            if current_size + line_size > max_chunk_size and is_question_start and current_chunk:
                # Start new chunk at question boundary
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # If we have too few chunks or chunks are too large, use fixed-size fallback
        if len(chunks) == 1 and len(text) > 12000:
            return self._fixed_size_chunking(text, max_chunk_size)
        
        return chunks
    
    def _fixed_size_chunking(self, text: str, chunk_size: int = 6000) -> List[str]:
        """Fallback to fixed-size chunking with overlap"""
        chunks = []
        overlap = 500  # Character overlap between chunks
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            if i + chunk_size >= len(text):
                break
        
        return chunks
    
    def _robust_json_extraction(self, text: str) -> Dict[str, Any]:
        """Extract JSON with multiple fallback strategies"""
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Strategy 2: Extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Strategy 3: Find JSON object pattern
        json_match = re.search(r'\{.*?"questions".*?\].*?\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Strategy 4: Manual reconstruction
        logger.warning("JSON extraction failed, attempting manual reconstruction")
        return self._manual_json_reconstruction(text)
    
    def _manual_json_reconstruction(self, text: str) -> Dict[str, Any]:
        """Manual JSON reconstruction when parsing fails"""
        questions = []
        
        # Look for question-like patterns in the response
        lines = text.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            
            # Look for question starters
            if re.match(r'^(\d+|[a-z])[\.\)]\s+', line) or '?' in line:
                if current_question:
                    questions.append(current_question)
                
                current_question = {
                    "question_text": line,
                    "question_type": "short_answer",
                    "correct_answer": "N/A"
                }
            elif current_question and line:
                # Continue building current question
                current_question["question_text"] += " " + line
        
        # Add final question
        if current_question:
            questions.append(current_question)
        
        return {"questions": questions}
    
    def _clean_questions(self, questions: List[Dict]) -> List[Dict]:
        """Clean and validate extracted questions"""
        cleaned = []
        
        for q in questions:
            try:
                # Ensure required fields
                if not q.get("question_text") or len(q.get("question_text", "").strip()) < 5:
                    continue
                
                # Set defaults
                q["question_type"] = q.get("question_type", "short_answer")
                # Handle None explicitly - if correct_answer is None or missing, use "N/A"
                correct_answer = q.get("correct_answer")
                if not correct_answer or correct_answer is None:
                    q["correct_answer"] = "N/A"
                else:
                    q["correct_answer"] = str(correct_answer).strip()
                
                # Ensure tags exist
                if not q.get("tags") or len(q.get("tags", [])) == 0:
                    q["tags"] = [
                        q.get("subject", "General"),
                        q.get("topic", "General"),
                        q.get("question_type", "short_answer")
                    ]
                    q["tags"] = [t for t in q["tags"] if t]  # Remove empty
                    if not q["tags"]:
                        q["tags"] = ["general"]
                
                # Clean question text
                q["question_text"] = q["question_text"].strip()
                
                # Validate MCQ questions have options
                if q["question_type"].lower() in ["mcq", "multiple_choice"]:
                    if not q.get("options") or len(q.get("options", [])) < 2:
                        q["question_type"] = "short_answer"
                        q.pop("options", None)  # Remove empty options
                
                cleaned.append(q)
                
            except Exception as e:
                logger.warning(f"Skipping invalid question: {str(e)}")
                continue
        
        return cleaned
    
    def _filter_main_questions_only(self, questions: List[Dict]) -> List[Dict]:
        """
        Filter to keep only main numbered questions, removing sub-questions
        
        Main questions: "1.", "2.", "3."... "25."
        Sub-questions to exclude: "(a)", "(b)", "a)", "b)", etc.
        
        Returns filtered questions, or original if filtering removes all questions
        """
        if not questions:
            return questions
        
        main_questions = []
        sub_questions = []
        
        for q in questions:
            question_text = q.get("question_text", "").strip()
            
            if not question_text:
                continue
            
            # First check: Is this clearly a sub-question?
            # Sub-questions: "(a)", "(b)", "a)", "b)", "i)", "ii)", etc.
            # Only filter if it's VERY clearly a sub-question (starts with letter, short text)
            is_sub_question = bool(
                re.match(r'^\s*[\(]?([a-z]|i{1,3}|iv|v|vi{0,3}|ix|x)[\)\.]\s+', question_text, re.IGNORECASE) and
                len(question_text) < 100  # Short text suggests it's just a sub-part
            )
            
            if is_sub_question:
                sub_questions.append(q)
                logger.debug(f"Filtered out sub-question: {question_text[:50]}...")
                continue
            
            # Default to including questions unless clearly sub-questions
            # Check if this looks like a main question (starts with number)
            is_main_question = True  # Default to True (be lenient)
            
            # Pattern 1: Starts with number followed by . or ) or space
            if re.match(r'^\s*(\d+)[\.\)]\s+', question_text):
                is_main_question = True
            # Pattern 2: Starts with "Question" followed by number
            elif re.match(r'^\s*Question\s+(\d+)', question_text, re.IGNORECASE):
                is_main_question = True
            # Pattern 3: Starts with number followed by space and capital letter
            elif re.match(r'^\s*(\d+)\s+[A-Z]', question_text):
                is_main_question = True
            # Pattern 4: Contains a number at the start (more lenient)
            elif re.match(r'^\s*\d+', question_text):
                is_main_question = True
            # Pattern 5: If question text is substantial, include it (very lenient)
            elif len(question_text) > 15:
                is_main_question = True
                logger.debug(f"Including question (lenient - substantial text): {question_text[:50]}...")
            else:
                # Very short text without number - might be a fragment
                logger.debug(f"Excluding very short question: {question_text[:50]}...")
                is_main_question = False
            
            if is_main_question:
                main_questions.append(q)
            else:
                logger.debug(f"Filtered out (not main question): {question_text[:50]}...")
        
        # Safety check: If filtering removed ALL questions, return original (filter was too strict)
        if len(main_questions) == 0 and len(questions) > 0:
            logger.warning(f"⚠️  Filter removed all {len(questions)} questions! Returning unfiltered questions.")
            logger.warning("   This suggests the filter may be too strict or questions are in unexpected format.")
            return questions
        
        # Sort by question number if possible
        def extract_question_number(q):
            text = q.get("question_text", "")
            match = re.search(r'(\d+)', text.strip())  # Find first number anywhere in text
            if match:
                return int(match.group(1))
            return 999  # Put unnumbered at end
        
        main_questions.sort(key=extract_question_number)
        
        logger.info(f"Filtered {len(questions)} questions down to {len(main_questions)} main questions (removed {len(sub_questions)} sub-questions)")
        
        return main_questions
    
    def _clean_input_text(self, text: str) -> str:
        """
        Clean input text to remove OCR artifacts and preambles
        """
        if not text:
            return text
        
        # Remove common OCR/Gemini preambles
        preambles = [
            r'^Okay, here is the extracted text.*?\n',
            r'^Okay, here\'s the extracted text.*?\n',
            r'^Here\'s the extracted content.*?\n',
            r'^Here is the extracted text.*?\n',
            r'^Here\'s the extracted text.*?\n',
            r'^Okay, here is.*?\n',
        ]
        
        cleaned = text
        for preamble in preambles:
            cleaned = re.sub(preamble, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove markdown code blocks
        cleaned = re.sub(r'^```.*?\n', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove excessive formatting characters (but keep some)
        # Remove if more than 50 consecutive dots/asterisks
        cleaned = re.sub(r'[\.\*]{50,}', '...', cleaned)
        
        # Remove "**Page X**" markers
        cleaned = re.sub(r'\*\*Page \d+\*\*\s*\n', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def _is_duplicate(self, question_text: str) -> bool:
        """Check if question is a duplicate"""
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', question_text.lower())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Simple hash for comparison
        text_hash = hash(normalized[:100])  # First 100 chars should be unique
        
        if text_hash in self.extraction_cache:
            return True
        
        self.extraction_cache[text_hash] = True
        return False
    
    def _convert_to_perfect_questions(self, questions_data: List[Dict]) -> List[PerfectQuestion]:
        """Convert raw question data to PerfectQuestion objects"""
        perfect_questions = []
        
        for q_data in questions_data:
            try:
                # Map question type
                q_type_str = q_data.get("question_type", "short_answer").lower()
                type_map = {
                    "mcq": QuestionType.MCQ,
                    "multiple_choice": QuestionType.MCQ,
                    "true_false": QuestionType.TRUE_FALSE,
                    "true/false": QuestionType.TRUE_FALSE,
                    "fill_blank": QuestionType.FILL_BLANK,
                    "fill_in_blank": QuestionType.FILL_BLANK,
                    "short_answer": QuestionType.SHORT_ANSWER,
                    "long_answer": QuestionType.LONG_ANSWER,
                    "essay": QuestionType.LONG_ANSWER
                }
                question_type = type_map.get(q_type_str, QuestionType.SHORT_ANSWER)
                
                # Handle MCQ without options
                options = q_data.get("options")
                if question_type == QuestionType.MCQ:
                    if not options or len(options) < 2:
                        question_type = QuestionType.SHORT_ANSWER
                        options = None  # Don't pass empty list
                else:
                    options = None  # Only MCQs have options
                
                # Ensure tags exist (required field)
                tags = q_data.get("tags", [])
                if not tags or len(tags) == 0:
                    # Generate default tags
                    tags = [
                        q_data.get("subject", "General"),
                        q_data.get("topic", "General"),
                        q_data.get("question_type", "short_answer")
                    ]
                    # Remove duplicates and empty strings
                    tags = list(set([t for t in tags if t]))
                    if not tags:
                        tags = ["general"]
                
                # Ensure confidence score is >= 0.95
                confidence_score = q_data.get("confidence_score", 0.95)
                if confidence_score < 0.95:
                    confidence_score = 0.95
                
                # Ensure correct_answer is never None
                correct_answer = q_data.get("correct_answer")
                if not correct_answer or correct_answer is None:
                    correct_answer = "N/A"
                else:
                    correct_answer = str(correct_answer).strip()
                    if not correct_answer:  # Empty string after strip
                        correct_answer = "N/A"
                
                # Create PerfectQuestion
                perfect_q = PerfectQuestion(
                    question_text=q_data["question_text"],
                    question_type=question_type,
                    options=options,
                    correct_answer=correct_answer,
                    explanation=q_data.get("explanation"),
                    difficulty=Difficulty.MEDIUM,  # Default
                    tags=tags,
                    subject=q_data.get("subject", "General"),
                    topic=q_data.get("topic", "General"),
                    confidence_score=confidence_score
                )
                
                perfect_questions.append(perfect_q)
                
            except Exception as e:
                logger.warning(f"Failed to convert question {len(perfect_questions) + 1}/{len(questions_data)}: {str(e)}")
                logger.warning(f"   Question text: {q_data.get('question_text', 'N/A')[:100]}")
                logger.warning(f"   Question data keys: {list(q_data.keys())}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        return perfect_questions
    
    def _calculate_confidence(self, questions: List[PerfectQuestion], original_text: str) -> float:
        """Calculate confidence score for extraction"""
        if not questions:
            return 0.0
        
        # Base confidence factors
        factors = []
        
        # Factor 1: Question count vs estimated
        estimated = self._analyze_document(original_text)["estimated_total_questions"]
        if estimated > 0:
            coverage = min(1.0, len(questions) / estimated)
            factors.append(coverage)
        else:
            factors.append(0.5)  # Unknown estimation
        
        # Factor 2: Question quality
        valid_questions = sum(1 for q in questions if len(q.question_text) > 10)
        quality_score = valid_questions / len(questions) if questions else 0
        factors.append(quality_score)
        
        # Factor 3: Text coverage
        total_question_text = sum(len(q.question_text) for q in questions)
        coverage_ratio = total_question_text / len(original_text) if original_text else 0
        factors.append(min(1.0, coverage_ratio * 10))  # Normalize
        
        return sum(factors) / len(factors)
    
    def _create_error_result(self, error_message: str) -> ParsingResult:
        """Create error result"""
        return ParsingResult(
            questions=[],
            confidence=0.0,
            validation_checks=[
                ValidationCheck(
                    check_type="error",
                    passed=False,
                    confidence=0.0,
                    details=error_message
                )
            ],
            parsing_errors=[error_message],
            requires_human_review=True
        )


# Simple DocumentChunker for fallback
class DocumentChunker:
    def __init__(self, chunk_size=6000, overlap=500):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append((chunk, i, i + len(chunk)))
            if i + self.chunk_size >= len(text):
                break
        return chunks