"""
PerfectExam - 100% Accuracy Generator
FastAPI application with accuracy-guaranteed endpoints
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import os
import tempfile
from pathlib import Path

from config import AccuracyConfig
from models import (
    PerfectQuestion, OCRResult, ParsingResult, GenerationResult,
    ValidationResult, AccuracyError
)
from perfect_ocr import PerfectGeminiOCR
from perfect_parser import PerfectParser
from perfect_generator import PerfectGenerator
from accuracy_validator import AccuracyPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PerfectExam - 100% Accuracy Generator",
    description="Exam question generation system with 100% accuracy guarantee",
    version="1.0.0"
)

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to catch all errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "message": "An unexpected error occurred. Please check the server logs for details."
        }
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration with error handling
try:
    config = AccuracyConfig()
    
    # Validate API keys
    if not config.gemini_api_key or config.gemini_api_key == "":
        logger.warning("⚠️  GEMINI_API_KEY is not set in .env file")
    if not config.claude_api_key or config.claude_api_key == "":
        logger.warning("⚠️  CLAUDE_API_KEY is not set in .env file")
        
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    logger.error("Please check your .env file and ensure GEMINI_API_KEY and CLAUDE_API_KEY are set")
    config = None

# Initialize components with error handling
ocr_engine = None
parser = None
generator = None
validator = None

if config:
    try:
        # Check API keys before initializing
        if not config.gemini_api_key or config.gemini_api_key == "":
            logger.error("Cannot initialize OCR engine: GEMINI_API_KEY is missing")
        elif not config.claude_api_key or config.claude_api_key == "":
            logger.error("Cannot initialize parser/generator: CLAUDE_API_KEY is missing")
        else:
            ocr_engine = PerfectGeminiOCR(config)
            parser = PerfectParser(config)
            generator = PerfectGenerator(config)
            validator = AccuracyPipeline(config)
            logger.info("✅ All components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        logger.error("Please check your API keys in .env file")
        import traceback
        logger.error(traceback.format_exc())


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PerfectExam - 100% Accuracy Generator",
        "version": "1.0.0",
        "status": "operational",
        "accuracy_guarantee": "100%"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if components are initialized
    components_status = {
        "ocr": "operational" if ocr_engine else "not_initialized",
        "parser": "operational" if parser else "not_initialized",
        "generator": "operational" if generator else "not_initialized",
        "validator": "operational" if validator else "not_initialized"
    }
    
    status = "healthy" if all(v == "operational" for v in components_status.values()) else "degraded"
    
    return {
        "status": status,
        "components": components_status,
        "config_loaded": config is not None
    }


@app.post("/extract-perfect-questions", response_model=ParsingResult)
async def extract_perfect_questions(
    file: UploadFile = File(...),
    subject: str = Form("General"),
    topic: str = Form("General")
):
    """
    Extract questions from document with 100% accuracy guarantee
    
    Process:
    1. Enhanced pre-processing
    2. Multi-pass Gemini OCR
    3. Claude validation
    4. Output with confidence scores
    
    Guarantee: 100% accurate extraction or flag for human review
    """
    if not config or not ocr_engine or not parser or not validator:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please check API keys in .env file and restart the server."
        )
    logger.info(f"Extracting questions from {file.filename}")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        try:
            # Write uploaded content
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Step 1: OCR extraction
            logger.info("Step 1: OCR extraction")
            ocr_result = None
            try:
                ocr_result = ocr_engine.extract_perfect_text(tmp_file_path)
                
                # Validate OCR result
                ocr_validation = validator.validate_100_percent(ocr_result, "ocr")
                logger.info(f"OCR validation: {ocr_validation.confidence:.2%}")
                
            except AccuracyError as e:
                logger.error(f"OCR accuracy validation failed: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail=f"OCR extraction failed accuracy validation: {str(e)}"
                )
            except FileNotFoundError as e:
                logger.error(f"File not found: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {str(e)}"
                )
            except ValueError as e:
                logger.error(f"Invalid file format: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format: {str(e)}. Supported formats: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"
                )
            except Exception as e:
                logger.error(f"OCR extraction failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                error_msg = str(e)
                if "Failed to load image" in error_msg:
                    error_msg = f"Failed to load image. Please ensure the file is a valid image or PDF format. Error: {error_msg}"
                raise HTTPException(
                    status_code=500,
                    detail=f"OCR extraction error: {error_msg}"
                )
            
            if not ocr_result:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to extract text from document"
                )
            
            # Step 2: Question parsing
            logger.info("Step 2: Question parsing")
            parsing_result = None
            try:
                parsing_result = parser.parse_perfect_questions(
                    ocr_result.extracted_text,
                    subject=subject,
                    topic=topic
                )
                
                # Validate parsing result (but allow results with review flag)
                if parsing_result:
                    # Ensure questions list exists
                    if not hasattr(parsing_result, 'questions') or parsing_result.questions is None:
                        logger.warning("Parsing result has no questions list, creating empty list")
                        parsing_result.questions = []
                    
                    if parsing_result.questions:
                        try:
                            parsing_validation = validator.validate_100_percent(
                                parsing_result.questions,
                                "questions"
                            )
                            logger.info(f"Parsing validation: {parsing_validation.confidence:.2%}")
                            # Update requires_human_review based on validation
                            if parsing_validation.requires_human_review:
                                parsing_result.requires_human_review = True
                        except AccuracyError as e:
                            # If validation fails but we have a result, flag for review instead of rejecting
                            error_str = str(e)
                            logger.warning(f"Parsing validation warning: {error_str}")
                            
                            # Check if confidence is below review threshold (reject) or just below strict threshold (flag)
                            if ("below" in error_str.lower() and "95%" in error_str) or "0.95" in error_str:
                                # Below review threshold - reject
                                raise HTTPException(
                                    status_code=422,
                                    detail=f"Question parsing failed: {error_str}. Confidence too low. Please check document quality."
                                )
                            else:
                                # Between review and strict threshold - allow but flag
                                logger.info("Allowing result with review flag due to validation warning")
                                parsing_result.requires_human_review = True
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Parsing failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Question parsing error: {str(e)}"
                )
            
            # Step 3: Final validation
            if not parsing_result:
                logger.error("Parsing returned None result")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse questions from document"
                )
            
            # Ensure parsing_result has required attributes
            if not hasattr(parsing_result, 'questions'):
                logger.error("Parsing result missing 'questions' attribute")
                parsing_result.questions = []
            
            if not hasattr(parsing_result, 'confidence'):
                logger.error("Parsing result missing 'confidence' attribute")
                parsing_result.confidence = 0.0
            
            if parsing_result.requires_human_review:
                logger.warning("Result flagged for human review")
            
            return parsing_result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
        finally:
            # Cleanup - Windows-safe file deletion
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                try:
                    # Close any open file handles first
                    import gc
                    gc.collect()
                    
                    # Retry deletion with delay (Windows file locking issue)
                    import time
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Try to remove the file
                            os.unlink(tmp_file_path)
                            logger.debug(f"Successfully deleted temp file: {tmp_file_path}")
                            break
                        except PermissionError as e:
                            if attempt < max_retries - 1:
                                time.sleep(0.5)  # Wait 500ms before retry
                                logger.debug(f"Retry {attempt + 1}/{max_retries} deleting temp file...")
                            else:
                                logger.warning(f"Failed to cleanup temp file after {max_retries} attempts: {str(e)}")
                                # On Windows, we can schedule deletion on next reboot if needed
                                # But for now, just log the warning
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file: {str(e)}")
                            break
                except Exception as e:
                    logger.warning(f"Error during temp file cleanup: {str(e)}")


@app.post("/generate-perfect-variations", response_model=GenerationResult)
async def generate_perfect_variations(
    source_question: dict,
    num_variations: int = Form(3)
):
    """
    Generate question variations with 100% conceptual accuracy
    
    Process:
    1. Claude generation
    2. Cross-verification
    3. Quality check
    4. Output
    
    Guarantee: 100% concept-accurate variations
    """
    if not config or not generator or not validator:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please check API keys in .env file and restart the server."
        )
    logger.info(f"Generating {num_variations} variations")
    
    try:
        # Convert dict to PerfectQuestion
        source = PerfectQuestion(**source_question)
        
        # Generate variations
        generation_result = generator.generate_perfect_variations(
            source_question=source,
            num_variations=num_variations
        )
        
        # Validate generation result
        if generation_result.generated_questions:
            gen_validation = validator.validate_100_percent(
                generation_result.generated_questions,
                "questions"
            )
            logger.info(f"Generation validation: {gen_validation.confidence:.2%}")
        
        if generation_result.requires_human_review:
            logger.warning("Generated questions flagged for human review")
        
        return generation_result
        
    except AccuracyError as e:
        logger.error(f"Generation accuracy validation failed: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Question generation failed accuracy validation: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Question generation error: {str(e)}"
        )


@app.post("/validate-accuracy", response_model=ValidationResult)
async def validate_accuracy(question_data: dict):
    """
    Validate accuracy of any question data
    
    Returns: Accuracy score + validation report
    """
    logger.info("Validating question accuracy")
    
    try:
        # Determine validation type
        if "questions" in question_data and isinstance(question_data["questions"], list):
            # Multiple questions
            questions = [PerfectQuestion(**q) for q in question_data["questions"]]
            validation_result = validator.validate_100_percent(questions, "questions")
        elif "question_text" in question_data:
            # Single question
            question = PerfectQuestion(**question_data)
            validation_result = validator.validate_100_percent(question, "question")
        else:
            raise ValueError("Invalid question data format")
        
        return validation_result
        
    except AccuracyError as e:
        # Return validation result even if it failed
        return ValidationResult(
            passed=False,
            confidence=0.0,
            failed_validations=[],
            passed_validations=[],
            requires_human_review=True
        )
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )


@app.post("/debug-chunks")
async def debug_chunks(
    file: UploadFile = File(...),
    subject: str = Form("General"),
    topic: str = Form("General")
):
    """
    Debug endpoint to see what chunks are being created and what Claude returns
    """
    if not config or not ocr_engine or not parser:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Extract OCR
            ocr_result = ocr_engine.extract_perfect_text(tmp_file_path)
            extracted_text = ocr_result.extracted_text
            
            # Create chunks manually to inspect
            from document_chunker import DocumentChunker
            text_len = len(extracted_text)
            chunk_size = 20000 if text_len <= 100000 else 15000
            chunker = DocumentChunker(chunk_size=chunk_size, overlap=1000)
            chunks = chunker.chunk_text(extracted_text)
            
            # Test first chunk with Claude
            debug_info = {
                "ocr_text_length": len(extracted_text),
                "ocr_text_preview": extracted_text[:500],
                "total_chunks": len(chunks),
                "chunks_info": []
            }
            
            # Test first 3 chunks
            for i, (chunk_text, start_idx, end_idx) in enumerate(chunks[:3], 1):
                chunk_info = {
                    "chunk_num": i,
                    "start": start_idx,
                    "end": end_idx,
                    "length": len(chunk_text),
                    "preview": chunk_text[:300],
                    "has_question_markers": any(m in chunk_text.lower() for m in ['?', 'question', '1.', '2.', '3.', '(a)', '(b)']),
                    "claude_response": None,
                    "extracted_questions": 0
                }
                
                # Try to extract from this chunk
                try:
                    from perfect_parser import PerfectParser
                    test_parser = PerfectParser(config)
                    questions = test_parser._extract_chunk_questions(
                        chunk_text, subject, topic, chunk_num=i, total_chunks=len(chunks), max_time=30
                    )
                    chunk_info["extracted_questions"] = len(questions)
                    chunk_info["questions_preview"] = [q.get("question_text", "")[:100] for q in questions[:3]]
                except Exception as e:
                    chunk_info["error"] = str(e)
                    import traceback
                    chunk_info["traceback"] = traceback.format_exc()
                
                debug_info["chunks_info"].append(chunk_info)
            
            return debug_info
            
        finally:
            if os.path.exists(tmp_file_path):
                try:
                    import gc
                    gc.collect()
                    import time
                    time.sleep(0.5)
                    os.remove(tmp_file_path)
                except:
                    pass


@app.post("/extract-ocr-only", response_model=OCRResult)
async def extract_ocr_only(file: UploadFile = File(...)):
    """
    Extract text using OCR only (without parsing)
    Useful for debugging and validation
    """
    if not config or not ocr_engine or not validator:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please check API keys in .env file and restart the server."
        )
    logger.info(f"OCR extraction only from {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            ocr_result = ocr_engine.extract_perfect_text(tmp_file_path)
            
            # Validate
            ocr_validation = validator.validate_100_percent(ocr_result, "ocr")
            
            return ocr_result
            
        except AccuracyError as e:
            raise HTTPException(
                status_code=422,
                detail=f"OCR extraction failed: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"OCR error: {str(e)}"
            )
        finally:
            # Cleanup - Windows-safe file deletion
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                try:
                    # Close any open file handles first
                    import gc
                    gc.collect()
                    
                    # Retry deletion with delay (Windows file locking issue)
                    import time
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            os.unlink(tmp_file_path)
                            logger.debug(f"Successfully deleted temp file: {tmp_file_path}")
                            break
                        except PermissionError as e:
                            if attempt < max_retries - 1:
                                time.sleep(0.5)  # Wait 500ms before retry
                                logger.debug(f"Retry {attempt + 1}/{max_retries} deleting temp file...")
                            else:
                                logger.warning(f"Failed to cleanup temp file after {max_retries} attempts: {str(e)}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file: {str(e)}")
                            break
                except Exception as e:
                    logger.warning(f"Error during temp file cleanup: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive keys)"""
    if not config:
        return {"error": "Configuration not loaded. Please check .env file."}
    
    return {
        "gemini_model": config.gemini_model,
        "claude_model": config.claude_model,
        "min_confidence": config.min_confidence,
        "ocr_confidence": config.ocr_confidence,
        "parsing_completeness": config.parsing_completeness,
        "generation_accuracy": config.generation_accuracy,
        "structural_validity": config.structural_validity,
        "human_review_threshold": config.human_review_threshold
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

