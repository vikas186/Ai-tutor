"""
Data models with strict validation for 100% accuracy
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal
from uuid import UUID, uuid4
from enum import Enum


class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValidationCheck(BaseModel):
    """Individual validation check result"""
    check_type: str
    passed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: str


class PerfectQuestion(BaseModel):
    """Perfect question model with strict validation"""
    id: UUID = Field(default_factory=uuid4)
    question_text: str = Field(..., min_length=10)
    question_type: QuestionType
    options: Optional[List[str]] = Field(None, min_length=2)  # Required for MCQs
    correct_answer: str = Field(..., min_length=1)
    explanation: Optional[str] = None
    difficulty: Difficulty
    tags: List[str] = Field(..., min_length=1)
    subject: str = Field(..., min_length=2)
    topic: str = Field(..., min_length=2)
    confidence_score: float = Field(default=0.95, ge=0.95, le=1.0)  # Minimum 95% confidence
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_options_for_mcq(self):
        """Ensure options are provided for MCQ type"""
        if self.question_type == QuestionType.MCQ:
            if not self.options or len(self.options) < 2:
                raise ValueError("MCQ questions must have at least 2 options")
        return self


class OCRResult(BaseModel):
    """OCR extraction result with confidence"""
    extracted_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    requires_human_review: bool = False


class ParsingResult(BaseModel):
    """Question parsing result"""
    questions: List[PerfectQuestion]
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    parsing_errors: List[str] = Field(default_factory=list)
    requires_human_review: bool = False


class GenerationResult(BaseModel):
    """Question generation result"""
    generated_questions: List[PerfectQuestion]
    source_question_id: UUID
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    requires_human_review: bool = False


class ValidationResult(BaseModel):
    """Overall validation result"""
    passed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    failed_validations: List[ValidationCheck] = Field(default_factory=list)
    passed_validations: List[ValidationCheck] = Field(default_factory=list)
    requires_human_review: bool = False


class AccuracyError(Exception):
    """Custom exception for accuracy failures"""
    pass


class ConsensusCheck(BaseModel):
    """Consensus validation check between multiple LLMs"""
    models_used: List[str]
    agreement_score: float = Field(..., ge=0.0, le=1.0)
    disagreements: List[str] = Field(default_factory=list)
    consensus_strategy: str
    final_choice: str  # Which model's output was chosen


class EnsembleParsingResult(BaseModel):
    """Enhanced parsing result with multi-LLM consensus"""
    questions: List[PerfectQuestion]
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    consensus_checks: List[ConsensusCheck] = Field(default_factory=list)
    parsing_errors: List[str] = Field(default_factory=list)
    requires_human_review: bool = False
    models_agreement_score: float = Field(default=1.0, ge=0.0, le=1.0)


class EnsembleOCRResult(BaseModel):
    """Enhanced OCR result with multi-LLM validation"""
    extracted_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    consensus_checks: List[ConsensusCheck] = Field(default_factory=list)
    requires_human_review: bool = False
    models_agreement_score: float = Field(default=1.0, ge=0.0, le=1.0)


class EnsembleGenerationResult(BaseModel):
    """Enhanced generation result with multi-LLM validation"""
    generated_questions: List[PerfectQuestion]
    source_question_id: UUID
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    consensus_checks: List[ConsensusCheck] = Field(default_factory=list)
    requires_human_review: bool = False
    models_agreement_score: float = Field(default=1.0, ge=0.0, le=1.0)
