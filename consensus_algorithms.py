"""
Consensus Algorithms for Multi-LLM Validation
Implements various strategies for merging and comparing outputs from multiple LLMs
"""
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
import json
import re
import logging

logger = logging.getLogger(__name__)


class TextSimilarity:
    """Text similarity metrics for comparing LLM outputs"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return TextSimilarity.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings (0-1)"""
        return SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def normalized_levenshtein(s1: str, s2: str) -> float:
        """Normalized Levenshtein similarity (0-1, higher is more similar)"""
        distance = TextSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        """Jaccard similarity based on word sets"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def combined_similarity(s1: str, s2: str) -> float:
        """Combined similarity metric using multiple approaches"""
        ratio = TextSimilarity.similarity_ratio(s1, s2)
        jaccard = TextSimilarity.jaccard_similarity(s1, s2)
        levenshtein = TextSimilarity.normalized_levenshtein(s1, s2)
        
        # Weighted average
        return (ratio * 0.4 + jaccard * 0.3 + levenshtein * 0.3)


class JSONMerger:
    """Merge JSON outputs from multiple LLMs"""
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict]:
        """Extract JSON from text response"""
        # Remove markdown code blocks
        text_cleaned = re.sub(r'```json\s*', '', text)
        text_cleaned = re.sub(r'```\s*', '', text_cleaned)
        text_cleaned = text_cleaned.strip()
        
        # Try to find JSON pattern
        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try parsing entire text
        try:
            return json.loads(text_cleaned)
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def merge_question_lists(json_a: Dict, json_b: Dict) -> Dict:
        """
        Merge two JSON responses containing question lists
        
        Args:
            json_a: First JSON response
            json_b: Second JSON response
            
        Returns:
            Merged JSON with deduplicated questions
        """
        questions_a = json_a.get("questions", [])
        questions_b = json_b.get("questions", [])
        
        logger.info(f"Merging question lists: {len(questions_a)} from A, {len(questions_b)} from B")
        
        # Use both lists and deduplicate
        merged_questions = []
        seen_texts = set()
        
        # Add all questions from A
        for q in questions_a:
            q_text = q.get("question_text", "").strip().lower()
            normalized = re.sub(r'\s+', ' ', q_text)
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            if normalized and normalized not in seen_texts:
                seen_texts.add(normalized)
                merged_questions.append(q)
        
        # Add questions from B that aren't duplicates
        for q in questions_b:
            q_text = q.get("question_text", "").strip().lower()
            normalized = re.sub(r'\s+', ' ', q_text)
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            if normalized and normalized not in seen_texts:
                # Check similarity with existing questions
                is_duplicate = False
                for existing_text in seen_texts:
                    similarity = TextSimilarity.similarity_ratio(normalized, existing_text)
                    if similarity > 0.85:  # High similarity threshold
                        is_duplicate = True
                        logger.debug(f"Found duplicate question (similarity: {similarity:.2f})")
                        break
                
                if not is_duplicate:
                    seen_texts.add(normalized)
                    merged_questions.append(q)
        
        logger.info(f"Merged result: {len(merged_questions)} unique questions")
        
        # Create merged JSON
        merged = {
            "questions": merged_questions,
            "confidence": (json_a.get("confidence", 0.95) + json_b.get("confidence", 0.95)) / 2,
            "validation_errors": json_a.get("validation_errors", []) + json_b.get("validation_errors", [])
        }
        
        return merged
    
    @staticmethod
    def merge_with_voting(json_list: List[Dict], field_name: str = "questions") -> Dict:
        """
        Merge multiple JSON responses using voting for each item
        
        Args:
            json_list: List of JSON responses
            field_name: Name of the list field to merge
            
        Returns:
            Merged JSON with items that appear in majority
        """
        if not json_list:
            return {"questions": [], "confidence": 0.0}
        
        if len(json_list) == 1:
            return json_list[0]
        
        # Collect all items
        all_items = []
        for json_obj in json_list:
            items = json_obj.get(field_name, [])
            all_items.extend(items)
        
        # Count occurrences (by similarity)
        item_counts = {}
        for item in all_items:
            item_text = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
            
            # Find similar existing items
            found_similar = False
            for existing_text in item_counts:
                similarity = TextSimilarity.similarity_ratio(item_text, existing_text)
                if similarity > 0.8:
                    item_counts[existing_text]["count"] += 1
                    found_similar = True
                    break
            
            if not found_similar:
                item_counts[item_text] = {"item": item, "count": 1}
        
        # Select items that appear in majority (or at least once if only 2 models)
        threshold = len(json_list) / 2 if len(json_list) > 2 else 1
        merged_items = [
            data["item"] for data in item_counts.values()
            if data["count"] >= threshold
        ]
        
        # Calculate average confidence
        avg_confidence = sum(j.get("confidence", 0.95) for j in json_list) / len(json_list)
        
        return {
            field_name: merged_items,
            "confidence": avg_confidence,
            "validation_errors": []
        }


class QuestionComparator:
    """Compare and match questions from different LLM outputs"""
    
    @staticmethod
    def find_matching_questions(questions_a: List[Dict], 
                               questions_b: List[Dict],
                               similarity_threshold: float = 0.85) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
        """
        Find matching questions between two lists
        
        Args:
            questions_a: First list of questions
            questions_b: Second list of questions
            similarity_threshold: Threshold for considering questions as matching
            
        Returns:
            Tuple of (matches, only_in_a, only_in_b)
        """
        matches = []
        only_in_a = []
        only_in_b = list(questions_b)  # Start with all from B
        
        for q_a in questions_a:
            text_a = q_a.get("question_text", "").strip()
            
            # Find best match in B
            best_match = None
            best_similarity = 0.0
            
            for q_b in questions_b:
                text_b = q_b.get("question_text", "").strip()
                similarity = TextSimilarity.combined_similarity(text_a, text_b)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = q_b
            
            # Check if match is good enough
            if best_match and best_similarity >= similarity_threshold:
                matches.append((q_a, best_match))
                if best_match in only_in_b:
                    only_in_b.remove(best_match)
            else:
                only_in_a.append(q_a)
        
        logger.info(f"Question matching: {len(matches)} matches, {len(only_in_a)} only in A, {len(only_in_b)} only in B")
        
        return matches, only_in_a, only_in_b
    
    @staticmethod
    def merge_matched_questions(q_a: Dict, q_b: Dict) -> Dict:
        """
        Merge two matching questions, combining their information
        
        Args:
            q_a: First question
            q_b: Second question
            
        Returns:
            Merged question with combined information
        """
        merged = q_a.copy()
        
        # Use longer/more detailed text
        text_a = q_a.get("question_text", "")
        text_b = q_b.get("question_text", "")
        merged["question_text"] = text_a if len(text_a) >= len(text_b) else text_b
        
        # Combine options if both have them
        options_a = q_a.get("options", [])
        options_b = q_b.get("options", [])
        if options_a and options_b:
            # Use the one with more options
            merged["options"] = options_a if len(options_a) >= len(options_b) else options_b
        elif options_b:
            merged["options"] = options_b
        
        # Use more specific answer if available
        answer_a = q_a.get("correct_answer") or "N/A"
        answer_b = q_b.get("correct_answer") or "N/A"
        if answer_b and answer_b != "N/A" and (not answer_a or answer_a == "N/A" or answer_a is None):
            merged["correct_answer"] = answer_b
        elif answer_a and answer_a != "N/A" and answer_a is not None:
            merged["correct_answer"] = answer_a
        else:
            merged["correct_answer"] = "N/A"
        
        # Combine explanations
        exp_a = q_a.get("explanation", "")
        exp_b = q_b.get("explanation", "")
        if exp_a and exp_b:
            merged["explanation"] = exp_a if len(exp_a) >= len(exp_b) else exp_b
        elif exp_b:
            merged["explanation"] = exp_b
        
        # Combine tags
        tags_a = set(q_a.get("tags", []))
        tags_b = set(q_b.get("tags", []))
        merged["tags"] = list(tags_a.union(tags_b))
        
        # Average confidence if both have it
        conf_a = q_a.get("confidence", 0.95)
        conf_b = q_b.get("confidence", 0.95)
        merged["confidence"] = (conf_a + conf_b) / 2
        
        return merged


class ConsensusScorer:
    """Calculate consensus scores for multi-LLM outputs"""
    
    @staticmethod
    def calculate_agreement_score(responses: List[str]) -> float:
        """
        Calculate agreement score across multiple text responses
        
        Args:
            responses: List of text responses
            
        Returns:
            Agreement score between 0 and 1
        """
        if len(responses) < 2:
            return 1.0
        
        # Compare all pairs
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = TextSimilarity.combined_similarity(responses[i], responses[j])
                similarities.append(sim)
        
        # Average similarity across all pairs
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        logger.info(f"Agreement score: {avg_similarity:.2f} across {len(responses)} responses")
        
        return avg_similarity
    
    @staticmethod
    def calculate_json_agreement(json_responses: List[Dict], key_fields: List[str]) -> float:
        """
        Calculate agreement for JSON responses based on key fields
        
        Args:
            json_responses: List of JSON responses
            key_fields: Fields to compare for agreement
            
        Returns:
            Agreement score between 0 and 1
        """
        if len(json_responses) < 2:
            return 1.0
        
        field_scores = []
        
        for field in key_fields:
            # Extract field values
            values = [str(j.get(field, "")) for j in json_responses]
            
            # Calculate agreement for this field
            if all(v == values[0] for v in values):
                field_scores.append(1.0)  # Perfect agreement
            else:
                # Partial agreement based on similarity
                similarities = []
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        sim = TextSimilarity.similarity_ratio(values[i], values[j])
                        similarities.append(sim)
                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                field_scores.append(avg_sim)
        
        # Average across all fields
        overall_score = sum(field_scores) / len(field_scores) if field_scores else 0.0
        
        return overall_score

