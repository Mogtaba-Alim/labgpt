"""
Quality Critic for Generated Q&A Pairs

This module provides sophisticated quality assessment for generated Q&A pairs,
scoring them on groundedness, specificity, clarity, and usefulness.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..tasks.qa_generator import GroundedQA
from ..assembly.config_manager import ConfigManager, QualityThresholds


@dataclass
class QualityScores:
    """Quality scores for a Q&A pair."""
    groundedness: float = 0.0      # How well answer is grounded in context (0-1)
    specificity: float = 0.0       # How specific and detailed the answer is (0-1)
    clarity: float = 0.0           # How clear and understandable the answer is (0-1)
    usefulness: float = 0.0        # How useful the Q&A is for training (0-1)
    consistency: float = 0.0       # Internal consistency of the answer (0-1)
    completeness: float = 0.0      # How complete the answer is (0-1)
    overall_score: float = 0.0     # Weighted overall score (0-1)
    
    def meets_thresholds(self, thresholds: QualityThresholds) -> bool:
        """Check if scores meet minimum thresholds."""
        return (
            self.groundedness >= thresholds.groundedness_min_score and
            self.specificity >= thresholds.specificity_min_score and
            self.clarity >= thresholds.clarity_min_score and
            self.usefulness >= thresholds.usefulness_min_score
        )


@dataclass
class CriticFeedback:
    """Detailed feedback from the quality critic."""
    scores: QualityScores
    detailed_feedback: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    citation_analysis: Dict[str, Any] = field(default_factory=dict)
    passes_quality_check: bool = False
    should_regenerate: bool = False
    regeneration_guidance: str = ""


@dataclass
class CriticStats:
    """Statistics from quality criticism process."""
    total_evaluations: int = 0
    passed_evaluations: int = 0
    failed_evaluations: int = 0
    average_scores: QualityScores = field(default_factory=QualityScores)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    regeneration_requests: int = 0


class QualityCritic:
    """Evaluates quality of generated Q&A pairs using LLM-based assessment."""
    
    def __init__(self, config_manager: ConfigManager, llm_client):
        """
        Initialize the quality critic.
        
        Args:
            config_manager: Configuration manager for quality thresholds
            llm_client: LLM client for quality evaluation
        """
        self.config_manager = config_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.stats = CriticStats()
        
        # Get quality thresholds
        self.thresholds = config_manager.get_quality_thresholds()
        
        # Scoring weights for overall score calculation
        self.score_weights = {
            'groundedness': 0.30,
            'specificity': 0.20,
            'clarity': 0.20,
            'usefulness': 0.15,
            'consistency': 0.10,
            'completeness': 0.05
        }
    
    def evaluate_qa_pair(self, qa_pair: GroundedQA) -> CriticFeedback:
        """
        Evaluate a single Q&A pair for quality.
        
        Args:
            qa_pair: The Q&A pair to evaluate
            
        Returns:
            Detailed feedback with scores and suggestions
        """
        try:
            # Get scores from LLM evaluation
            scores = self._get_llm_scores(qa_pair)
            
            # Perform citation analysis
            citation_analysis = self._analyze_citations(qa_pair)
            
            # Generate detailed feedback
            feedback = self._generate_detailed_feedback(qa_pair, scores, citation_analysis)
            
            # Determine if it passes quality check
            passes_check = scores.meets_thresholds(self.thresholds)
            
            # Generate regeneration guidance if needed
            should_regenerate, regeneration_guidance = self._should_regenerate(qa_pair, scores)
            
            # Create feedback object
            critic_feedback = CriticFeedback(
                scores=scores,
                detailed_feedback=feedback['detailed_feedback'],
                strengths=feedback['strengths'],
                weaknesses=feedback['weaknesses'],
                suggestions=feedback['suggestions'],
                citation_analysis=citation_analysis,
                passes_quality_check=passes_check,
                should_regenerate=should_regenerate,
                regeneration_guidance=regeneration_guidance
            )
            
            # Update statistics
            self._update_stats(critic_feedback)
            
            return critic_feedback
            
        except Exception as e:
            self.logger.error(f"Error evaluating Q&A pair: {e}")
            # Return default failing feedback
            return CriticFeedback(
                scores=QualityScores(),
                detailed_feedback=f"Error during evaluation: {str(e)}",
                passes_quality_check=False,
                should_regenerate=True,
                regeneration_guidance="Re-evaluation needed due to error."
            )
    
    def _get_llm_scores(self, qa_pair: GroundedQA) -> QualityScores:
        """Get quality scores from LLM evaluation."""
        
        system_prompt = """You are a strict data quality auditor for training data generation. Evaluate Q&A pairs on multiple quality dimensions.

EVALUATION CRITERIA:
1. GROUNDEDNESS (0-1): How well is the answer supported by the provided context?
   - 1.0: Answer directly derivable from context with clear evidence
   - 0.7-0.9: Answer mostly supported by context with some inference
   - 0.5-0.6: Answer partially supported, some speculation
   - 0.0-0.4: Answer not supported by context or requires external knowledge

2. SPECIFICITY (0-1): How specific and detailed is the answer?
   - 1.0: Highly specific with concrete details and examples
   - 0.7-0.9: Good specificity with relevant details
   - 0.5-0.6: Moderately specific
   - 0.0-0.4: Vague or generic

3. CLARITY (0-1): How clear and understandable is the answer?
   - 1.0: Crystal clear, well-structured, easy to understand
   - 0.7-0.9: Clear with minor ambiguities
   - 0.5-0.6: Somewhat clear but could be improved
   - 0.0-0.4: Unclear, confusing, or poorly structured

4. USEFULNESS (0-1): How useful is this Q&A for training a code understanding model?
   - 1.0: Highly valuable for learning code understanding
   - 0.7-0.9: Good training value
   - 0.5-0.6: Moderate training value
   - 0.0-0.4: Low training value

5. CONSISTENCY (0-1): Is the answer internally consistent?
   - 1.0: Perfectly consistent, no contradictions
   - 0.7-0.9: Mostly consistent with minor issues
   - 0.5-0.6: Some inconsistencies
   - 0.0-0.4: Major inconsistencies or contradictions

6. COMPLETENESS (0-1): How complete is the answer for the question asked?
   - 1.0: Fully answers the question
   - 0.7-0.9: Mostly complete with minor gaps
   - 0.5-0.6: Partially complete
   - 0.0-0.4: Incomplete or doesn't address the question

Return ONLY a valid JSON object with scores:
{
  "groundedness": 0.0,
  "specificity": 0.0,
  "clarity": 0.0,
  "usefulness": 0.0,
  "consistency": 0.0,
  "completeness": 0.0
}

Be strict in your evaluation. High scores should be reserved for truly excellent Q&A pairs."""

        # Handle both regular QA pairs and NegativeExample objects
        answer_text = getattr(qa_pair, 'answer', getattr(qa_pair, 'expected_response', 'NOT_IN_CONTEXT'))
        
        user_prompt = f"""QUESTION: {qa_pair.question}

ANSWER: {answer_text}

CODE CONTEXT:
```python
{getattr(qa_pair, 'context_symbol_text', getattr(qa_pair, 'context_code', ''))}
```

CONTEXT INFO:
- Symbol: {qa_pair.context_symbol_name}
- Type: {qa_pair.context_symbol_type}
- Lines: {getattr(qa_pair, 'context_start_line', 0)}-{getattr(qa_pair, 'context_end_line', 0)}
- Focus Area: {getattr(qa_pair, 'task_focus_area', 'general')}
- Complexity: {getattr(qa_pair, 'complexity_level', getattr(qa_pair, 'difficulty_level', 'moderate'))}
- Is Negative Example: {getattr(qa_pair, 'is_negative_example', hasattr(qa_pair, 'negative_type'))}

Evaluate this Q&A pair according to the criteria above and return only the JSON scores:"""

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            scores_text = response.content[0].text.strip()
            
            # Parse JSON scores
            scores_data = self._parse_scores_json(scores_text)
            
            # Create QualityScores object
            scores = QualityScores(
                groundedness=scores_data.get('groundedness', 0.0),
                specificity=scores_data.get('specificity', 0.0),
                clarity=scores_data.get('clarity', 0.0),
                usefulness=scores_data.get('usefulness', 0.0),
                consistency=scores_data.get('consistency', 0.0),
                completeness=scores_data.get('completeness', 0.0)
            )
            
            # Calculate overall score
            scores.overall_score = self._calculate_overall_score(scores)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error getting LLM scores: {e}")
            return QualityScores()  # Return zeros on error
    
    def _parse_scores_json(self, scores_text: str) -> Dict[str, float]:
        """Parse scores from LLM response."""
        try:
            # Try direct JSON parsing
            return json.loads(scores_text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^}]+\}', scores_text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract individual scores
            scores = {}
            score_patterns = [
                (r'"groundedness":\s*([0-9.]+)', 'groundedness'),
                (r'"specificity":\s*([0-9.]+)', 'specificity'),
                (r'"clarity":\s*([0-9.]+)', 'clarity'),
                (r'"usefulness":\s*([0-9.]+)', 'usefulness'),
                (r'"consistency":\s*([0-9.]+)', 'consistency'),
                (r'"completeness":\s*([0-9.]+)', 'completeness')
            ]
            
            for pattern, key in score_patterns:
                match = re.search(pattern, scores_text)
                if match:
                    try:
                        scores[key] = float(match.group(1))
                    except ValueError:
                        scores[key] = 0.0
                else:
                    scores[key] = 0.0
            
            return scores
    
    def _calculate_overall_score(self, scores: QualityScores) -> float:
        """Calculate weighted overall score."""
        overall = (
            scores.groundedness * self.score_weights['groundedness'] +
            scores.specificity * self.score_weights['specificity'] +
            scores.clarity * self.score_weights['clarity'] +
            scores.usefulness * self.score_weights['usefulness'] +
            scores.consistency * self.score_weights['consistency'] +
            scores.completeness * self.score_weights['completeness']
        )
        return round(overall, 3)
    
    def _analyze_citations(self, qa_pair: GroundedQA) -> Dict[str, Any]:
        """Analyze citations in the Q&A pair."""
        # Handle objects without citations (like NegativeExample)
        citations = getattr(qa_pair, 'citations', [])

        analysis = {
            'has_citations': len(citations) > 0,
            'citation_count': len(citations),
            'citation_types': [],
            'code_coverage': 0.0,
            'line_references': 0,
            'code_snippets': 0
        }

        # Analyze citation types
        for citation in citations:
            if citation.startswith('line'):
                analysis['citation_types'].append('line_reference')
                analysis['line_references'] += 1
            elif citation.startswith('code:'):
                analysis['citation_types'].append('code_snippet')
                analysis['code_snippets'] += 1
        
        # Calculate code coverage (rough estimate)
        answer_text = getattr(qa_pair, 'answer', getattr(qa_pair, 'expected_response', ''))
        context_text = getattr(qa_pair, 'context_symbol_text', getattr(qa_pair, 'context_code', ''))
        
        if answer_text and context_text:
            # Count how many words from the answer appear in the context
            answer_words = set(re.findall(r'\w+', answer_text.lower()))
            context_words = set(re.findall(r'\w+', context_text.lower()))
            
            if answer_words:
                overlap = len(answer_words.intersection(context_words))
                analysis['code_coverage'] = overlap / len(answer_words)
        
        return analysis
    
    def _generate_detailed_feedback(self, qa_pair: GroundedQA, scores: QualityScores, 
                                  citation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed feedback for the Q&A pair."""
        strengths = []
        weaknesses = []
        suggestions = []
        
        # Analyze groundedness
        if scores.groundedness >= 0.8:
            strengths.append("Answer is well-grounded in the provided context")
        elif scores.groundedness < 0.5:
            weaknesses.append("Answer lacks proper grounding in context")
            suggestions.append("Ensure answer is directly derivable from the code context")
        
        # Analyze specificity
        if scores.specificity >= 0.8:
            strengths.append("Answer provides specific and detailed information")
        elif scores.specificity < 0.5:
            weaknesses.append("Answer is too vague or generic")
            suggestions.append("Include more specific details and examples from the code")
        
        # Analyze clarity
        if scores.clarity >= 0.8:
            strengths.append("Answer is clear and well-structured")
        elif scores.clarity < 0.5:
            weaknesses.append("Answer is unclear or poorly structured")
            suggestions.append("Improve clarity and organization of the answer")
        
        # Analyze usefulness
        if scores.usefulness >= 0.8:
            strengths.append("Q&A pair has high training value")
        elif scores.usefulness < 0.5:
            weaknesses.append("Q&A pair has limited training value")
            suggestions.append("Focus on more educationally valuable aspects of the code")
        
        # Analyze citations
        if not citation_analysis['has_citations'] and not getattr(qa_pair, 'is_negative_example', False):
            weaknesses.append("Answer lacks proper citations to the code")
            suggestions.append("Include line references or code snippets to support claims")
        elif citation_analysis['has_citations']:
            strengths.append("Answer includes proper citations")
        
        # Generate overall feedback
        if scores.overall_score >= 0.8:
            detailed_feedback = "Excellent Q&A pair with high quality across all dimensions."
        elif scores.overall_score >= 0.6:
            detailed_feedback = "Good Q&A pair with some areas for improvement."
        elif scores.overall_score >= 0.4:
            detailed_feedback = "Acceptable Q&A pair but needs significant improvement."
        else:
            detailed_feedback = "Poor quality Q&A pair that should be regenerated."
        
        return {
            'detailed_feedback': detailed_feedback,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'suggestions': suggestions
        }
    
    def _should_regenerate(self, qa_pair: GroundedQA, scores: QualityScores) -> Tuple[bool, str]:
        """Determine if Q&A pair should be regenerated."""
        should_regenerate = False
        guidance = ""
        
        # Check if it fails threshold requirements
        if not scores.meets_thresholds(self.thresholds):
            should_regenerate = True
            
            # Provide specific guidance
            guidance_parts = []
            if scores.groundedness < self.thresholds.groundedness_min_score:
                guidance_parts.append("improve grounding in provided context")
            if scores.specificity < self.thresholds.specificity_min_score:
                guidance_parts.append("increase specificity and detail")
            if scores.clarity < self.thresholds.clarity_min_score:
                guidance_parts.append("improve clarity and structure")
            if scores.usefulness < self.thresholds.usefulness_min_score:
                guidance_parts.append("focus on educational value")
            
            guidance = "Regenerate with focus on: " + ", ".join(guidance_parts)
        
        # Special check for NOT_IN_CONTEXT answers
        answer_text = getattr(qa_pair, 'answer', getattr(qa_pair, 'expected_response', ''))
        is_negative = getattr(qa_pair, 'is_negative_example', hasattr(qa_pair, 'negative_type'))
        
        if answer_text == "NOT_IN_CONTEXT":
            if not is_negative:
                should_regenerate = True
                guidance = "Question seems to require external knowledge. Consider simplifying or changing the question to be answerable from context."
        
        return should_regenerate, guidance
    
    def _update_stats(self, feedback: CriticFeedback):
        """Update criticism statistics."""
        self.stats.total_evaluations += 1
        
        if feedback.passes_quality_check:
            self.stats.passed_evaluations += 1
        else:
            self.stats.failed_evaluations += 1
            
            # Track failure reasons
            for weakness in feedback.weaknesses:
                self.stats.failure_reasons[weakness] = (
                    self.stats.failure_reasons.get(weakness, 0) + 1
                )
        
        if feedback.should_regenerate:
            self.stats.regeneration_requests += 1
        
        # Update average scores
        scores = feedback.scores
        n = self.stats.total_evaluations
        
        # Running average calculation
        self.stats.average_scores.groundedness = (
            (self.stats.average_scores.groundedness * (n - 1) + scores.groundedness) / n
        )
        self.stats.average_scores.specificity = (
            (self.stats.average_scores.specificity * (n - 1) + scores.specificity) / n
        )
        self.stats.average_scores.clarity = (
            (self.stats.average_scores.clarity * (n - 1) + scores.clarity) / n
        )
        self.stats.average_scores.usefulness = (
            (self.stats.average_scores.usefulness * (n - 1) + scores.usefulness) / n
        )
        self.stats.average_scores.consistency = (
            (self.stats.average_scores.consistency * (n - 1) + scores.consistency) / n
        )
        self.stats.average_scores.completeness = (
            (self.stats.average_scores.completeness * (n - 1) + scores.completeness) / n
        )
        self.stats.average_scores.overall_score = (
            (self.stats.average_scores.overall_score * (n - 1) + scores.overall_score) / n
        )
    
    def evaluate_batch(self, qa_pairs: List[GroundedQA]) -> List[CriticFeedback]:
        """Evaluate a batch of Q&A pairs."""
        feedbacks = []
        
        for qa_pair in qa_pairs:
            feedback = self.evaluate_qa_pair(qa_pair)
            feedbacks.append(feedback)
        
        return feedbacks
    
    def filter_high_quality(self, qa_pairs: List[GroundedQA]) -> List[GroundedQA]:
        """Filter Q&A pairs to keep only high quality ones."""
        high_quality_pairs = []
        
        for qa_pair in qa_pairs:
            feedback = self.evaluate_qa_pair(qa_pair)
            if feedback.passes_quality_check:
                # Update the QA pair with confidence score
                qa_pair.confidence_score = feedback.scores.overall_score
                high_quality_pairs.append(qa_pair)
        
        return high_quality_pairs
    
    def get_stats(self) -> CriticStats:
        """Get criticism statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset criticism statistics."""
        self.stats = CriticStats() 