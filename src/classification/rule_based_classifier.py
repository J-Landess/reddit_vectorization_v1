"""
Rule-based classifier for medical category classification using keywords and patterns.
"""
import re
import logging
from typing import Tuple, Dict, List
from .base import ClassificationAnalyzer, MEDICAL_CATEGORIES

logger = logging.getLogger(__name__)

class RuleBasedClassifier(ClassificationAnalyzer):
    """Rule-based classifier using keywords and patterns for medical categories."""
    
    def __init__(self):
        """Initialize the rule-based classifier with medical keywords."""
        super().__init__()
        self._build_keyword_patterns()
        
    def _build_keyword_patterns(self):
        """Build keyword patterns for each medical category."""
        
        # Medical Insurance keywords
        self.insurance_keywords = {
            'exact': [
                'health insurance', 'healthcare insurance', 'medical insurance',
                'dental insurance', 'vision insurance', 'prescription coverage',
                'co-pay', 'copay', 'deductible', 'premium', 'out-of-pocket',
                'coverage', 'benefits', 'enrollment', 'open enrollment',
                'medicare', 'medicaid', 'obamacare', 'aca', 'affordable care act',
                'hmo', 'ppo', 'epo', 'pos', 'hsa', 'fsa', 'hsa account',
                'insurance plan', 'health plan', 'benefit plan'
            ],
            'partial': [
                'insurance', 'coverage', 'premium', 'deductible', 'copay',
                'benefits', 'enrollment', 'medicare', 'medicaid'
            ]
        }
        
        # Medical Provider keywords  
        self.provider_keywords = {
            'exact': [
                'doctor', 'physician', 'nurse', 'specialist', 'surgeon',
                'medical provider', 'healthcare provider', 'primary care',
                'family doctor', 'general practitioner', 'gp', 'md', 'do',
                'clinic', 'hospital', 'medical center', 'health center',
                'appointment', 'consultation', 'examination', 'checkup',
                'medical advice', 'treatment', 'diagnosis', 'prescription',
                'medical care', 'healthcare services', 'patient care'
            ],
            'partial': [
                'doctor', 'nurse', 'physician', 'clinic', 'hospital',
                'appointment', 'treatment', 'medical', 'healthcare'
            ]
        }
        
        # Medical Broker keywords
        self.broker_keywords = {
            'exact': [
                'insurance broker', 'health insurance broker', 'benefits broker',
                'insurance agent', 'health insurance agent', 'benefits consultant',
                'insurance advisor', 'health insurance advisor', 'benefits advisor',
                'insurance marketplace', 'health insurance marketplace',
                'insurance shopping', 'compare plans', 'plan comparison',
                'brokerage', 'insurance sales', 'benefits sales'
            ],
            'partial': [
                'broker', 'agent', 'advisor', 'consultant', 'marketplace',
                'shopping', 'compare', 'sales'
            ]
        }
        
        # Employer keywords
        self.employer_keywords = {
            'exact': [
                'employer', 'company', 'workplace', 'hr', 'human resources',
                'employee benefits', 'work benefits', 'company benefits',
                'employer provided', 'work provided', 'company provided',
                'hr department', 'benefits department', 'payroll',
                'employee handbook', 'benefits package', 'compensation',
                'job', 'employment', 'workplace health', 'corporate health'
            ],
            'partial': [
                'employer', 'company', 'workplace', 'hr', 'employee',
                'work', 'job', 'employment'
            ]
        }
        
        # Compile regex patterns for better matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.patterns = {}
        
        for category in self.categories:
            self.patterns[category] = {
                'exact': [re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) 
                         for keyword in self._get_keywords(category, 'exact')],
                'partial': [re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) 
                           for keyword in self._get_keywords(category, 'partial')]
            }
    
    def _get_keywords(self, category: str, match_type: str) -> List[str]:
        """Get keywords for a specific category and match type."""
        keyword_map = {
            'medical_insurance': self.insurance_keywords,
            'medical_provider': self.provider_keywords, 
            'medical_broker': self.broker_keywords,
            'employer': self.employer_keywords
        }
        return keyword_map.get(category, {}).get(match_type, [])
    
    def classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text using rule-based keyword matching.
        
        Args:
            text: The text to classify
            embedding: Not used in rule-based classification
            
        Returns:
            Tuple of (predicted_category, confidence_score, category_probabilities)
        """
        if not text or not text.strip():
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
        
        # Calculate scores for each category
        category_scores = {}
        
        for category in self.categories:
            score = self._calculate_category_score(text, category)
            category_scores[category] = score
        
        # Find the category with highest score
        predicted_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[predicted_category]
        
        # Calculate confidence (normalize scores)
        total_score = sum(category_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        # Normalize probabilities
        probabilities = {cat: score / total_score if total_score > 0 else 0.25 
                        for cat, score in category_scores.items()}
        
        return predicted_category, confidence, probabilities
    
    def _calculate_category_score(self, text: str, category: str) -> float:
        """Calculate score for a specific category based on keyword matches."""
        score = 0.0
        text_lower = text.lower()
        
        # Exact match patterns (higher weight)
        for pattern in self.patterns[category]['exact']:
            matches = pattern.findall(text_lower)
            score += len(matches) * 2.0  # Higher weight for exact matches
        
        # Partial match patterns (lower weight)
        for pattern in self.patterns[category]['partial']:
            matches = pattern.findall(text_lower)
            score += len(matches) * 1.0  # Lower weight for partial matches
        
        # Bonus for multiple different keyword matches
        unique_matches = set()
        for pattern in self.patterns[category]['exact'] + self.patterns[category]['partial']:
            matches = pattern.findall(text_lower)
            unique_matches.update(matches)
        
        # Diversity bonus
        if len(unique_matches) > 1:
            score += len(unique_matches) * 0.5
        
        return score
    
    def get_keyword_matches(self, text: str, category: str) -> Dict[str, List[str]]:
        """
        Get the specific keywords that matched for a category.
        
        Args:
            text: The text to analyze
            category: The category to check
            
        Returns:
            Dictionary with 'exact' and 'partial' keyword matches
        """
        matches = {'exact': [], 'partial': []}
        text_lower = text.lower()
        
        for pattern in self.patterns[category]['exact']:
            found = pattern.findall(text_lower)
            matches['exact'].extend(found)
        
        for pattern in self.patterns[category]['partial']:
            found = pattern.findall(text_lower)
            matches['partial'].extend(found)
        
        return matches
