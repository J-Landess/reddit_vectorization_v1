"""
Intelligent filtering and prioritization for healthcare Reddit data collection.
Focuses on claim denials, coverage issues, troubleshooting, and personal experiences.
"""
import re
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class IntelligentHealthcareFilter:
    """Intelligent filtering for healthcare-related Reddit content."""
    
    def __init__(self):
        """Initialize the intelligent filter with healthcare-specific patterns."""
        # Healthcare keywords for prioritization
        self.healthcare_keywords = {
            'claim_denials': [
                'denied', 'denial', 'rejected', 'refused', 'declined', 'not covered',
                'coverage denied', 'claim denied', 'insurance denied', 'preauthorization',
                'prior authorization', 'appeal', 'appealing', 'grievance'
            ],
            'coverage_issues': [
                'coverage', 'insurance', 'deductible', 'copay', 'copayment', 'premium',
                'out of pocket', 'network', 'in network', 'out of network', 'formulary',
                'prior auth', 'preauth', 'step therapy', 'fail first', 'tier'
            ],
            'troubleshooting': [
                'help', 'advice', 'troubleshoot', 'issue', 'problem', 'error', 'fix',
                'solution', 'guidance', 'support', 'assistance', 'stuck', 'confused',
                'don\'t understand', 'how to', 'what should', 'need help'
            ],
            'personal_experiences': [
                'my experience', 'i had', 'i was', 'i am', 'i feel', 'i think',
                'personally', 'for me', 'in my case', 'my situation', 'my story',
                'happened to me', 'went through', 'dealing with'
            ]
        }
        
        # Question words and first-person indicators
        self.question_words = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could',
            'should', 'would', 'is', 'are', 'do', 'does', 'did', 'will', 'have',
            'has', 'had', 'may', 'might', 'must'
        ]
        
        self.first_person_pronouns = [
            'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
        ]
        
        # Patterns to filter out (news, links, low-quality content)
        self.exclude_patterns = [
            r'https?://\S+',  # URLs
            r'www\.\S+',      # www links
            r'\[deleted\]',   # Deleted content
            r'\[removed\]',   # Removed content
            r'^[^\w\s]*$',    # Only punctuation/symbols
            r'^\s*$',         # Empty/whitespace only
        ]
        
        # News/link-heavy indicators
        self.news_indicators = [
            'breaking', 'news', 'report', 'study shows', 'research', 'according to',
            'sources say', 'officials', 'announced', 'released', 'published',
            'article', 'link', 'source', 'reference'
        ]
        
        logger.info("Intelligent healthcare filter initialized")
    
    def calculate_relevance_score(self, text: str, title: str = "") -> float:
        """
        Calculate relevance score for healthcare content.
        
        Args:
            text: Post/comment text
            title: Post title (optional)
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        combined_text = f"{title} {text}".lower()
        score = 0.0
        
        # Check for healthcare keywords (weighted by importance)
        for category, keywords in self.healthcare_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    category_score += 1
            
            # Weight different categories
            if category == 'claim_denials':
                score += category_score * 0.4  # Highest priority
            elif category == 'coverage_issues':
                score += category_score * 0.3
            elif category == 'troubleshooting':
                score += category_score * 0.2
            elif category == 'personal_experiences':
                score += category_score * 0.1
        
        # Check for question words (indicates seeking help)
        question_count = sum(1 for word in self.question_words if word in combined_text)
        score += question_count * 0.05
        
        # Check for first-person pronouns (personal experience)
        first_person_count = sum(1 for word in self.first_person_pronouns if word in combined_text)
        score += first_person_count * 0.03
        
        # Normalize score (max possible is around 2.0)
        normalized_score = min(score / 2.0, 1.0)
        
        return normalized_score
    
    def is_high_quality_content(self, text: str, title: str = "") -> bool:
        """
        Determine if content is high quality and relevant.
        
        Args:
            text: Post/comment text
            title: Post title (optional)
            
        Returns:
            True if content should be included
        """
        if not text or len(text.strip()) < 20:
            return False
        
        combined_text = f"{title} {text}".lower()
        
        # Check for exclusion patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return False
        
        # Check for news/link-heavy content
        news_count = sum(1 for indicator in self.news_indicators if indicator in combined_text)
        if news_count > 2:  # Too many news indicators
            return False
        
        # Check for minimum word count
        word_count = len(combined_text.split())
        if word_count < 10:
            return False
        
        # Check for excessive punctuation (spam indicator)
        punctuation_ratio = len(re.findall(r'[!?]{2,}', combined_text)) / max(word_count, 1)
        if punctuation_ratio > 0.1:  # More than 10% excessive punctuation
            return False
        
        return True
    
    def prioritize_content(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize posts based on healthcare relevance.
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Prioritized list of posts
        """
        logger.info(f"Prioritizing {len(posts)} posts")
        
        prioritized_posts = []
        
        for post in posts:
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(
                post.get('text', ''), 
                post.get('title', '')
            )
            
            # Check if content is high quality
            if not self.is_high_quality_content(
                post.get('text', ''), 
                post.get('title', '')
            ):
                continue
            
            # Add relevance score to post metadata
            post['relevance_score'] = relevance_score
            post['is_healthcare_relevant'] = relevance_score > 0.1
            
            prioritized_posts.append(post)
        
        # Sort by relevance score (highest first)
        prioritized_posts.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Prioritized to {len(prioritized_posts)} relevant posts")
        return prioritized_posts
    
    def filter_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter comments for healthcare relevance.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Filtered list of comments
        """
        logger.info(f"Filtering {len(comments)} comments")
        
        filtered_comments = []
        
        for comment in comments:
            text = comment.get('text', '')
            
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(text)
            
            # Check if content is high quality
            if not self.is_high_quality_content(text):
                continue
            
            # Add relevance score to comment metadata
            comment['relevance_score'] = relevance_score
            comment['is_healthcare_relevant'] = relevance_score > 0.1
            
            filtered_comments.append(comment)
        
        # Sort by relevance score (highest first)
        filtered_comments.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Filtered to {len(filtered_comments)} relevant comments")
        return filtered_comments
    
    def get_collection_strategy(self, target_samples: int = 50000) -> Dict[str, Any]:
        """
        Get collection strategy based on target sample count.
        
        Args:
            target_samples: Target number of samples
            
        Returns:
            Collection strategy dictionary
        """
        # Calculate posts per subreddit based on target
        subreddits = [
            'healthinsurance', 'Medicare', 'Medicaid', 'medicalproviders',
            'AskDocs', 'Health', 'ChronicIllness', 'PatientExperience',
            'MedicalBilling', 'Pharmacy', 'MentalHealth', 'medical', 'Obamacare'
        ]
        
        posts_per_subreddit = max(50, target_samples // (len(subreddits) * 10))  # Conservative estimate
        comments_per_post = max(20, target_samples // (len(subreddits) * posts_per_subreddit))
        
        strategy = {
            'target_samples': target_samples,
            'subreddits': subreddits,
            'posts_per_subreddit': posts_per_subreddit,
            'comments_per_post': comments_per_post,
            'estimated_total': len(subreddits) * posts_per_subreddit * (1 + comments_per_post),
            'prioritization_enabled': True,
            'quality_filtering': True
        }
        
        logger.info(f"Collection strategy: {strategy}")
        return strategy
    
    def get_filtering_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about filtering results.
        
        Args:
            data: List of filtered data dictionaries
            
        Returns:
            Filtering statistics
        """
        if not data:
            return {}
        
        total_items = len(data)
        relevant_items = len([item for item in data if item.get('is_healthcare_relevant', False)])
        
        # Calculate average relevance scores
        relevance_scores = [item.get('relevance_score', 0) for item in data]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Count by relevance categories
        high_relevance = len([item for item in data if item.get('relevance_score', 0) > 0.5])
        medium_relevance = len([item for item in data if 0.2 < item.get('relevance_score', 0) <= 0.5])
        low_relevance = len([item for item in data if 0 < item.get('relevance_score', 0) <= 0.2])
        
        stats = {
            'total_items': total_items,
            'relevant_items': relevant_items,
            'relevance_rate': relevant_items / total_items if total_items > 0 else 0,
            'avg_relevance_score': avg_relevance,
            'high_relevance_count': high_relevance,
            'medium_relevance_count': medium_relevance,
            'low_relevance_count': low_relevance,
            'filtering_effectiveness': f"{(relevant_items/total_items)*100:.1f}%" if total_items > 0 else "0%"
        }
        
        return stats
