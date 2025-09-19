"""
Text preprocessing and cleaning utilities for Reddit data.
"""
import re
import string
import logging
from typing import List, Dict, Any
import nltk
from bs4 import BeautifulSoup
import urllib.parse

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextCleaner:
    """Text cleaning and preprocessing utilities."""
    
    def __init__(self):
        """Initialize the text cleaner with stopwords."""
        self.stop_words = set(stopwords.words('english'))
        # Add common Reddit-specific stopwords
        self.stop_words.update([
            'reddit', 'redditor', 'redditors', 'upvote', 'downvote',
            'edit', 'edited', 'update', 'updates', 'ps', 'pss', 'psss',
            'tl', 'dr', 'tldr', 'fyi', 'imo', 'imho', 'afaik', 'eli5',
            'yta', 'nta', 'esh', 'nah', 'nah', 'yep', 'nope', 'yeah',
            'lol', 'lmao', 'rofl', 'wtf', 'omg', 'fml', 'smh', 'tbh'
        ])
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = self._remove_urls(text)
        
        # Remove Reddit-specific formatting
        text = self._remove_reddit_formatting(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep some important ones
        text = self._clean_punctuation(text)
        
        # Remove stopwords
        text = self._remove_stopwords(text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _remove_reddit_formatting(self, text: str) -> str:
        """Remove Reddit-specific formatting."""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        
        # Remove quote formatting
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # Remove spoiler formatting
        text = re.sub(r'>!.*?!<', '', text)
        
        # Remove user mentions
        text = re.sub(r'u/[a-zA-Z0-9_-]+', '', text)
        text = re.sub(r'u_[a-zA-Z0-9_-]+', '', text)
        
        # Remove subreddit mentions
        text = re.sub(r'r/[a-zA-Z0-9_-]+', '', text)
        
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean punctuation while preserving sentence structure."""
        # Keep important punctuation for sentence structure
        important_punct = {'.', '!', '?', ',', ';', ':'}
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Remove most punctuation but keep sentence endings
        cleaned_chars = []
        for char in text:
            if char.isalnum() or char.isspace() or char in important_punct:
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(' ')
        
        return ''.join(cleaned_chars)
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess_reddit_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess a list of Reddit posts/comments.
        
        Args:
            data: List of Reddit data dictionaries
            
        Returns:
            List of preprocessed data dictionaries
        """
        processed_data = []
        
        for item in data:
            # Combine title and text for posts
            if item['type'] == 'post':
                raw_text = f"{item.get('title', '')} {item.get('text', '')}"
            else:
                raw_text = item.get('text', '')
            
            # Clean the text
            cleaned_text = self.clean_text(raw_text)
            
            # Skip if text is too short after cleaning
            if len(cleaned_text.split()) < 3:
                continue
            
            # Create processed item
            processed_item = {
                'id': item['id'],
                'type': item['type'],
                'subreddit': item.get('subreddit', ''),
                'author': item.get('author', ''),
                'created_utc': item.get('created_utc'),
                'score': item.get('score', 0),
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }
            
            # Add post-specific fields
            if item['type'] == 'post':
                processed_item.update({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'num_comments': item.get('num_comments', 0),
                    'upvote_ratio': item.get('upvote_ratio', 0)
                })
            else:
                processed_item['post_id'] = item.get('post_id', '')
            
            processed_data.append(processed_item)
        
        logger.info(f"Preprocessed {len(processed_data)} items from {len(data)} raw items")
        return processed_data
    
    def get_text_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the preprocessed text data.
        
        Args:
            data: List of preprocessed data dictionaries
            
        Returns:
            Dictionary of text statistics
        """
        if not data:
            return {}
        
        word_counts = [item['word_count'] for item in data]
        char_counts = [item['char_count'] for item in data]
        
        stats = {
            'total_items': len(data),
            'total_words': sum(word_counts),
            'total_characters': sum(char_counts),
            'avg_words_per_item': sum(word_counts) / len(word_counts) if word_counts else 0,
            'avg_chars_per_item': sum(char_counts) / len(char_counts) if char_counts else 0,
            'min_words': min(word_counts) if word_counts else 0,
            'max_words': max(word_counts) if word_counts else 0,
            'min_chars': min(char_counts) if char_counts else 0,
            'max_chars': max(char_counts) if char_counts else 0
        }
        
        return stats
