import re
import string
import nltk
import spacy
import textstat
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter
import logging

from config import config, logger

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("Could not download NLTK data. Some features may not work.")

class TextPreprocessor:
    """Advanced text preprocessing for essay scoring."""
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using spaCy."""
        if not self.nlp:
            return text
        
        doc = self.nlp(text)
        lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        return ' '.join(lemmatized)
    
    def preprocess(self, text: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text

class FeatureExtractor:
    """Extract comprehensive features from essays."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def extract_basic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features."""
        if not text:
            return {}
        
        # Word and character counts
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'word_count': len(words),
            'char_count': len(text),
            'char_count_no_spaces': len(text.replace(' ', '')),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_words_per_sentence': len(words) / max(len([s for s in sentences if s.strip()]), 1),
            'avg_chars_per_word': len(text) / max(len(words), 1),
        }
        
        return features
    
    def extract_readability_features(self, text: str) -> Dict[str, Any]:
        """Extract readability features."""
        if not text:
            return {}
        
        try:
            features = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'difficult_words': textstat.difficult_words(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'text_standard': textstat.text_standard(text),
            }
        except Exception as e:
            logger.warning(f"Error calculating readability features: {e}")
            features = {}
        
        return features
    
    def extract_vocabulary_features(self, text: str) -> Dict[str, Any]:
        """Extract vocabulary-related features."""
        if not text:
            return {}
        
        words = text.lower().split()
        word_counts = Counter(words)
        
        # Unique words
        unique_words = len(set(words))
        total_words = len(words)
        
        # Vocabulary diversity
        lexical_diversity = unique_words / max(total_words, 1)
        
        # Word length statistics
        word_lengths = [len(word) for word in words]
        
        features = {
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity,
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'max_word_length': max(word_lengths) if word_lengths else 0,
            'min_word_length': min(word_lengths) if word_lengths else 0,
            'word_length_std': np.std(word_lengths) if word_lengths else 0,
        }
        
        # Most common words
        most_common = word_counts.most_common(10)
        features['most_common_word'] = most_common[0][0] if most_common else ''
        features['most_common_word_freq'] = most_common[0][1] if most_common else 0
        
        return features
    
    def extract_grammar_features(self, text: str) -> Dict[str, Any]:
        """Extract grammar and syntax features."""
        if not text or not self.preprocessor.nlp:
            return {}
        
        try:
            doc = self.preprocessor.nlp(text)
            
            # POS tag counts
            pos_counts = Counter([token.pos_ for token in doc])
            
            # Named entities
            entities = [ent.label_ for ent in doc.ents]
            entity_counts = Counter(entities)
            
            features = {
                'noun_count': pos_counts.get('NOUN', 0),
                'verb_count': pos_counts.get('VERB', 0),
                'adjective_count': pos_counts.get('ADJ', 0),
                'adverb_count': pos_counts.get('ADV', 0),
                'pronoun_count': pos_counts.get('PRON', 0),
                'conjunction_count': pos_counts.get('CONJ', 0),
                'preposition_count': pos_counts.get('ADP', 0),
                'determiner_count': pos_counts.get('DET', 0),
                'punctuation_count': pos_counts.get('PUNCT', 0),
                'entity_count': len(entities),
                'unique_entities': len(set(entities)),
            }
            
            # Calculate ratios
            total_tokens = len(doc)
            if total_tokens > 0:
                features.update({
                    'noun_ratio': features['noun_count'] / total_tokens,
                    'verb_ratio': features['verb_count'] / total_tokens,
                    'adjective_ratio': features['adjective_count'] / total_tokens,
                    'adverb_ratio': features['adverb_count'] / total_tokens,
                })
            
        except Exception as e:
            logger.warning(f"Error calculating grammar features: {e}")
            features = {}
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment-related features."""
        if not text or not self.preprocessor.nlp:
            return {}
        
        try:
            doc = self.preprocessor.nlp(text)
            
            # Sentiment analysis (simplified)
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'beneficial']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'harmful', 'damaging', 'destructive']
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            features = {
                'positive_word_count': positive_count,
                'negative_word_count': negative_count,
                'sentiment_score': (positive_count - negative_count) / max(len(words), 1),
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment features: {e}")
            features = {}
        
        return features
    
    def extract_structure_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features."""
        if not text:
            return {}
        
        features = {
            'paragraph_count': len(text.split('\n\n')),
            'question_count': text.count('?'),
            'exclamation_count': text.count('!'),
            'comma_count': text.count(','),
            'semicolon_count': text.count(';'),
            'colon_count': text.count(':'),
            'quotation_count': text.count('"') + text.count("'"),
            'parentheses_count': text.count('(') + text.count(')'),
            'dash_count': text.count('-'),
        }
        
        # Calculate ratios
        word_count = len(text.split())
        if word_count > 0:
            features.update({
                'question_ratio': features['question_count'] / word_count,
                'exclamation_ratio': features['exclamation_count'] / word_count,
                'comma_ratio': features['comma_count'] / word_count,
            })
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, Any]:
        """Extract all available features."""
        if not text:
            return {}
        
        features = {}
        
        # Extract different feature categories
        features.update(self.extract_basic_features(text))
        features.update(self.extract_readability_features(text))
        features.update(self.extract_vocabulary_features(text))
        features.update(self.extract_grammar_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_structure_features(text))
        
        return features
    
    def preprocess_and_extract_features(self, text: str) -> Dict[str, Any]:
        """Preprocess text and extract all features."""
        # Clean text
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Extract features from both original and cleaned text
        original_features = self.extract_all_features(text)
        cleaned_features = self.extract_all_features(cleaned_text)
        
        # Combine features with prefixes
        all_features = {}
        for key, value in original_features.items():
            all_features[f'original_{key}'] = value
        
        for key, value in cleaned_features.items():
            all_features[f'cleaned_{key}'] = value
        
        return all_features

def analyze_essay_features(essays: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze features for a list of essays."""
    extractor = FeatureExtractor()
    
    feature_data = []
    
    for essay in essays:
        features = extractor.preprocess_and_extract_features(essay['content'])
        features['essay_id'] = essay['id']
        features['title'] = essay['title']
        features['score'] = essay['score']
        features['grade_level'] = essay['grade_level']
        feature_data.append(features)
    
    return pd.DataFrame(feature_data)

def get_feature_importance(features_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate feature importance based on correlation with scores."""
    # Select only numeric features
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['essay_id', 'score']]
    
    correlations = {}
    for feature in numeric_features:
        try:
            corr = features_df[feature].corr(features_df['score'])
            if not np.isnan(corr):
                correlations[feature] = abs(corr)
        except:
            continue
    
    # Sort by correlation strength
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_features[:20])  # Top 20 most important features

if __name__ == "__main__":
    # Test feature extraction
    from database import EssayDatabase
    
    # Initialize database
    db = EssayDatabase()
    essays = db.get_all_essays()
    
    if essays:
        # Analyze features
        features_df = analyze_essay_features(essays)
        
        print(f"\n📊 Feature Analysis Results:")
        print(f"   Total essays analyzed: {len(essays)}")
        print(f"   Total features extracted: {len(features_df.columns) - 4}")  # Exclude ID, title, score, grade_level
        
        # Get feature importance
        importance = get_feature_importance(features_df)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (feature, importance_score) in enumerate(list(importance.items())[:10]):
            print(f"   {i+1}. {feature}: {importance_score:.3f}")
        
        # Show sample features for first essay
        if len(features_df) > 0:
            sample_features = features_df.iloc[0]
            print(f"\n📝 Sample Features for '{sample_features['title']}':")
            print(f"   Word Count: {sample_features.get('original_word_count', 'N/A')}")
            print(f"   Flesch Reading Ease: {sample_features.get('original_flesch_reading_ease', 'N/A'):.1f}")
            print(f"   Lexical Diversity: {sample_features.get('original_lexical_diversity', 'N/A'):.3f}")
            print(f"   Average Word Length: {sample_features.get('original_avg_word_length', 'N/A'):.1f}")
            print(f"   Noun Ratio: {sample_features.get('original_noun_ratio', 'N/A'):.3f}")
            print(f"   Verb Ratio: {sample_features.get('original_verb_ratio', 'N/A'):.3f}")
    else:
        print("No essays found in database. Please run database.py first to initialize with mock data.")
