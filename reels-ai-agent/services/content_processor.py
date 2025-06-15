"""
Enhanced Content Processing Service

This module handles:
- Video/audio transcription using Whisper
- Content summarization using LLaMA
- Keyword extraction and categorization
- Metadata extraction
- Structured data storage
"""

import os
import whisper
import yt_dlp
import uuid
import re
from llama_cpp import Llama
import logging
from typing import Dict, List, Optional, Tuple
import tempfile
import subprocess
import dotenv
from datetime import datetime
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
try:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    whisper_model = whisper.load_model("base", download_root=None, in_memory=False)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

# Global LLM model loading
def load_llama_model():
    model_path = os.getenv('LLAMA_MODEL_PATH')
    logger.info(f"Loading LLaMA model from: {model_path}")
    
    if not model_path or not os.path.exists(model_path):
        logger.error("LLaMA model not found")
        return None
    
    try:
        llm = Llama(
            model_path=model_path, 
            n_ctx=2048,
            n_batch=512,
            verbose=False,  # Disable verbose output
            logits_all=False,
            use_mmap=True,
            use_mlock=False
        )
        logger.info("LLaMA model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLaMA model: {e}")
        return None

# Load the LLaMA model
llm = load_llama_model()

class ContentProcessor:
    """
    Enhanced content processor with structured data extraction
    """
    
    def __init__(self):
        self.whisper_model = whisper_model
        self.llm = llm
        
        # Predefined categories for content classification
        self.categories = [
            'technology', 'business', 'entertainment', 'education', 
            'lifestyle', 'health', 'sports', 'news', 'comedy', 'music',
            'travel', 'food', 'fashion', 'art', 'science', 'politics'
        ]
    
    def extract_metadata_from_url(self, url: str) -> Dict[str, str]:
        """
        Extract metadata from URL using yt-dlp with enhanced title extraction
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Extract and clean title
                title = info.get('title', '')
                if title:
                    # Clean up common title artifacts
                    title = re.sub(r'\s*\|\s*.*$', '', title)  # Remove "| Channel Name" suffix
                    title = re.sub(r'\s*-\s*.*$', '', title)   # Remove "- Channel Name" suffix
                    title = re.sub(r'^\s*\d+\.\s*', '', title)  # Remove leading numbers
                    title = title.strip()
                
                # Extract other metadata
                author = info.get('uploader', '') or info.get('channel', '')
                duration = info.get('duration', 0)
                
                # Try to get upload date
                upload_date = info.get('upload_date', '')
                published_at = ''
                if upload_date:
                    try:
                        from datetime import datetime
                        published_at = datetime.strptime(upload_date, '%Y%m%d').isoformat()
                    except:
                        pass
                
                return {
                    'title': title,
                    'author': author,
                    'platform_id': info.get('id', ''),
                    'duration': duration,
                    'published_at': published_at,
                    'description': info.get('description', '')[:200] if info.get('description') else ''
                }
                
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {
                'title': '',
                'author': '',
                'platform_id': '',
                'duration': 0,
                'published_at': '',
                'description': ''
            }
    
    def download_audio(self, url: str) -> str:
        """
        Download audio from video URL
        """
        try:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s')
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info_dict)
                audio_path = os.path.splitext(video_path)[0] + '.wav'
                
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not created: {audio_path}")
                
                logger.info(f"Successfully downloaded audio to: {audio_path}")
                return audio_path
        
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper
        """
        if not self.whisper_model:
            raise ValueError("Whisper model not loaded")
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise ValueError(f"Invalid audio file: {audio_path}")
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            transcription = result['text'].strip()
            logger.info("Transcription completed successfully")
            return transcription
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords using TF-IDF
        """
        if not text or len(text.strip()) < 10:
            return []
        
        try:
            # Clean text
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            # Use TF-IDF to extract keywords
            vectorizer = TfidfVectorizer(
                max_features=max_keywords * 2,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            keywords = [kw for kw, score in keyword_scores[:max_keywords] if score > 0]
            
            logger.info(f"Extracted {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def categorize_content(self, text: str, keywords: List[str]) -> str:
        """
        Categorize content based on text and keywords
        """
        if not text and not keywords:
            return 'general'
        
        # Combine text and keywords for analysis
        combined_text = f"{text} {' '.join(keywords)}".lower()
        
        # Simple keyword-based categorization
        category_keywords = {
            'technology': ['tech', 'software', 'ai', 'computer', 'digital', 'app', 'coding', 'programming'],
            'business': ['business', 'entrepreneur', 'startup', 'money', 'finance', 'investment', 'marketing'],
            'entertainment': ['movie', 'film', 'show', 'celebrity', 'entertainment', 'fun', 'funny'],
            'education': ['learn', 'education', 'tutorial', 'teach', 'study', 'course', 'lesson'],
            'health': ['health', 'fitness', 'workout', 'diet', 'medical', 'wellness', 'exercise'],
            'lifestyle': ['lifestyle', 'life', 'daily', 'routine', 'personal', 'home', 'family'],
            'music': ['music', 'song', 'artist', 'album', 'concert', 'band', 'singer'],
            'sports': ['sport', 'game', 'team', 'player', 'match', 'football', 'basketball'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'eat', 'meal', 'chef'],
            'travel': ['travel', 'trip', 'vacation', 'destination', 'hotel', 'flight', 'tourism']
        }
        
        category_scores = {}
        for category, keywords_list in category_keywords.items():
            score = sum(1 for keyword in keywords_list if keyword in combined_text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Simple sentiment analysis
        """
        if not text:
            return 'neutral'
        
        positive_words = ['good', 'great', 'awesome', 'amazing', 'excellent', 'love', 'best', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def generate_summary(self, text: str) -> str:
        """
        Generate summary using LLaMA with improved prompting
        """
        if not self.llm:
            return "AI summarization not available"
        
        if not text or len(text.strip()) < 50:
            return "Content too short to summarize"
        
        # Enhanced prompt for better summaries
        prompt = f"""Please provide a concise, informative summary of the following content. Focus on the main points, key insights, and actionable information:

Content: {text}

Summary:"""
        
        try:
            response = self.llm(
                prompt, 
                max_tokens=300, 
                stop=["Content:", "Summary:", "\n\n"], 
                echo=False,
                temperature=0.3
            )
            summary = response['choices'][0]['text'].strip()
            
            # Clean up the summary
            if summary:
                # Remove any remaining prompt artifacts
                summary = re.sub(r'^(Summary:|Content:)', '', summary).strip()
                return summary
            else:
                return "Failed to generate summary"
                
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Summary generation failed"
    
    def detect_source_type(self, url: str) -> str:
        """
        Detect content source type from URL
        """
        domain = urlparse(url).netloc.lower()
        
        if 'instagram.com' in domain:
            return 'instagram'
        elif 'twitter.com' in domain or 'x.com' in domain:
            return 'twitter'
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return 'youtube'
        elif 'tiktok.com' in domain:
            return 'tiktok'
        else:
            return 'other'
    
    def process_content(self, url: str) -> Dict:
        """
        Main processing function that handles the entire pipeline
        """
        audio_path = None
        
        try:
            logger.info(f"Starting content processing for URL: {url}")
            
            # Extract basic metadata
            metadata = self.extract_metadata_from_url(url)
            source_type = self.detect_source_type(url)
            
            # Download and transcribe audio
            audio_path = self.download_audio(url)
            transcription = self.transcribe_audio(audio_path)
            
            # Extract keywords and analyze content
            keywords = self.extract_keywords(transcription)
            category = self.categorize_content(transcription, keywords)
            sentiment = self.analyze_sentiment(transcription)
            
            # Generate enhanced summary
            summary = self.generate_summary(transcription)
            
            # Generate title based on summary and keywords
            title = self.generate_title(summary, keywords, transcription)
            # If no title from metadata, use the generated one
            if not metadata.get('title') or metadata.get('title').strip() == '':
                metadata['title'] = title
            
            # Prepare structured result
            result = {
                'source_type': source_type,
                'original_url': url,
                'title': metadata.get('title', title),  # Use generated title as fallback
                'author': metadata.get('author', ''),
                'platform_id': metadata.get('platform_id', ''),
                'duration': metadata.get('duration', 0),
                'transcription': transcription,
                'summary': summary,
                'keywords': keywords,
                'category': category,
                'sentiment': sentiment,
                'content_type': 'video',
                'language': 'en',  # TODO: Add language detection
                'processing_status': 'completed',
                'published_at': metadata.get('published_at', '')
            }
            
            logger.info("Content processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            
            # Try to extract at least basic metadata for failed processing
            try:
                metadata = self.extract_metadata_from_url(url)
                fallback_title = metadata.get('title', '') or self._generate_fallback_title("", [], "")
            except:
                fallback_title = "Failed to Process Content"
            
            return {
                'source_type': self.detect_source_type(url),
                'original_url': url,
                'title': fallback_title,
                'processing_status': 'failed',
                'error_message': str(e),
                'transcription': '',
                'summary': '',
                'keywords': [],
                'category': 'general',
                'sentiment': 'neutral'
            }
        
        finally:
            # Clean up temporary files
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    # Also remove the temp directory
                    temp_dir = os.path.dirname(audio_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                    logger.info("Cleaned up temporary files")
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")

    def generate_title(self, summary: str, keywords: List[str], transcription: str = "") -> str:
        """
        Generate an engaging title based on summary, keywords, and transcription
        """
        if not self.llm:
            # Fallback title generation without AI
            return self._generate_fallback_title(summary, keywords, transcription)
        
        # Prepare content for title generation
        content_parts = []
        if summary and len(summary.strip()) > 10:
            content_parts.append(f"Summary: {summary[:200]}")
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords[:8])}")
        if transcription and len(transcription.strip()) > 20:
            content_parts.append(f"Key content: {transcription[:150]}")
        
        if not content_parts:
            return "Untitled Content"
        
        content_text = "\n".join(content_parts)
        
        # Enhanced prompt for title generation
        prompt = f"""Based on the following content information, create a concise, engaging title (maximum 8 words) that captures the main topic or key insight:

{content_text}

Generate a clear, descriptive title that would help someone quickly understand what this content is about. Focus on the main topic, key insight, or actionable information.

Title:"""
        
        try:
            response = self.llm(
                prompt, 
                max_tokens=50, 
                stop=["Content:", "Summary:", "Keywords:", "\n\n"], 
                echo=False,
                temperature=0.4
            )
            title = response['choices'][0]['text'].strip()
            
            # Clean up the title
            if title:
                # Remove any remaining prompt artifacts
                title = re.sub(r'^(Title:|Content:|Summary:)', '', title).strip()
                # Remove quotes if present
                title = title.strip('"\'')
                # Ensure reasonable length
                if len(title) > 80:
                    title = title[:77] + "..."
                # Capitalize first letter
                title = title[0].upper() + title[1:] if title else ""
                
                return title if title else self._generate_fallback_title(summary, keywords, transcription)
            else:
                return self._generate_fallback_title(summary, keywords, transcription)
                
        except Exception as e:
            logger.error(f"Title generation error: {e}")
            return self._generate_fallback_title(summary, keywords, transcription)
    
    def _generate_fallback_title(self, summary: str, keywords: List[str], transcription: str = "") -> str:
        """
        Generate a fallback title without AI when LLaMA is not available
        """
        # Try to extract title from summary first
        if summary and len(summary.strip()) > 10:
            # Take first sentence or first 60 characters
            first_sentence = summary.split('.')[0].strip()
            if len(first_sentence) > 10 and len(first_sentence) <= 60:
                return first_sentence
            elif len(summary) <= 60:
                return summary.strip()
            else:
                return summary[:57].strip() + "..."
        
        # Use keywords to create title
        if keywords:
            if len(keywords) >= 3:
                return f"{keywords[0].title()} & {keywords[1].title()} Tips"
            elif len(keywords) >= 2:
                return f"{keywords[0].title()} and {keywords[1].title()}"
            else:
                return f"{keywords[0].title()} Content"
        
        # Last resort: try to extract from transcription
        if transcription and len(transcription.strip()) > 20:
            # Look for common patterns that might indicate a title
            lines = transcription.split('\n')
            for line in lines[:3]:  # Check first 3 lines
                line = line.strip()
                if len(line) > 10 and len(line) <= 60:
                    return line
            
            # Take first 50 characters
            return transcription[:47].strip() + "..."
        
        return "Untitled Content"

# Create global instance
content_processor = ContentProcessor() 