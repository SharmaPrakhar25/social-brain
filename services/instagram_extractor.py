"""
Enhanced Instagram Content Extractor

Specialized extraction for Instagram reels with rich metadata including
hashtags, mentions, engagement metrics, and platform-specific features.
"""

import re
import json
import logging
import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class InstagramExtractor:
    """
    Enhanced Instagram content extractor for comprehensive metadata
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1'
        })
    
    def extract_enhanced_metadata(self, url: str) -> Dict:
        """
        Extract comprehensive Instagram reel metadata
        """
        try:
            # Basic URL validation
            if not self._is_instagram_url(url):
                raise ValueError("Not a valid Instagram URL")
            
            # Extract post shortcode from URL
            shortcode = self._extract_shortcode(url)
            if not shortcode:
                raise ValueError("Could not extract post ID from URL")
            
            # Get enhanced metadata using web scraping approach
            metadata = self._fetch_post_metadata(url, shortcode)
            
            return {
                'platform_type': 'instagram_reel',
                'shortcode': shortcode,
                'caption': metadata.get('caption', ''),
                'hashtags': metadata.get('hashtags', []),
                'mentions': metadata.get('mentions', []),
                'engagement_metrics': metadata.get('engagement', {}),
                'media_type': metadata.get('media_type', 'video'),
                'location': metadata.get('location', ''),
                'music_info': metadata.get('music', {}),
                'creation_timestamp': metadata.get('timestamp', ''),
                'author_info': metadata.get('author', {}),
                'visual_tags': metadata.get('visual_tags', []),
                'content_warnings': metadata.get('warnings', []),
                'enhanced_extraction': True
            }
            
        except Exception as e:
            logger.warning(f"Enhanced Instagram extraction failed: {e}")
            return self._fallback_extraction(url)
    
    def _is_instagram_url(self, url: str) -> bool:
        """Check if URL is from Instagram"""
        domain = urlparse(url).netloc.lower()
        return 'instagram.com' in domain or 'instagr.am' in domain
    
    def _extract_shortcode(self, url: str) -> Optional[str]:
        """Extract Instagram post shortcode from URL"""
        patterns = [
            r'/reel/([A-Za-z0-9_-]+)',
            r'/p/([A-Za-z0-9_-]+)',
            r'/tv/([A-Za-z0-9_-]+)',
            r'/reels/([A-Za-z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _fetch_post_metadata(self, url: str, shortcode: str) -> Dict:
        """
        Fetch comprehensive post metadata using web scraping
        """
        try:
            # Attempt to fetch page content
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
            
            # Extract metadata from page content
            metadata = self._parse_page_content(content, shortcode)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Web scraping failed for {shortcode}: {e}")
            # Return basic structure with extracted caption if possible
            return self._extract_basic_metadata(url)
    
    def _parse_page_content(self, content: str, shortcode: str) -> Dict:
        """
        Parse Instagram page content for metadata
        """
        metadata = {
            'caption': '',
            'hashtags': [],
            'mentions': [],
            'engagement': {},
            'media_type': 'video',
            'location': '',
            'music': {},
            'timestamp': '',
            'author': {},
            'visual_tags': [],
            'warnings': []
        }
        
        try:
            # Extract JSON-LD structured data
            json_ld_match = re.search(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', content, re.DOTALL)
            if json_ld_match:
                try:
                    json_data = json.loads(json_ld_match.group(1))
                    if isinstance(json_data, list):
                        json_data = json_data[0]
                    
                    # Extract from structured data
                    if 'caption' in json_data:
                        metadata['caption'] = json_data['caption']
                    elif 'description' in json_data:
                        metadata['caption'] = json_data['description']
                    
                    if 'author' in json_data:
                        author_data = json_data['author']
                        if isinstance(author_data, dict):
                            metadata['author'] = {
                                'username': author_data.get('alternateName', ''),
                                'display_name': author_data.get('name', ''),
                                'url': author_data.get('url', '')
                            }
                except json.JSONDecodeError:
                    pass
            
            # Extract from meta tags
            meta_description = re.search(r'<meta property="og:description" content="([^"]*)"', content)
            if meta_description and not metadata['caption']:
                metadata['caption'] = meta_description.group(1)
            
            # Extract author from meta tags
            meta_title = re.search(r'<meta property="og:title" content="([^"]*)"', content)
            if meta_title:
                title_text = meta_title.group(1)
                # Instagram titles often follow pattern: "Username on Instagram: caption"
                username_match = re.match(r'^([^:]+) on Instagram:', title_text)
                if username_match:
                    metadata['author']['username'] = username_match.group(1).strip()
            
            # Extract hashtags and mentions from caption
            if metadata['caption']:
                metadata['hashtags'] = self.extract_hashtags(metadata['caption'])
                metadata['mentions'] = self.extract_mentions(metadata['caption'])
            
            # Try to extract from window._sharedData or similar
            shared_data_match = re.search(r'window\._sharedData\s*=\s*({.*?});', content)
            if shared_data_match:
                try:
                    shared_data = json.loads(shared_data_match.group(1))
                    # Navigate the Instagram data structure
                    entry_data = shared_data.get('entry_data', {})
                    post_page = entry_data.get('PostPage', [])
                    if post_page:
                        post_data = post_page[0].get('graphql', {}).get('shortcode_media', {})
                        if post_data:
                            # Extract detailed information
                            edge_text = post_data.get('edge_media_to_caption', {}).get('edges', [])
                            if edge_text:
                                caption_text = edge_text[0].get('node', {}).get('text', '')
                                if caption_text:
                                    metadata['caption'] = caption_text
                                    metadata['hashtags'] = self.extract_hashtags(caption_text)
                                    metadata['mentions'] = self.extract_mentions(caption_text)
                            
                            # Extract owner info
                            owner = post_data.get('owner', {})
                            if owner:
                                metadata['author'] = {
                                    'username': owner.get('username', ''),
                                    'display_name': owner.get('full_name', ''),
                                    'follower_count': owner.get('edge_followed_by', {}).get('count', 0),
                                    'verified': owner.get('is_verified', False)
                                }
                            
                            # Extract engagement
                            metadata['engagement'] = {
                                'likes': post_data.get('edge_media_preview_like', {}).get('count', 0),
                                'comments': post_data.get('edge_media_to_comment', {}).get('count', 0),
                                'views': post_data.get('video_view_count', 0)
                            }
                            
                            # Extract location
                            location_data = post_data.get('location')
                            if location_data:
                                metadata['location'] = location_data.get('name', '')
                            
                            # Extract media type
                            if post_data.get('is_video'):
                                metadata['media_type'] = 'video'
                            elif post_data.get('__typename') == 'GraphSidecar':
                                metadata['media_type'] = 'carousel'
                            else:
                                metadata['media_type'] = 'image'
                                
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.warning(f"Error parsing Instagram page content: {e}")
        
        return metadata
    
    def _extract_basic_metadata(self, url: str) -> Dict:
        """
        Extract basic metadata when advanced parsing fails
        """
        return {
            'platform_type': 'instagram_basic',
            'shortcode': self._extract_shortcode(url) or '',
            'caption': '',
            'hashtags': [],
            'mentions': [],
            'engagement_metrics': {},
            'media_type': 'video',
            'location': '',
            'music_info': {},
            'creation_timestamp': '',
            'author_info': {},
            'visual_tags': [],
            'content_warnings': [],
            'enhanced_extraction': False
        }
    
    def _fallback_extraction(self, url: str) -> Dict:
        """Fallback extraction when enhanced methods fail"""
        return {
            'platform_type': 'instagram_fallback',
            'shortcode': self._extract_shortcode(url) or '',
            'caption': '',
            'hashtags': [],
            'mentions': [],
            'engagement_metrics': {},
            'error': 'Enhanced extraction failed, using fallback',
            'enhanced_extraction': False
        }
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        if not text:
            return []
        
        # Find hashtags (including those with underscores and numbers)
        hashtags = re.findall(r'#[\w\d_]+', text, re.UNICODE)
        
        # Clean and deduplicate
        clean_hashtags = []
        seen = set()
        for tag in hashtags:
            clean_tag = tag.lower()
            if clean_tag not in seen and len(clean_tag) > 1:
                clean_hashtags.append(tag)
                seen.add(clean_tag)
        
        return clean_hashtags
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract user mentions from text"""
        if not text:
            return []
        
        # Find mentions
        mentions = re.findall(r'@[\w\d_.]+', text, re.UNICODE)
        
        # Clean and deduplicate
        clean_mentions = []
        seen = set()
        for mention in mentions:
            clean_mention = mention.lower()
            if clean_mention not in seen and len(clean_mention) > 1:
                clean_mentions.append(mention)
                seen.add(clean_mention)
        
        return clean_mentions
    
    def analyze_caption_sentiment(self, caption: str) -> str:
        """
        Basic sentiment analysis of caption
        """
        if not caption:
            return 'neutral'
        
        positive_indicators = ['love', 'amazing', 'great', 'awesome', 'perfect', '❤️', '😍', '🔥', '💯', '✨']
        negative_indicators = ['hate', 'terrible', 'worst', 'bad', 'awful', '😢', '😭', '💔']
        
        caption_lower = caption.lower()
        positive_count = sum(1 for word in positive_indicators if word in caption_lower)
        negative_count = sum(1 for word in negative_indicators if word in caption_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_topics_from_hashtags(self, hashtags: List[str]) -> List[str]:
        """
        Extract general topics from hashtags
        """
        if not hashtags:
            return []
        
        topic_mapping = {
            'fitness': ['#fitness', '#workout', '#gym', '#training', '#health'],
            'food': ['#food', '#recipe', '#cooking', '#chef', '#delicious'],
            'travel': ['#travel', '#vacation', '#explore', '#adventure', '#wanderlust'],
            'fashion': ['#fashion', '#style', '#outfit', '#ootd', '#clothing'],
            'technology': ['#tech', '#technology', '#ai', '#coding', '#programming'],
            'business': ['#business', '#entrepreneur', '#startup', '#marketing', '#success'],
            'art': ['#art', '#artist', '#drawing', '#painting', '#creative'],
            'music': ['#music', '#song', '#artist', '#concert', '#musician'],
            'lifestyle': ['#lifestyle', '#life', '#daily', '#inspiration', '#motivation']
        }
        
        detected_topics = []
        hashtags_lower = [tag.lower() for tag in hashtags]
        
        for topic, topic_tags in topic_mapping.items():
            if any(tag in hashtags_lower for tag in topic_tags):
                detected_topics.append(topic)
        
        return detected_topics

# Global instance
instagram_extractor = InstagramExtractor()