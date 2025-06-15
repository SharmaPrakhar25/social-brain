"""
Enhanced Chat Service

This module provides intelligent conversational responses about processed content,
including context-aware answers, structured responses, and content recommendations.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from llama_cpp import Llama
from chromadb import PersistentClient
from sqlalchemy.orm import Session
from models.content import Content, ContentKeyword
import json
import re
from datetime import datetime, timedelta
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatService:
    """
    Enhanced chat service with context-aware responses
    """
    
    def __init__(self):
        # Disable ChromaDB telemetry
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        self.llm = self._load_llama_model()
        self.chroma_client = PersistentClient(path="./chroma_db")
        
        # Response templates for different query types
        self.response_templates = {
            'summary': "Based on your saved content, here's what I found:\n\n{content}\n\n**Key Points:**\n{key_points}\n\n**Sources:** {sources}",
            'recommendation': "Here are some recommendations based on your interests:\n\n{recommendations}\n\n**Why these might interest you:** {reasoning}",
            'search': "I found {count} pieces of content related to '{query}':\n\n{results}",
            'stats': "Here's an overview of your content library:\n\n{stats}",
            'general': "{response}\n\n**Related Content:** {related_content}"
        }
    
    def _load_llama_model(self):
        """Load LLaMA model for chat responses"""
        model_path = os.getenv('LLAMA_MODEL_PATH')
        
        if not model_path or not os.path.exists(model_path):
            logger.error("LLaMA model not found for chat service")
            return None
        
        try:
            llm = Llama(
                model_path=model_path, 
                n_ctx=4096,  # Larger context for chat
                n_batch=512,
                verbose=False,  # Disable verbose output
                logits_all=False,
                use_mmap=True,
                use_mlock=False
            )
            logger.info("Chat LLaMA model loaded successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to load chat LLaMA model: {e}")
            return None
    
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the user's query intent
        """
        query_lower = query.lower()
        
        # Intent classification based on keywords
        if any(word in query_lower for word in ['summarize', 'summary', 'what did', 'tell me about']):
            return 'summary'
        elif any(word in query_lower for word in ['recommend', 'suggest', 'what should', 'similar']):
            return 'recommendation'
        elif any(word in query_lower for word in ['find', 'search', 'show me', 'look for']):
            return 'search'
        elif any(word in query_lower for word in ['stats', 'statistics', 'how many', 'overview']):
            return 'stats'
        else:
            return 'general'
    
    def _search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search content using ChromaDB semantic search
        """
        try:
            collection = self.chroma_client.get_or_create_collection("enhanced_content")
            
            # Perform semantic search
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance_score': 1 - distance,  # Convert distance to relevance
                    'content_id': metadata.get('content_id', ''),
                    'title': metadata.get('title', 'Untitled'),
                    'author': metadata.get('author', 'Unknown'),
                    'category': metadata.get('category', 'general'),
                    'url': metadata.get('url', ''),
                    'source_type': metadata.get('source_type', 'unknown'),
                    'keywords': json.loads(metadata.get('keywords', '[]'))
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []
    
    def _get_content_stats(self, db: Session) -> Dict:
        """
        Get content statistics for stats queries
        """
        try:
            total_content = db.query(Content).count()
            
            # Category distribution
            category_stats = db.query(Content.category, db.func.count(Content.id)).group_by(Content.category).all()
            
            # Recent content (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_content = db.query(Content).filter(Content.created_at >= week_ago).count()
            
            # Top keywords
            top_keywords = db.query(ContentKeyword.keyword, ContentKeyword.frequency)\
                           .order_by(ContentKeyword.frequency.desc())\
                           .limit(10).all()
            
            return {
                'total_content': total_content,
                'category_distribution': dict(category_stats),
                'recent_content': recent_content,
                'top_keywords': dict(top_keywords)
            }
            
        except Exception as e:
            logger.error(f"Failed to get content stats: {e}")
            return {}
    
    def _generate_contextual_response(self, query: str, context_data: List[Dict], intent: str) -> str:
        """
        Generate contextual response using LLaMA
        """
        if not self.llm:
            return "I'm sorry, but I'm unable to generate a detailed response right now. However, I can still help you search through your content."
        
        # Prepare context for the LLM
        context_text = ""
        if context_data:
            context_text = "\n\n".join([
                f"Content {i+1}:\nTitle: {item.get('title', 'Untitled')}\nCategory: {item.get('category', 'general')}\nSummary: {item.get('content', '')[:300]}..."
                for i, item in enumerate(context_data[:3])  # Limit to top 3 results
            ])
        
        # Create intent-specific prompts
        if intent == 'summary':
            prompt = f"""Based on the user's saved content below, provide a comprehensive and helpful summary that answers their question: "{query}"

Relevant Content:
{context_text}

Please provide a clear, informative response that:
1. Directly answers the user's question
2. Highlights key insights from the content
3. Mentions specific sources when relevant
4. Uses a conversational, helpful tone

Response:"""
        
        elif intent == 'recommendation':
            prompt = f"""Based on the user's content library and their request: "{query}", provide thoughtful recommendations.

User's Content:
{context_text}

Please provide:
1. Specific recommendations based on their interests
2. Clear reasoning for why these recommendations fit
3. Actionable next steps
4. A friendly, advisory tone

Response:"""
        
        else:  # general
            prompt = f"""The user asked: "{query}"

Here's relevant content from their personal library:
{context_text}

Please provide a helpful, conversational response that:
1. Directly addresses their question
2. References specific content when relevant
3. Offers additional insights or connections
4. Maintains a friendly, knowledgeable tone

Response:"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=500,
                stop=["User:", "Human:", "Response:", "\n\n---"],
                echo=False,
                temperature=0.4
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean up the response
            if generated_text:
                # Remove any remaining prompt artifacts
                cleaned_response = re.sub(r'^(Response:|Answer:)', '', generated_text).strip()
                return cleaned_response
            else:
                return "I found some relevant content but couldn't generate a detailed response. Let me show you what I found instead."
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I found some relevant content for you, though I'm having trouble generating a detailed response right now."
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for display with clickable URLs
        """
        if not results:
            return "No relevant content found."
        
        formatted = []
        for i, result in enumerate(results[:5], 1):
            title = result.get('title', 'Untitled')
            author = result.get('author', 'Unknown')
            category = result.get('category', 'general')
            source_type = result.get('source_type', 'unknown')
            keywords = result.get('keywords', [])
            relevance = result.get('relevance_score', 0)
            url = result.get('url', '')
            
            # Create platform emoji
            platform_emojis = {
                'instagram': '📸',
                'youtube': '📺',
                'tiktok': '🎵',
                'twitter': '🐦',
                'x': '🐦',
                'other': '🔗'
            }
            platform_emoji = platform_emojis.get(source_type.lower(), '🔗')
            
            url_text = f"[🔗 View Original]({url})" if url else "No URL available"
            
            formatted.append(
                f"**{i}. {title}**\n"
                f"   • {platform_emoji} Platform: {source_type.title()}\n"
                f"   • 👤 Author: {author}\n"
                f"   • 🏷️ Category: {category.title()}\n"
                f"   • 🔍 Relevance: {relevance:.1%}\n"
                f"   • 🏷️ Keywords: {', '.join(keywords[:5])}\n"
                f"   • {url_text}\n"
            )
        
        return "\n".join(formatted)
    
    def _format_stats_response(self, stats: Dict) -> str:
        """
        Format statistics for display
        """
        if not stats:
            return "Unable to retrieve statistics at the moment."
        
        total = stats.get('total_content', 0)
        recent = stats.get('recent_content', 0)
        categories = stats.get('category_distribution', {})
        keywords = stats.get('top_keywords', {})
        
        response = f"📊 **Your Content Library Overview**\n\n"
        response += f"• **Total Content:** {total} items\n"
        response += f"• **Added This Week:** {recent} items\n\n"
        
        if categories:
            response += "**Top Categories:**\n"
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                response += f"   • {category.title()}: {count} items\n"
            response += "\n"
        
        if keywords:
            response += "**Trending Keywords:**\n"
            for keyword, freq in list(keywords.items())[:8]:
                response += f"   • {keyword} ({freq}x)\n"
        
        return response
    
    def process_chat_query(self, query: str, db: Session) -> Dict:
        """
        Main method to process chat queries and return structured responses
        """
        try:
            logger.info(f"Processing chat query: {query}")
            
            # Classify query intent
            intent = self._classify_query_intent(query)
            logger.info(f"Detected intent: {intent}")
            
            # Handle different intents
            if intent == 'stats':
                stats = self._get_content_stats(db)
                response_text = self._format_stats_response(stats)
                
                return {
                    'status': 'success',
                    'intent': intent,
                    'response': response_text,
                    'data': stats,
                    'sources': []
                }
            
            else:
                # Search for relevant content
                search_results = self._search_content(query, limit=5)
                
                if not search_results:
                    return {
                        'status': 'success',
                        'intent': intent,
                        'response': f"I couldn't find any content related to '{query}' in your library. Try adding more content or rephrasing your question.",
                        'data': {},
                        'sources': []
                    }
                
                # Generate contextual response
                if intent == 'search':
                    response_text = f"I found {len(search_results)} pieces of content related to '{query}':\n\n"
                    response_text += self._format_search_results(search_results)
                else:
                    response_text = self._generate_contextual_response(query, search_results, intent)
                
                # Prepare sources
                sources = [
                    {
                        'title': result.get('title', 'Untitled'),
                        'author': result.get('author', 'Unknown'),
                        'category': result.get('category', 'general'),
                        'source_type': result.get('source_type', 'unknown'),
                        'url': result.get('url', ''),
                        'relevance': result.get('relevance_score', 0)
                    }
                    for result in search_results[:3]
                ]
                
                return {
                    'status': 'success',
                    'intent': intent,
                    'response': response_text,
                    'data': {
                        'search_results': search_results,
                        'result_count': len(search_results)
                    },
                    'sources': sources
                }
        
        except Exception as e:
            logger.error(f"Chat query processing failed: {e}")
            return {
                'status': 'error',
                'intent': 'unknown',
                'response': f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question.",
                'data': {},
                'sources': []
            }

# Create global instance
chat_service = EnhancedChatService() 