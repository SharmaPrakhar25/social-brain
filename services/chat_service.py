"""
Enhanced Chat Service

This module provides intelligent conversational responses about processed content,
including context-aware answers, structured responses, and content recommendations.
"""

import os
import json
import logging
import re
from typing import Dict, List
from chromadb import PersistentClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
from services.llama_agent import generate_contextual_response_ollama

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedChatService:
    """
    Enhanced chat service with context-aware responses
    """
    
    def __init__(self):
        logger.info("Initializing EnhancedChatService")
        # Set ChromaDB telemetry from environment
        os.environ["ANONYMIZED_TELEMETRY"] = os.getenv("ANONYMIZED_TELEMETRY", "False")
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        logger.info(f"Connecting to ChromaDB at: {chroma_path}")
        self.chroma_client = PersistentClient(path=chroma_path)
        
        # Response templates for different query types
        self.response_templates = {
            'summary': "Based on your saved content, here's what I found:\n\n{content}\n\n**Key Points:**\n{key_points}\n\n**Sources:** {sources}",
            'recommendation': "Here are some recommendations based on your interests:\n\n{recommendations}\n\n**Why these might interest you:** {reasoning}",
            'search': "I found {count} pieces of content related to '{query}':\n\n{results}",
            'stats': "Here's an overview of your content library:\n\n{stats}",
            'general': "{response}\n\n**Related Content:** {related_content}"
        }
        logger.info("EnhancedChatService initialized successfully")
    
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the user's query intent
        """
        logger.debug(f"Classifying intent for query: '{query}'")
        query_lower = query.lower()
        
        # Intent classification based on keywords
        if any(word in query_lower for word in ['summarize', 'summary', 'what did', 'tell me about']):
            intent = 'summary'
        elif any(word in query_lower for word in ['recommend', 'suggest', 'what should', 'similar']):
            intent = 'recommendation'
        elif any(word in query_lower for word in ['find', 'search', 'show me', 'look for']):
            intent = 'search'
        elif any(word in query_lower for word in ['stats', 'statistics', 'how many', 'overview']):
            intent = 'stats'
        else:
            intent = 'general'
            
        logger.info(f"Query intent classified as: {intent}")
        return intent
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms for better search
        """
        logger.debug(f"Expanding query: '{query}'")
        # Basic query expansion
        expanded_queries = [query]
        
        # Add hashtag variations if query doesn't start with #
        if not query.startswith('#'):
            # Check if query might be looking for hashtag content
            hashtag_indicators = ['hashtag', 'tag', 'about', 'related to']
            if any(indicator in query.lower() for indicator in hashtag_indicators):
                logger.debug("Query indicates hashtag search, adding hashtag variations")
                # Extract potential hashtag terms
                words = query.lower().split()
                for word in words:
                    if len(word) > 3 and word not in ['hashtag', 'tag', 'about', 'related', 'find', 'show', 'content']:
                        expanded_queries.append(f"#{word}")
        
        # Add automation-related terms for workflow queries
        automation_terms = {
            'automation': ['workflow', 'n8n', 'zapier', 'automate', 'process'],
            'workflow': ['automation', 'n8n', 'process', 'flow'],
            'n8n': ['automation', 'workflow', 'integration', 'api'],
            'technology': ['tech', 'software', 'digital', 'programming'],
            'business': ['entrepreneur', 'startup', 'marketing', 'strategy']
        }
        
        query_lower = query.lower()
        for main_term, related_terms in automation_terms.items():
            if main_term in query_lower:
                logger.debug(f"Found automation term '{main_term}', adding related terms")
                expanded_queries.extend([f"{query} {term}" for term in related_terms[:2]])
                break
        
        final_queries = expanded_queries[:3]  # Limit to 3 variations
        logger.info(f"Query expanded to {len(final_queries)} variations: {final_queries}")
        return final_queries
    
    def _search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Enhanced content search using ChromaDB with query expansion
        """
        logger.info(f"Starting content search for query: '{query}' with limit: {limit}")
        try:
            collection = self.chroma_client.get_or_create_collection("enhanced_content")
            logger.debug("ChromaDB collection 'enhanced_content' accessed")
            
            # Expand query for better matching
            expanded_queries = self._expand_query(query)
            
            all_results = []
            seen_content_ids = set()
            
            # Search with each expanded query
            for i, search_query in enumerate(expanded_queries, 1):
                logger.debug(f"Searching with query {i}/{len(expanded_queries)}: '{search_query}'")
                try:
                    results = collection.query(
                        query_texts=[search_query],
                        n_results=limit,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        found_count = len(results['documents'][0])
                        logger.debug(f"Found {found_count} results for query: '{search_query}'")
                        
                        # Process results for this query
                        for doc, metadata, distance in zip(
                            results['documents'][0], 
                            results['metadatas'][0], 
                            results['distances'][0]
                        ):
                            content_id = metadata.get('content_id', '')
                            
                            # Avoid duplicates
                            if content_id and content_id not in seen_content_ids:
                                seen_content_ids.add(content_id)
                                relevance_score = 1 - distance
                                
                                logger.debug(f"Processing content_id: {content_id}, relevance_score: {relevance_score:.3f}")
                                
                                # Parse JSON fields safely
                                keywords = []
                                hashtags = []
                                mentions = []
                                
                                try:
                                    keywords = json.loads(metadata.get('keywords', '[]'))
                                except Exception as e:
                                    logger.debug(f"Failed to parse keywords for {content_id}: {e}")
                                    pass
                                
                                try:
                                    hashtags = json.loads(metadata.get('hashtags', '[]'))
                                except Exception as e:
                                    logger.debug(f"Failed to parse hashtags for {content_id}: {e}")
                                    pass
                                
                                try:
                                    mentions = json.loads(metadata.get('mentions', '[]'))
                                except Exception as e:
                                    logger.debug(f"Failed to parse mentions for {content_id}: {e}")
                                    pass
                                
                                all_results.append({
                                    'content': doc,
                                    'metadata': metadata,
                                    'relevance_score': relevance_score,
                                    'content_id': content_id,
                                    'title': metadata.get('title', 'Untitled'),
                                    'author': metadata.get('author', 'Unknown'),
                                    'category': metadata.get('category', 'general'),
                                    'url': metadata.get('url', ''),
                                    'source_type': metadata.get('source_type', 'unknown'),
                                    'keywords': keywords,
                                    'hashtags': hashtags,
                                    'mentions': mentions,
                                    'location': metadata.get('location', ''),
                                    'enhanced_extraction': metadata.get('enhanced_extraction', 'false') == 'true',
                                    'search_query_used': search_query  # Track which query found this
                                })
                            else:
                                logger.debug(f"Skipping duplicate content_id: {content_id}")
                    else:
                        logger.debug(f"No results found for query: '{search_query}'")
                
                except Exception as search_error:
                    logger.warning(f"Search failed for query '{search_query}': {search_error}")
                    continue
            
            # Sort by relevance score and return top results
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            final_results = all_results[:limit]
            
            logger.info(f"Content search completed. Found {len(all_results)} unique results, returning top {len(final_results)}")
            
            # Log top results summary
            for i, result in enumerate(final_results[:3], 1):
                logger.debug(f"Top result {i}: '{result['title']}' (score: {result['relevance_score']:.3f}, "
                           f"category: {result['category']}, author: {result['author']})")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []
    
    def _get_content_stats_file(self) -> dict:
        """
        Get content statistics from the content file
        """
        content_file = os.getenv("CONTENT_FILE", "data/content.json")
        logger.debug(f"Reading content stats from: {content_file}")
        
        if not os.path.exists(content_file):
            logger.warning(f"Content file not found: {content_file}")
            return {"total_content": 0, "latest_title": None}
            
        with open(content_file, "r") as f:
            try:
                content_list = json.load(f)
                logger.debug(f"Loaded {len(content_list)} items from content file")
            except Exception as e:
                logger.error(f"Failed to parse content file: {e}")
                return {"total_content": 0, "latest_title": None}
                
        total = len(content_list)
        latest = max(content_list, key=lambda x: x.get("created_at", ""), default=None)
        
        stats = {
            "total_content": total,
            "latest_title": latest["title"] if latest else None
        }
        logger.info(f"Content stats: {stats}")
        return stats
    
    def _explain_relevance(self, query: str, content_item: Dict) -> str:
        """
        Generate explanation for why content is relevant to the query
        """
        logger.debug(f"Explaining relevance for content: '{content_item.get('title', 'Untitled')}'")
        explanations = []
        query_lower = query.lower()
        
        # Check title relevance
        title = content_item.get('title', '').lower()
        if any(word in title for word in query_lower.split() if len(word) > 2):
            explanations.append("title matches your query")
            logger.debug("Added title match explanation")
        
        # Check keyword relevance
        keywords = content_item.get('keywords', [])
        matching_keywords = [kw for kw in keywords if any(word in kw.lower() for word in query_lower.split() if len(word) > 2)]
        if matching_keywords:
            explanations.append(f"contains keywords: {', '.join(matching_keywords[:3])}")
            logger.debug(f"Added keyword match explanation: {matching_keywords[:3]}")
        
        # Check category relevance
        category = content_item.get('category', '')
        if category.lower() in query_lower:
            explanations.append(f"belongs to {category} category")
            logger.debug(f"Added category match explanation: {category}")
        
        # Check hashtag relevance
        hashtags = content_item.get('hashtags', [])
        matching_hashtags = [tag for tag in hashtags if any(word in tag.lower() for word in query_lower.split() if len(word) > 2)]
        if matching_hashtags:
            explanations.append(f"has relevant hashtags: {', '.join(matching_hashtags[:2])}")
            logger.debug(f"Added hashtag match explanation: {matching_hashtags[:2]}")
        
        # Check search query used
        search_query_used = content_item.get('search_query_used', '')
        if search_query_used and search_query_used != query:
            explanations.append(f"found through related search: '{search_query_used}'")
            logger.debug(f"Added search query explanation: {search_query_used}")
        
        if explanations:
            explanation = f"This content is relevant because it {' and '.join(explanations)}."
        else:
            explanation = "This content matches your query based on semantic similarity."
            
        logger.debug(f"Final relevance explanation: {explanation}")
        return explanation
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for display
        """
        logger.debug(f"Formatting {len(results)} search results for display")
        if not results:
            return "No relevant content found."
        
        formatted_results = []
        for i, item in enumerate(results[:3], 1):
            relevance_explanation = self._explain_relevance("search", item)
            result_text = f"""
**{i}. {item['title']}**
- **Author:** {item['author']}
- **Category:** {item['category']}
- **Source:** {item['source_type']}
- **Relevance:** {relevance_explanation}
- **Keywords:** {', '.join(item['keywords'][:5]) if item['keywords'] else 'None'}
"""
            if item['url']:
                result_text += f"- **URL:** {item['url']}\n"
            
            formatted_results.append(result_text.strip())
        
        logger.debug("Search results formatted successfully")
        return "\n\n".join(formatted_results)
    
    def _format_stats_response(self, stats: dict) -> str:
        """
        Format statistics response
        """
        logger.debug("Formatting stats response")
        total = stats.get('total_content', 0)
        latest = stats.get('latest_title', 'None')
        
        return f"""
**Content Library Overview:**
- **Total Content:** {total} items
- **Latest Content:** {latest if latest else 'None'}
"""
    
    def process_chat_query(self, query: str) -> dict:
        """
        Process a chat query and return structured response
        """
        logger.info(f"Processing chat query: '{query}'")
        start_time = datetime.now()
        
        try:
            # Classify intent
            intent = self._classify_query_intent(query)
            
            # Handle different intents
            if intent == 'stats':
                logger.info("Processing stats query")
                # Get stats from file
                stats = self._get_content_stats_file()
                stats_text = self._format_stats_response(stats)
                
                result = {
                    'status': 'success',
                    'intent': intent,
                    'response': stats_text,
                    'data': stats,
                    'sources': []
                }
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Stats query processed successfully in {processing_time:.2f} seconds")
                return result
            
            else:
                logger.info(f"Processing {intent} query with content search")
                # Search for relevant content
                search_results = self._search_content(query)
                
                if not search_results:
                    logger.warning("No search results found for query")
                    result = {
                        'status': 'success',
                        'intent': intent,
                        'response': "I couldn't find any relevant content in your library for that query. Try adding more content or rephrasing your question.",
                        'data': {},
                        'sources': []
                    }
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Query processed with no results in {processing_time:.2f} seconds")
                    return result
                
                # Add relevance explanations to results
                logger.info("Adding relevance explanations to search results")
                for result in search_results:
                    result['relevance_explanation'] = self._explain_relevance(query, result)
                
                # Generate contextual response using Ollama
                logger.info("Generating AI response using Ollama")
                try:
                    ai_response = generate_contextual_response_ollama(query, search_results, intent)
                    
                    if ai_response and ai_response.strip():
                        response_text = ai_response
                        logger.info("AI response generated successfully")
                    else:
                        logger.warning("AI response was empty, using fallback formatting")
                        # Fallback to formatted search results
                        response_text = self._format_search_results(search_results)
                        
                except Exception as llm_error:
                    logger.error(f"LLM response generation failed: {llm_error}")
                    # Fallback to formatted search results
                    response_text = self._format_search_results(search_results)
                
                # Prepare sources for response
                logger.debug("Preparing source information for response")
                sources = []
                for result in search_results[:3]:
                    sources.append({
                        'title': result['title'],
                        'author': result['author'],
                        'category': result['category'],
                        'url': result['url'],
                        'relevance_score': result['relevance_score'],
                        'content_id': result['content_id']
                    })
                
                final_result = {
                    'status': 'success',
                    'intent': intent,
                    'response': response_text,
                    'data': {
                        'query': query,
                        'results_count': len(search_results),
                        'search_method': 'semantic_search'
                    },
                    'sources': sources
                }
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Query processed successfully in {processing_time:.2f} seconds with {len(search_results)} results")
                return final_result
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Chat query processing failed after {processing_time:.2f} seconds: {e}")
            return {
                'status': 'error',
                'intent': 'unknown',
                'response': f"I encountered an error while processing your query: {str(e)}",
                'data': {},
                'sources': []
            }

# Create global instance
chat_service = EnhancedChatService() 