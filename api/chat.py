"""
Enhanced Chat API Endpoint

This module provides intelligent conversational interface for querying processed content
with context-aware responses, intent classification, and structured output.
"""

import os
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import chat_service
from services.llama_agent import test_ollama_connection
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for chat queries"""
    message: str
    conversation_id: Optional[str] = None  # For future conversation tracking

class ChatResponse(BaseModel):
    """Response model for chat queries"""
    status: str
    intent: str
    response: str
    data: Dict
    sources: List[Dict]
    conversation_id: Optional[str] = None

@router.post("/chat/", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Enhanced chat endpoint with intelligent response generation.
    
    Args:
        request (ChatRequest): Chat request containing user message
    
    Returns:
        ChatResponse: Structured response with intent, content, and sources
    
    Raises:
        HTTPException: If chat processing fails
    """
    import time
    start_time = time.time()
    
    logger.info(f"=== CHAT API REQUEST START ===")
    logger.info(f"Message: '{request.message}'")
    logger.info(f"Conversation ID: {request.conversation_id}")
    logger.info(f"Message length: {len(request.message)} characters")
    
    try:
        # Validate input
        if not request.message or not request.message.strip():
            logger.warning("Empty message received")
            raise HTTPException(
                status_code=400, 
                detail="Message cannot be empty"
            )
        
        cleaned_message = request.message.strip()
        logger.debug(f"Cleaned message: '{cleaned_message}'")
        
        # Process the chat query
        logger.info("Forwarding request to chat service")
        result = chat_service.process_chat_query(cleaned_message)
        
        processing_time = time.time() - start_time
        logger.info(f"Chat service processing completed in {processing_time:.2f} seconds")
        
        # Log result summary
        logger.info(f"Result status: {result.get('status')}")
        logger.info(f"Detected intent: {result.get('intent')}")
        logger.info(f"Response length: {len(result.get('response', ''))} characters")
        logger.info(f"Number of sources: {len(result.get('sources', []))}")
        
        # Handle processing errors
        if result.get('status') == 'error':
            logger.error(f"Chat processing error: {result.get('response', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result.get('response', 'Failed to process chat query')
            )
        
        # Log successful processing details
        data = result.get('data', {})
        if 'results_count' in data:
            logger.info(f"Search results found: {data['results_count']}")
        if 'search_method' in data:
            logger.info(f"Search method used: {data['search_method']}")
        
        # Log source details
        sources = result.get('sources', [])
        for i, source in enumerate(sources, 1):
            logger.debug(f"Source {i}: '{source.get('title', 'Untitled')}' "
                        f"(category: {source.get('category', 'unknown')}, "
                        f"relevance: {source.get('relevance_score', 0):.3f})")
        
        # Return structured response
        response = ChatResponse(
            status=result['status'],
            intent=result['intent'],
            response=result['response'],
            data=result['data'],
            sources=result['sources'],
            conversation_id=request.conversation_id
        )
        
        total_time = time.time() - start_time
        logger.info(f"=== CHAT API REQUEST COMPLETED in {total_time:.2f} seconds ===")
        
        return response
        
    except HTTPException as http_ex:
        total_time = time.time() - start_time
        logger.error(f"HTTP Exception after {total_time:.2f} seconds: {http_ex.detail}")
        logger.info(f"=== CHAT API REQUEST FAILED ===")
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Unexpected error after {total_time:.2f} seconds: {str(e)}")
        logger.info(f"=== CHAT API REQUEST ERROR ===")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

def load_content_from_file() -> List[Dict]:
    """Load content from JSON file"""
    content_file = os.getenv("CONTENT_FILE", "data/content.json")
    logger.debug(f"Loading content from file: {content_file}")
    
    if not os.path.exists(content_file):
        logger.warning(f"Content file not found: {content_file}")
        return []
    try:
        with open(content_file, "r") as f:
            content = json.load(f)
            logger.debug(f"Loaded {len(content)} items from content file")
            return content
    except Exception as e:
        logger.error(f"Failed to load content file: {e}")
        return []

@router.get("/chat/suggestions/")
def get_chat_suggestions() -> Dict:
    """
    Get suggested questions based on user's content library
    """
    logger.info("Generating chat suggestions")
    try:
        content_list = load_content_from_file()
        total_content = len(content_list)
        logger.info(f"Content library has {total_content} items")
        
        if total_content == 0:
            logger.info("Empty content library, returning default suggestions")
            return {
                "status": "success",
                "suggestions": [
                    "Add some content first by processing Instagram Reels or other videos!",
                    "Try the /process/ endpoint to get started."
                ]
            }
        
        # Get top categories and keywords for suggestions
        categories = {}
        all_keywords = []
        
        for content in content_list:
            category = content.get('category', 'general')
            categories[category] = categories.get(category, 0) + 1
            
            # Extract keywords if they exist
            if 'keywords' in content and isinstance(content['keywords'], list):
                all_keywords.extend(content['keywords'])
        
        # Get top categories
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.debug(f"Top categories: {top_categories}")
        
        # Get top keywords (simple frequency count)
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.debug(f"Top keywords: {top_keywords}")
        
        # Generate dynamic suggestions
        suggestions = [
            "What's an overview of my content library?",
            "Show me my recent content",
        ]
        
        # Add category-based suggestions
        for category, count in top_categories:
            suggestions.append(f"Tell me about my {category} content")
            suggestions.append(f"What are the key insights from my {category} videos?")
        
        # Add keyword-based suggestions
        for keyword, freq in top_keywords[:3]:
            suggestions.append(f"Find content about {keyword}")
        
        # Add general suggestions
        suggestions.extend([
            "Recommend similar content to what I've saved",
            "What are the main themes in my content?",
            "Show me content from this week"
        ])
        
        final_suggestions = suggestions[:8]  # Limit to 8 suggestions
        logger.info(f"Generated {len(final_suggestions)} suggestions")
        
        return {
            "status": "success",
            "suggestions": final_suggestions
        }
        
    except Exception as e:
        logger.error(f"Error generating chat suggestions: {str(e)}")
        return {
            "status": "error",
            "suggestions": [
                "What's in my content library?",
                "Show me recent content",
                "Find content about technology",
                "Give me an overview of my saved content"
            ]
        }

@router.get("/chat/intents/")
def get_supported_intents() -> Dict:
    """
    Get information about supported chat intents and example queries
    """
    logger.info("Returning supported chat intents")
    return {
        "status": "success",
        "intents": {
            "summary": {
                "description": "Get summaries and insights from your content",
                "examples": [
                    "Summarize my content about AI",
                    "What did I learn from my business videos?",
                    "Tell me about the main points from my saved content"
                ]
            },
            "search": {
                "description": "Find specific content in your library",
                "examples": [
                    "Find content about machine learning",
                    "Show me videos by John Doe",
                    "Look for content in the technology category"
                ]
            },
            "recommendation": {
                "description": "Get recommendations based on your interests",
                "examples": [
                    "Recommend similar content to what I've saved",
                    "What should I watch next?",
                    "Suggest content based on my interests"
                ]
            },
            "stats": {
                "description": "Get statistics about your content library",
                "examples": [
                    "How many videos do I have?",
                    "What's my content overview?",
                    "Show me my library statistics"
                ]
            }
        }
    }

@router.post("/chat/feedback/")
def submit_chat_feedback(
    response_helpful: bool,
    response_id: Optional[str] = None,
    feedback_text: Optional[str] = None
) -> Dict:
    """
    Submit feedback for chat responses (for future improvements)
    """
    logger.info(f"Feedback received: helpful={response_helpful}, response_id={response_id}")
    if feedback_text:
        logger.info(f"Feedback text: '{feedback_text}'")
    
    try:
        # Log feedback for analysis
        logger.info(f"Chat feedback recorded successfully")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!"
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to submit feedback"
        }

@router.get("/chat/test-ollama/")
def test_ollama_endpoint() -> Dict:
    """
    Test Ollama connection and model availability for debugging
    """
    logger.info("Ollama connection test requested via API")
    
    try:
        result = test_ollama_connection()
        logger.info(f"Ollama test result: {result['status']}")
        return result
    except Exception as e:
        logger.error(f"Error during Ollama test: {str(e)}")
        return {
            "status": "error", 
            "message": f"Test failed with error: {str(e)}"
        }