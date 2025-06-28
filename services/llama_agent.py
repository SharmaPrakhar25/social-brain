import os
import logging
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")


def generate_contextual_response_ollama(query: str, context_data: List[Dict], intent: str) -> str:
    """
    Generate contextual response using Ollama API with enhanced relevance explanations.
    Args:
        query (str): User's query
        context_data (List[Dict]): Relevant content from ChromaDB
        intent (str): Query intent (summary, recommendation, search, general)
    Returns:
        str: Generated response
    """
    logger.info(f"Starting contextual response generation for query: '{query}'")
    logger.info(f"Intent detected: {intent}")
    logger.info(f"Context data received: {len(context_data)} items")
    
    # Log context data details
    for i, item in enumerate(context_data):
        logger.debug(f"Context item {i+1}: Title='{item.get('title', 'Untitled')}', "
                    f"Category='{item.get('category', 'general')}', "
                    f"Keywords count={len(item.get('keywords', []))}, "
                    f"Content length={len(item.get('content', ''))}")
    
    # Prepare enhanced context with relevance explanations
    logger.info("Processing context data for relevance explanations")
    enhanced_context = ""
    processed_items = min(3, len(context_data))
    logger.info(f"Processing top {processed_items} most relevant items")
    
    for i, item in enumerate(context_data[:3], 1):
        relevance_explanation = item.get('relevance_explanation', '')
        if not relevance_explanation:
            # Fallback: basic explanation
            relevance_explanation = f"This content is related to your query."
            logger.debug(f"Using fallback relevance explanation for item {i}")
        else:
            logger.debug(f"Using provided relevance explanation for item {i}")
            
        enhanced_context += f"""
Content {i}:
Title: {item.get('title', 'Untitled')}
Category: {item.get('category', 'general')}
Keywords: {', '.join(item.get('keywords', [])[:5])}
Why it's relevant: {relevance_explanation}
Summary: {item.get('content', '')[:200]}...

"""
    
    logger.info(f"Enhanced context prepared. Length: {len(enhanced_context)} characters")

    # Create enhanced intent-specific prompts
    logger.info(f"Creating intent-specific prompt for: {intent}")
    
    if intent == 'summary':
        logger.debug("Using summary prompt template")
        prompt = f'''The user asked: "{query}"

Based on their saved content below, provide a comprehensive response that explains WHY each piece of content is relevant and what insights they offer:

{enhanced_context}

Please provide a response that:
1. Directly answers their question using the content
2. Explains WHY each piece of content is relevant to their query
3. Highlights key insights and connections between content pieces
4. Uses specific examples from the content
5. Maintains a conversational, helpful tone

Focus on making connections clear and actionable.

Response:'''
    elif intent == 'recommendation':
        logger.debug("Using recommendation prompt template")
        prompt = f'''The user asked: "{query}"

Based on their content library below, provide thoughtful recommendations with clear explanations:

{enhanced_context}

Please provide:
1. Specific recommendations based on their saved content patterns
2. Clear explanations for WHY these recommendations fit their interests
3. Connections between their saved content and the recommendations
4. Actionable next steps they can take
5. A friendly, advisory tone

Response:'''
    else:  # general and search
        logger.debug("Using general/search prompt template")
        prompt = f'''The user asked: "{query}"

Here's relevant content from their personal library with relevance explanations:

{enhanced_context}

Please provide a helpful response that:
1. Directly addresses their question using the found content
2. Explains WHY each piece of content is relevant (build on the provided explanations)
3. Makes connections between different content pieces
4. Offers additional insights or patterns you notice
5. Maintains a conversational, knowledgeable tone

Make the relevance connections clear and specific.

Response:'''

    logger.info(f"Prompt created. Length: {len(prompt)} characters")
    logger.debug(f"Using Ollama model: {OLLAMA_MODEL}")
    logger.debug(f"Ollama API URL: {OLLAMA_API_URL}")

    try:
        logger.info("Sending request to Ollama API")
        data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False  # Disable streaming to get a single JSON response
        }
        
        logger.debug(f"Request payload size: {len(str(data))} characters")
        
        # Add timeout to prevent hanging requests
        response = requests.post(
            OLLAMA_API_URL, 
            json=data, 
            timeout=60  # 60 second timeout
        )
        
        logger.info(f"Ollama API response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        # Log raw response content for debugging
        raw_content = response.text
        logger.debug(f"Raw response content length: {len(raw_content)} characters")
        logger.debug(f"Raw response preview (first 200 chars): {raw_content[:200]}")
        
        try:
            result = response.json()
            logger.debug(f"Response JSON keys: {list(result.keys())}")
        except ValueError as json_error:
            logger.error(f"JSON parsing failed: {json_error}")
            logger.error(f"Full response content: {raw_content}")
            return "I'm sorry, but I received an invalid response from the AI service."
        
        # Validate response structure
        if "message" not in result:
            logger.error(f"Missing 'message' key in response. Available keys: {list(result.keys())}")
            logger.error(f"Full response: {result}")
            return "I'm sorry, but I received an unexpected response format from the AI service."
        
        if "content" not in result["message"]:
            logger.error(f"Missing 'content' key in message. Available keys: {list(result['message'].keys())}")
            logger.error(f"Full message: {result['message']}")
            return "I'm sorry, but I received an incomplete response from the AI service."
        
        generated_text = result["message"]["content"].strip()
        logger.info(f"Generated text length: {len(generated_text)} characters")
        
        # Clean up the response
        if generated_text:
            logger.debug("Cleaning up generated response")
            cleaned_response = re.sub(r'^(Response:|Answer:)', '', generated_text).strip()
            
            if cleaned_response != generated_text:
                logger.debug("Response prefix cleaned")
            
            logger.info("Successfully generated contextual response")
            return cleaned_response
        else:
            logger.warning("Generated text is empty")
            return "I found some relevant content but couldn't generate a detailed response. Let me show you what I found instead."
            
    except requests.exceptions.Timeout as e:
        logger.error(f"Ollama API request timed out: {e}")
        return "I'm sorry, but the AI service is taking too long to respond. Please try again."
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to Ollama API: {e}")
        return "I'm sorry, but I couldn't connect to the AI service. Please ensure Ollama is running."
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Ollama API request: {e}")
        # Try to get response content if available
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Error response content: {e.response.text}")
        return "I'm sorry, but I couldn't connect to the AI service at the moment."
    except KeyError as e:
        logger.error(f"Unexpected response format from Ollama API: {e}")
        return "I'm sorry, but I received an unexpected response from the AI service."
    except Exception as e:
        logger.error(f"Unexpected error during response generation: {e}")
        return "I'm sorry, but I couldn't generate a response at the moment."

def test_ollama_connection() -> Dict[str, Any]:
    """
    Test Ollama API connection and model availability
    Returns:
        Dict with test results
    """
    logger.info("Testing Ollama connection...")
    
    try:
        # Test basic connectivity
        base_url = OLLAMA_API_URL.replace('/api/chat', '')
        ping_response = requests.get(f"{base_url}/api/tags", timeout=10)
        
        if ping_response.status_code == 200:
            models = ping_response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            logger.info(f"Ollama is running. Available models: {available_models}")
            
            if OLLAMA_MODEL in available_models:
                logger.info(f"Target model '{OLLAMA_MODEL}' is available")
                
                # Test a simple chat request
                test_data = {
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": False
                }
                
                test_response = requests.post(OLLAMA_API_URL, json=test_data, timeout=30)
                
                if test_response.status_code == 200:
                    test_result = test_response.json()
                    if "message" in test_result and "content" in test_result["message"]:
                        logger.info("Ollama API test successful")
                        return {
                            "status": "success",
                            "message": "Ollama connection and model working correctly",
                            "available_models": available_models,
                            "target_model": OLLAMA_MODEL,
                            "test_response": test_result["message"]["content"][:100]
                        }
                    else:
                        logger.error(f"Invalid test response structure: {test_result}")
                        return {
                            "status": "error",
                            "message": "Invalid response structure from Ollama",
                            "details": test_result
                        }
                else:
                    logger.error(f"Test request failed with status {test_response.status_code}")
                    return {
                        "status": "error",
                        "message": f"Test request failed: {test_response.status_code}",
                        "details": test_response.text
                    }
            else:
                logger.error(f"Model '{OLLAMA_MODEL}' not found in available models")
                return {
                    "status": "error",
                    "message": f"Model '{OLLAMA_MODEL}' not available",
                    "available_models": available_models
                }
        else:
            logger.error(f"Failed to connect to Ollama: {ping_response.status_code}")
            return {
                "status": "error",
                "message": f"Failed to connect to Ollama: {ping_response.status_code}",
                "details": ping_response.text
            }
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_API_URL}: {e}")
        return {
            "status": "error",
            "message": "Cannot connect to Ollama. Is it running?",
            "url": OLLAMA_API_URL,
            "details": str(e)
        }
    except Exception as e:
        logger.error(f"Ollama connection test failed: {e}")
        return {
            "status": "error",
            "message": "Ollama connection test failed",
            "details": str(e)
        }