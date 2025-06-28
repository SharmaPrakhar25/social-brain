import os
import logging
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    # Prepare enhanced context with relevance explanations
    enhanced_context = ""
    for i, item in enumerate(context_data[:3], 1):
        relevance_explanation = item.get('relevance_explanation', '')
        if not relevance_explanation:
            # Fallback: basic explanation
            relevance_explanation = f"This content is related to your query."
        enhanced_context += f"""
Content {i}:
Title: {item.get('title', 'Untitled')}
Category: {item.get('category', 'general')}
Keywords: {', '.join(item.get('keywords', [])[:5])}
Why it's relevant: {relevance_explanation}
Summary: {item.get('content', '')[:200]}...

"""

    # Create enhanced intent-specific prompts
    if intent == 'summary':
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

    try:
        data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result["message"]["content"].strip()
        
        # Clean up the response
        if generated_text:
            cleaned_response = re.sub(r'^(Response:|Answer:)', '', generated_text).strip()
            return cleaned_response
        else:
            return "I found some relevant content but couldn't generate a detailed response. Let me show you what I found instead."
    except Exception as e:
        logger.error(f"Ollama HTTP response generation error: {e}")
        return "I'm sorry, but I couldn't generate a response at the moment."