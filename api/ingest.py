"""
Enhanced Video Ingestion and Processing Endpoint

This module handles:
- Content processing with structured data extraction
- File-based storage with JSON
- Keyword extraction and categorization
- ChromaDB integration for semantic search
"""

import os
import json
import logging
from fastapi import APIRouter, Form, HTTPException
from chromadb import PersistentClient
from services.content_processor import content_processor
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Set ChromaDB telemetry from environment
os.environ["ANONYMIZED_TELEMETRY"] = os.getenv("ANONYMIZED_TELEMETRY", "False")

# Initialize ChromaDB client
chroma_client = PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db"))

# File-based storage configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
CONTENT_FILE = os.getenv("CONTENT_FILE", "data/content.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Helper functions for file-based storage

def load_all_content() -> List[Dict]:
    if not os.path.exists(CONTENT_FILE):
        return []
    with open(CONTENT_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def save_all_content(content_list: List[Dict]):
    with open(CONTENT_FILE, "w") as f:
        json.dump(content_list, f, indent=2, default=str)

def find_content_by_url(url: str) -> Dict:
    content_list = load_all_content()
    for item in content_list:
        if item.get("original_url") == url:
            return item
    return None

def find_content_by_id(content_id: str) -> Dict:
    content_list = load_all_content()
    for item in content_list:
        if str(item.get("id")) == str(content_id):
            return item
    return None

def store_in_chromadb(content_data: Dict, content_id: str):
    try:
        collection = chroma_client.get_or_create_collection("enhanced_content")
        hashtags_text = ' '.join(content_data.get('hashtags', []))
        mentions_text = ' '.join(content_data.get('mentions', []))
        document_text = f"""
        Title: {content_data.get('title', '')}
        Author: {content_data.get('author', '')}
        Category: {content_data.get('category', '')}
        Summary: {content_data.get('summary', '')}
        Keywords: {', '.join(content_data.get('keywords', []))}
        Hashtags: {hashtags_text}
        Mentions: {mentions_text}
        Location: {content_data.get('location', '')}
        Transcription: {content_data.get('transcription', '')}
        """
        metadata = {
            "content_id": content_id,
            "source_type": content_data.get('source_type', ''),
            "title": content_data.get('title', ''),
            "category": content_data.get('category', ''),
            "sentiment": content_data.get('sentiment', ''),
            "author": content_data.get('author', ''),
            "duration": content_data.get('duration', 0),
            "keywords": json.dumps(content_data.get('keywords', [])),
            "hashtags": json.dumps(content_data.get('hashtags', [])),
            "mentions": json.dumps(content_data.get('mentions', [])),
            "location": content_data.get('location', ''),
            "enhanced_extraction": str(content_data.get('enhanced_extraction', False)),
            "url": content_data.get('original_url', '')
        }
        collection.add(
            documents=[document_text.strip()],
            metadatas=[metadata],
            ids=[content_id]
        )
        logger.info(f"Successfully stored content {content_id} in ChromaDB with enhanced metadata")
    except Exception as e:
        logger.error(f"Failed to store in ChromaDB: {e}")

@router.post("/process/")
def process_content_endpoint(url: str = Form(...)) -> Dict:
    try:
        logger.info(f"Processing content from URL: {url}")
        # Check if content already exists
        existing_content = find_content_by_url(url)
        if existing_content:
            logger.info(f"Content already exists with ID: {existing_content['id']}")
            return {
                "status": "already_exists",
                "content_id": existing_content['id'],
                "message": "Content has already been processed",
                "data": existing_content
            }
        # Process the content
        processed_data = content_processor.process_content(url)
        # Handle processing failure
        if processed_data.get('processing_status') == 'failed':
            failed_content = {
                "id": str(int(datetime.now().timestamp() * 1000)),
                "source_type": processed_data.get('source_type', 'unknown'),
                "original_url": url,
                "title": processed_data.get('title', 'Failed to Process Content'),
                "processing_status": 'failed',
                "error_message": processed_data.get('error_message', 'Unknown error'),
                "content_type": 'video',
                "created_at": datetime.now().isoformat()
            }
            content_list = load_all_content()
            content_list.append(failed_content)
            save_all_content(content_list)
            raise HTTPException(
                status_code=500,
                detail=f"Content processing failed: {processed_data.get('error_message', 'Unknown error')}"
            )
        # Assign a unique ID
        content_id = str(int(datetime.now().timestamp() * 1000))
        content = {
            "id": content_id,
            "source_type": processed_data['source_type'],
            "original_url": processed_data['original_url'],
            "title": processed_data.get('title', ''),
            "author": processed_data.get('author', ''),
            "platform_id": processed_data.get('platform_id', ''),
            "transcription": processed_data['transcription'],
            "summary": processed_data['summary'],
            "content_type": processed_data.get('content_type', 'video'),
            "duration": processed_data.get('duration', 0),
            "category": processed_data['category'],
            "sentiment": processed_data['sentiment'],
            "language": processed_data.get('language', 'en'),
            "processing_status": 'completed',
            "hashtags": processed_data.get('hashtags', []),
            "mentions": processed_data.get('mentions', []),
            "engagement_data": processed_data.get('engagement_metrics', {}),
            "location_data": processed_data.get('location', ''),
            "music_data": processed_data.get('music_info', {}),
            "author_data": processed_data.get('author_info', {}),
            "visual_tags": processed_data.get('visual_tags', []),
            "content_warnings": processed_data.get('content_warnings', []),
            "enhanced_extraction": processed_data.get('enhanced_extraction', False),
            "created_at": datetime.now().isoformat()
        }
        content_list = load_all_content()
        content_list.append(content)
        save_all_content(content_list)
        # Store in ChromaDB for semantic search
        store_in_chromadb(processed_data, content_id)
        logger.info(f"Successfully processed and stored content with ID: {content_id}")
        return {
            "status": "success",
            "content_id": content_id,
            "message": "Content processed successfully",
            "data": content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during content processing: {str(e)}"
        )

@router.get("/content/{content_id}")
def get_content(content_id: str) -> Dict:
    try:
        content = find_content_by_id(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        return {
            "status": "success",
            "data": content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content {content_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving content: {str(e)}"
        )

@router.get("/content/")
def list_content(skip: int = 0, limit: int = 20, category: str = None, source_type: str = None) -> Dict:
    try:
        content_list = load_all_content()
        
        # Apply filters
        filtered_content = content_list
        if category:
            filtered_content = [item for item in filtered_content if item.get('category') == category]
        if source_type:
            filtered_content = [item for item in filtered_content if item.get('source_type') == source_type]
        
        # Apply pagination
        total_count = len(filtered_content)
        paginated_content = filtered_content[skip:skip + limit]
        
        return {
            "status": "success",
            "data": paginated_content,
            "pagination": {
                "total": total_count,
                "skip": skip,
                "limit": limit,
                "has_more": skip + limit < total_count
            }
        }
    except Exception as e:
        logger.error(f"Error listing content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing content: {str(e)}"
        )