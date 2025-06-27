"""
Enhanced Video Ingestion and Processing Endpoint

This module handles:
- Content processing with structured data extraction
- Proper database storage with metadata
- Keyword extraction and categorization
- ChromaDB integration for semantic search
"""

from fastapi import APIRouter, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from services.content_processor import content_processor
from models.database import get_db, init_db
from models.content import Content, ContentKeyword
from chromadb import PersistentClient
import logging
from typing import Dict, List
import json
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize database on startup
init_db()

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Initialize ChromaDB client
chroma_client = PersistentClient(path="./chroma_db")

def get_or_create_keywords(db: Session, keywords: List[str]) -> List[ContentKeyword]:
    """
    Get existing keywords or create new ones
    """
    keyword_objects = []
    
    for keyword_text in keywords:
        # Check if keyword already exists
        existing_keyword = db.query(ContentKeyword).filter(
            ContentKeyword.keyword == keyword_text
        ).first()
        
        if existing_keyword:
            # Update frequency
            existing_keyword.frequency += 1
            keyword_objects.append(existing_keyword)
        else:
            # Create new keyword
            new_keyword = ContentKeyword(
                keyword=keyword_text,
                category='general',  # TODO: Add keyword categorization
                frequency=1
            )
            db.add(new_keyword)
            keyword_objects.append(new_keyword)
    
    db.commit()
    return keyword_objects

def store_in_chromadb(content_data: Dict, content_id: str):
    """
    Store content in ChromaDB for semantic search
    """
    try:
        collection = chroma_client.get_or_create_collection("enhanced_content")
        
        # Create a rich document for better search
        document_text = f"""
        Title: {content_data.get('title', '')}
        Author: {content_data.get('author', '')}
        Category: {content_data.get('category', '')}
        Summary: {content_data.get('summary', '')}
        Keywords: {', '.join(content_data.get('keywords', []))}
        Transcription: {content_data.get('transcription', '')}
        """
        
        # Enhanced metadata for filtering
        metadata = {
            "content_id": content_id,
            "source_type": content_data.get('source_type', ''),
            "title": content_data.get('title', ''),
            "category": content_data.get('category', ''),
            "sentiment": content_data.get('sentiment', ''),
            "author": content_data.get('author', ''),
            "duration": content_data.get('duration', 0),
            "keywords": json.dumps(content_data.get('keywords', [])),
            "url": content_data.get('original_url', '')
        }
        
        collection.add(
            documents=[document_text.strip()],
            metadatas=[metadata],
            ids=[content_id]
        )
        
        logger.info(f"Successfully stored content {content_id} in ChromaDB")
        
    except Exception as e:
        logger.error(f"Failed to store in ChromaDB: {e}")
        # Don't raise exception - ChromaDB failure shouldn't break the main flow

@router.post("/process/")
def process_content_endpoint(url: str = Form(...), db: Session = Depends(get_db)) -> Dict:
    """
    Enhanced content processing endpoint with structured data storage.

    Args:
        url (str): The URL of the content to process.
        db (Session): Database session dependency.

    Returns:
        dict: Structured response with processing results and metadata.

    Raises:
        HTTPException: If processing fails, returns a 500 error with details.
    """
    try:
        logger.info(f"Processing content from URL: {url}")
        
        # Check if content already exists
        existing_content = db.query(Content).filter(Content.original_url == url).first()
        if existing_content:
            logger.info(f"Content already exists with ID: {existing_content.id}")
            return {
                "status": "already_exists",
                "content_id": existing_content.id,
                "message": "Content has already been processed",
                "data": {
                    "title": existing_content.title,
                    "category": existing_content.category,
                    "summary": existing_content.summary,
                    "keywords": [kw.keyword for kw in existing_content.keywords],
                    "sentiment": existing_content.sentiment,
                    "processing_status": existing_content.processing_status
                }
            }
        
        # Process the content
        processed_data = content_processor.process_content(url)
        
        # Handle processing failure
        if processed_data.get('processing_status') == 'failed':
            # Still store the failed attempt for tracking
            failed_content = Content(
                source_type=processed_data.get('source_type', 'unknown'),
                original_url=url,
                title=processed_data.get('title', 'Failed to Process Content'),
                processing_status='failed',
                error_message=processed_data.get('error_message', 'Unknown error'),
                content_type='video'
            )
            db.add(failed_content)
            db.commit()
            
            raise HTTPException(
                status_code=500, 
                detail=f"Content processing failed: {processed_data.get('error_message', 'Unknown error')}"
            )
        
        # Create and store content in database
        content = Content(
            source_type=processed_data['source_type'],
            original_url=processed_data['original_url'],
            title=processed_data.get('title', ''),
            author=processed_data.get('author', ''),
            platform_id=processed_data.get('platform_id', ''),
            transcription=processed_data['transcription'],
            summary=processed_data['summary'],
            content_type=processed_data.get('content_type', 'video'),
            duration=processed_data.get('duration', 0),
            category=processed_data['category'],
            sentiment=processed_data['sentiment'],
            language=processed_data.get('language', 'en'),
            processing_status='completed'
        )
        
        db.add(content)
        db.flush()  # Get the ID without committing
        
        # Handle keywords
        if processed_data.get('keywords'):
            keyword_objects = get_or_create_keywords(db, processed_data['keywords'])
            content.keywords = keyword_objects
        
        db.commit()
        
        # Store in ChromaDB for semantic search
        store_in_chromadb(processed_data, content.id)
        
        logger.info(f"Successfully processed and stored content with ID: {content.id}")
        
        # Return structured response
        return {
            "status": "success",
            "content_id": content.id,
            "message": "Content processed successfully",
            "data": {
                "title": content.title,
                "author": content.author,
                "source_type": content.source_type,
                "category": content.category,
                "sentiment": content.sentiment,
                "duration": content.duration,
                "summary": content.summary,
                "keywords": [kw.keyword for kw in content.keywords],
                "transcription_length": len(content.transcription) if content.transcription else 0,
                "processing_status": content.processing_status,
                "created_at": content.created_at.isoformat() if content.created_at else None
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing content: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during content processing: {str(e)}"
        )

@router.get("/content/{content_id}")
def get_content(content_id: str, db: Session = Depends(get_db)) -> Dict:
    """
    Retrieve processed content by ID
    """
    try:
        content = db.query(Content).filter(Content.id == content_id).first()
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "status": "success",
            "data": {
                "id": content.id,
                "title": content.title,
                "author": content.author,
                "source_type": content.source_type,
                "original_url": content.original_url,
                "category": content.category,
                "sentiment": content.sentiment,
                "duration": content.duration,
                "summary": content.summary,
                "transcription": content.transcription,
                "keywords": [kw.keyword for kw in content.keywords],
                "processing_status": content.processing_status,
                "created_at": content.created_at.isoformat() if content.created_at else None,
                "updated_at": content.updated_at.isoformat() if content.updated_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving content: {str(e)}")

@router.get("/content/")
def list_content(
    skip: int = 0, 
    limit: int = 20, 
    category: str = None,
    source_type: str = None,
    db: Session = Depends(get_db)
) -> Dict:
    """
    List processed content with optional filtering
    """
    try:
        query = db.query(Content)
        
        # Apply filters
        if category:
            query = query.filter(Content.category == category)
        if source_type:
            query = query.filter(Content.source_type == source_type)
        
        # Get total count
        total = query.count()
        
        # Apply pagination and ordering
        contents = query.order_by(Content.created_at.desc()).offset(skip).limit(limit).all()
        
        return {
            "status": "success",
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": [
                {
                    "id": content.id,
                    "title": content.title,
                    "author": content.author,
                    "source_type": content.source_type,
                    "category": content.category,
                    "sentiment": content.sentiment,
                    "duration": content.duration,
                    "summary": content.summary[:200] + "..." if len(content.summary) > 200 else content.summary,
                    "keywords": [kw.keyword for kw in content.keywords][:5],  # First 5 keywords
                    "processing_status": content.processing_status,
                    "created_at": content.created_at.isoformat() if content.created_at else None
                }
                for content in contents
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing content: {str(e)}")

@router.get("/stats/")
def get_stats(db: Session = Depends(get_db)) -> Dict:
    """
    Get processing statistics
    """
    try:
        total_content = db.query(Content).count()
        completed_content = db.query(Content).filter(Content.processing_status == 'completed').count()
        failed_content = db.query(Content).filter(Content.processing_status == 'failed').count()
        
        # Category distribution
        category_stats = db.query(Content.category, func.count(Content.id)).group_by(Content.category).all()
        categories = [cat[0] for cat in category_stats if cat[0]]
        
        # Source type distribution
        source_stats = db.query(Content.source_type, func.count(Content.id)).group_by(Content.source_type).all()
        
        # Get total keywords
        total_keywords = db.query(ContentKeyword).count()
        
        # Get recent content (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_content = db.query(Content).filter(Content.created_at >= week_ago).count()
        
        return {
            "status": "success",
            "total_content": total_content,
            "completed_content": completed_content,
            "failed_content": failed_content,
            "success_rate": round((completed_content / total_content * 100), 1) if total_content > 0 else 0,
            "categories": categories,
            "total_keywords": total_keywords,
            "recent_content": recent_content,
            "category_distribution": dict(category_stats),
            "source_distribution": dict(source_stats)
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")