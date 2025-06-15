"""
Enhanced Instagram Reels AI Agent

A FastAPI application that processes Instagram Reels and other video content,
extracts structured data with keywords and metadata, and provides an intelligent
chat interface for querying the processed content.

Features:
- Video content processing with Whisper transcription
- LLaMA-powered summarization and chat
- Keyword extraction and categorization
- Structured database storage with SQLAlchemy
- Semantic search with ChromaDB
- Enhanced chat interface with intent classification
"""

import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.ingest import router as ingest_router
from api.chat import router as chat_router

# Configure logging levels to reduce verbosity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce verbosity for specific loggers
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Instagram Reels AI Agent",
    description="An AI-powered personal knowledge assistant for processing and querying video content",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API routers
app.include_router(ingest_router, prefix="/api", tags=["Content Processing"])
app.include_router(chat_router, prefix="/api", tags=["Chat Interface"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    """
    Serve the main web interface
    """
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "video_processing",
            "keyword_extraction", 
            "structured_storage",
            "semantic_search",
            "intelligent_chat"
        ]
    }

@app.get("/api/info")
async def api_info():
    """
    API information and capabilities
    """
    return {
        "name": "Instagram Reels AI Agent API",
        "version": "2.0.0",
        "description": "Enhanced AI-powered content processing and chat interface",
        "endpoints": {
            "processing": {
                "POST /api/process/": "Process video content from URLs",
                "GET /api/content/": "List processed content with filtering",
                "GET /api/content/{id}": "Get specific content by ID",
                "GET /api/stats/": "Get processing statistics"
            },
            "chat": {
                "POST /api/chat/": "Chat with your content library",
                "GET /api/chat/suggestions/": "Get suggested questions",
                "GET /api/chat/intents/": "Get supported chat intents",
                "POST /api/chat/feedback/": "Submit chat feedback"
            }
        },
        "supported_platforms": [
            "Instagram Reels",
            "YouTube",
            "TikTok", 
            "Twitter/X",
            "Other video platforms"
        ],
        "features": {
            "content_processing": {
                "transcription": "Whisper-based audio transcription",
                "summarization": "LLaMA-powered intelligent summaries",
                "keyword_extraction": "TF-IDF based keyword extraction",
                "categorization": "Automatic content categorization",
                "metadata_extraction": "Rich metadata from video platforms"
            },
            "storage": {
                "structured_database": "SQLAlchemy with SQLite/PostgreSQL",
                "vector_database": "ChromaDB for semantic search",
                "keyword_tracking": "Frequency-based keyword management"
            },
            "chat_interface": {
                "intent_classification": "Automatic query intent detection",
                "semantic_search": "Vector-based content retrieval",
                "contextual_responses": "LLaMA-powered conversational AI",
                "structured_output": "Formatted responses with sources"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced Instagram Reels AI Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8000)