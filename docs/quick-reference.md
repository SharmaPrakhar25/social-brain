# Quick Reference - Function Call Map

## Most Common Event Patterns

### 🎯 Content Processing (POST /api/process/)
```
User submits URL → process_content_endpoint() → content_processor.process_content() → 
[Parallel]: metadata extraction + audio download + transcription + AI analysis → 
JSON storage + ChromaDB storage → Return success
```

### 💬 Chat Query (POST /api/chat/)
```
User sends message → chat_endpoint() → chat_service.process_chat_query() →
intent classification → semantic search → AI response generation → formatted response
```

### 📋 Content Listing (GET /api/content/)
```
User requests content → list_content() → load_all_content() → apply filters + pagination → return results
```

---

## Key Function Locations

### Entry Points
- `main.py:app` - FastAPI application
- `api/ingest.py:router` - Content processing endpoints  
- `api/chat.py:router` - Chat endpoints

### Core Services
- `services/content_processor.py:ContentProcessor` - Main content processing
- `services/chat_service.py:EnhancedChatService` - Chat handling
- `services/instagram_extractor.py:InstagramExtractor` - Instagram-specific extraction
- `services/llama_agent.py:generate_contextual_response_ollama` - AI responses

### Storage Operations
- `api/ingest.py:load_all_content()` - Read JSON storage
- `api/ingest.py:save_all_content()` - Write JSON storage
- `api/ingest.py:store_in_chromadb()` - Vector storage
- `services/chat_service.py:_search_content()` - Vector search

---

## Error Handling Patterns

### Critical Errors (Stop Processing)
```python
try:
    # Critical operation
    result = critical_function()
except Exception as e:
    logger.error(f"Critical failure: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Non-Critical Errors (Continue Processing)
```python
try:
    # Optional operation
    optional_function()
except Exception as e:
    logger.warning(f"Optional operation failed: {e}")
    # Continue processing
```

### Graceful Degradation
```python
try:
    ai_response = generate_ai_response()
except Exception as e:
    logger.error(f"AI failed: {e}")
    ai_response = fallback_response()  # Use fallback
```

---

## Configuration Points

### Environment Variables
- `OLLAMA_MODEL` - AI model name (default: "llama3.1:8b")
- `OLLAMA_API_URL` - Ollama API endpoint (default: "http://localhost:11434/api/chat")
- `WHISPER_MODEL` - Whisper model size (default: "base")
- `DATA_DIR` - Data storage directory (default: "data")
- `CONTENT_FILE` - Content JSON file (default: "data/content.json")
- `CHROMA_DB_PATH` - ChromaDB path (default: "./chroma_db")

### Service Dependencies
- **Ollama** - Required for AI features (summaries, chat)
- **Whisper** - Required for transcription
- **ChromaDB** - Required for semantic search
- **yt-dlp** - Required for video/audio download

---

## Data Models

### Content Object Structure
```json
{
  "id": "timestamp_id",
  "source_type": "instagram|youtube|tiktok|other",
  "original_url": "https://...",
  "title": "Content title",
  "author": "Creator name",
  "transcription": "Full transcription text",
  "summary": "AI-generated summary",
  "keywords": ["keyword1", "keyword2"],
  "hashtags": ["#hashtag1", "#hashtag2"],
  "mentions": ["@user1", "@user2"],
  "category": "technology|business|entertainment|...",
  "sentiment": "positive|negative|neutral",
  "processing_status": "completed|failed",
  "created_at": "ISO timestamp"
}
```

### Chat Response Structure
```json
{
  "status": "success|error",
  "intent": "summary|search|recommendation|stats|general",
  "response": "Generated response text",
  "data": { "query_specific_data": "..." },
  "sources": [
    {
      "title": "Source title",
      "author": "Source author", 
      "category": "Source category",
      "url": "Source URL",
      "relevance_score": 0.95
    }
  ]
}
```

---

## Development Debugging

### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --reload
```

### Test Ollama Connection
```bash
curl http://localhost:8000/api/chat/test-ollama/
```

### Check Content Stats
```bash
curl http://localhost:8000/api/content/ | jq '.pagination.total'
```

### Manual Content Processing
```bash
curl -X POST http://localhost:8000/api/process/ \
  -F "url=https://instagram.com/reel/..."
```

### Test Chat Functionality
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "What content do I have?"}'
```

---

## Performance Monitoring

### Key Metrics to Monitor
- Content processing time (target: <60s)
- Chat response time (target: <10s)  
- File I/O operations (target: <1s)
- Memory usage (target: <1GB)
- Error rates (target: <5%)

### Bottleneck Identification
1. **Slow processing**: Check Whisper model size, audio length
2. **Slow chat**: Check Ollama model size, query complexity
3. **High memory**: Check ChromaDB index size, content volume
4. **Disk space**: Check downloads/ cleanup, content.json size

---

## Common Issues & Solutions

### Issue: Whisper model fails to load
**Solution**: Check available disk space, model download permissions
```bash
python -c "import whisper; whisper.load_model('base')"
```

### Issue: Ollama connection refused
**Solution**: Ensure Ollama is running and accessible
```bash
curl http://localhost:11434/api/tags
ollama serve  # If not running
```

### Issue: ChromaDB permission errors
**Solution**: Check directory permissions
```bash
chmod -R 755 ./chroma_db/
```

### Issue: Content file corruption
**Solution**: Backup and recover
```bash
cp data/content.json data/content.json.backup
# Manually fix JSON or restore from backup
```

### Issue: High processing times
**Solution**: Optimize model sizes
```bash
export WHISPER_MODEL=tiny  # Faster but less accurate
export OLLAMA_MODEL=llama3.1:8b  # Smaller model
```