# Instagram AI Agent - Workflow & Function Call Map

This document provides a comprehensive map of all possible events in the Instagram AI Agent application and the complete function call chains that occur for each event.

## Application Architecture Overview

```
Frontend (static/index.html) 
    ↓ HTTP Requests
FastAPI Application (main.py)
    ↓ Route Handlers
API Routers (api/chat.py, api/ingest.py)
    ↓ Service Layer
Services (chat_service.py, content_processor.py, instagram_extractor.py, llama_agent.py)
    ↓ Storage Layer
Data Storage (JSON files, ChromaDB)
```

## Event Categories

1. **Server Startup Events**
2. **Content Processing Events**
3. **Chat Query Events**
4. **Content Retrieval Events**
5. **System Health & Info Events**
6. **Error Handling Events**

---

## 1. SERVER STARTUP EVENTS

### Event: Application Startup
**Trigger**: `uvicorn main:app --reload`

#### Function Call Chain:
```
main.py:app creation
├── load_dotenv()                           # Environment loading
├── logging.basicConfig()                   # Logging setup
├── FastAPI()                              # App initialization
├── app.include_router(ingest_router)       # API route registration
├── app.include_router(chat_router)         # API route registration
├── app.mount("/static", StaticFiles())     # Static file mounting
│
├── services/content_processor.py imports
│   ├── whisper.load_model()               # Whisper model loading
│   ├── instagram_extractor initialization
│   └── ContentProcessor() instantiation
│
├── services/chat_service.py imports
│   ├── EnhancedChatService.__init__()
│   ├── PersistentClient(path="./chroma_db") # ChromaDB initialization
│   └── Response templates setup
│
└── API routers initialization
    ├── api/ingest.py
    │   ├── PersistentClient() initialization
    │   ├── os.makedirs(DATA_DIR)          # Ensure data directory
    │   └── load_dotenv()
    │
    └── api/chat.py
        ├── load_dotenv()
        └── logging configuration
```

---

## 2. CONTENT PROCESSING EVENTS

### Event: Process Instagram Reel/Video
**Trigger**: `POST /api/process/` with URL

#### Function Call Chain:
```
POST /api/process/
├── api/ingest.py:process_content_endpoint()
│   ├── logger.info(f"Processing content from URL: {url}")
│   ├── find_content_by_url(url)                    # Check duplicates
│   │   └── load_all_content()                      # Load from JSON file
│   │
│   ├── content_processor.process_content(url)      # Main processing
│   │   ├── logger.info("Starting content processing...")
│   │   ├── extract_metadata_from_url(url)          # Metadata extraction
│   │   │   ├── detect_source_type(url)             # Platform detection
│   │   │   │   └── urlparse(url).netloc analysis
│   │   │   │
│   │   │   ├── IF instagram:
│   │   │   │   ├── instagram_extractor.extract_enhanced_metadata(url)
│   │   │   │   │   ├── _is_instagram_url(url)
│   │   │   │   │   ├── _extract_shortcode(url)     # Extract post ID
│   │   │   │   │   ├── _fetch_post_metadata(url, shortcode)
│   │   │   │   │   │   ├── requests.get(url)       # Fetch page content
│   │   │   │   │   │   ├── _parse_page_content()   # Extract JSON-LD, meta tags
│   │   │   │   │   │   │   ├── re.search() for structured data
│   │   │   │   │   │   │   ├── json.loads() for Instagram data
│   │   │   │   │   │   │   ├── extract_hashtags()  # Parse hashtags
│   │   │   │   │   │   │   └── extract_mentions()  # Parse mentions
│   │   │   │   │   │   └── Return enhanced metadata
│   │   │   │   │   └── extract_topics_from_hashtags()
│   │   │   │   │
│   │   │   │   └── _extract_basic_metadata_yt_dlp(url) # Fallback
│   │   │   │
│   │   │   └── ELSE: _extract_basic_metadata_yt_dlp(url)
│   │   │       ├── yt_dlp.YoutubeDL.extract_info()
│   │   │       └── Clean and structure metadata
│   │   │
│   │   ├── download_audio(url)                     # Audio extraction
│   │   │   ├── tempfile.mkdtemp()                  # Temp directory
│   │   │   ├── yt_dlp.YoutubeDL() with audio options
│   │   │   ├── ydl.extract_info(url, download=True)
│   │   │   └── Return audio file path
│   │   │
│   │   ├── transcribe_audio(audio_path)            # Speech-to-text
│   │   │   ├── whisper_model.transcribe(audio_path)
│   │   │   └── Return transcription text
│   │   │
│   │   ├── extract_keywords(transcription, hashtags) # Keyword extraction
│   │   │   ├── TfidfVectorizer setup
│   │   │   ├── vectorizer.fit_transform([text])
│   │   │   ├── _extract_keywords_from_hashtags()   # Enhanced with hashtags
│   │   │   └── Return merged keywords
│   │   │
│   │   ├── categorize_content(transcription, keywords, hashtag_topics)
│   │   │   ├── IF hashtag_topics: return hashtag_topics[0]
│   │   │   ├── ELSE: keyword matching against categories
│   │   │   └── Return category
│   │   │
│   │   ├── analyze_sentiment(transcription)        # Sentiment analysis
│   │   │   ├── Simple word-based sentiment scoring
│   │   │   └── Return sentiment
│   │   │
│   │   ├── generate_summary(transcription)         # AI summarization
│   │   │   ├── call_ollama_api() with summary prompt
│   │   │   │   ├── requests.post(OLLAMA_API_URL)
│   │   │   │   ├── response.json()
│   │   │   │   └── Extract generated text
│   │   │   └── Clean and return summary
│   │   │
│   │   ├── generate_title(summary, keywords, transcription)
│   │   │   ├── call_ollama_api() with title prompt
│   │   │   └── Clean and return title
│   │   │
│   │   └── Return structured result with all metadata
│   │
│   ├── datetime.now().timestamp() * 1000          # Generate unique ID
│   ├── Prepare content object with all fields
│   ├── load_all_content()                         # Load existing content
│   ├── content_list.append(content)               # Add new content
│   ├── save_all_content(content_list)             # Save to JSON
│   │   └── json.dump() to data/content.json
│   │
│   ├── store_in_chromadb(processed_data, content_id) # Vector storage
│   │   ├── chroma_client.get_or_create_collection("enhanced_content")
│   │   ├── Create rich document with all metadata
│   │   ├── collection.add() with documents and metadata
│   │   └── logger.info("Successfully stored...")
│   │
│   └── Return success response with content data
│
└── Cleanup: os.remove(audio_path) and temp directory
```

---

## 3. CHAT QUERY EVENTS

### Event: User Chat Query
**Trigger**: `POST /api/chat/` with message

#### Function Call Chain:
```
POST /api/chat/
├── api/chat.py:chat_endpoint(request)
│   ├── logger.info("=== CHAT API REQUEST START ===")
│   ├── Validate request.message
│   ├── chat_service.process_chat_query(cleaned_message)
│   │   ├── logger.info(f"Processing chat query: '{query}'")
│   │   ├── _classify_query_intent(query)           # Intent detection
│   │   │   ├── query.lower() analysis
│   │   │   ├── Keyword matching for intents:
│   │   │   │   ├── 'summary': ['summarize', 'summary', 'what did', 'tell me about']
│   │   │   │   ├── 'recommendation': ['recommend', 'suggest', 'what should', 'similar']
│   │   │   │   ├── 'search': ['find', 'search', 'show me', 'look for']
│   │   │   │   ├── 'stats': ['stats', 'statistics', 'how many', 'overview']
│   │   │   │   └── 'general': default
│   │   │   └── Return intent
│   │   │
│   │   ├── IF intent == 'stats':
│   │   │   ├── _get_content_stats_file()
│   │   │   │   ├── os.path.exists(content_file)
│   │   │   │   ├── json.load() from data/content.json
│   │   │   │   ├── Calculate total_content, latest_title
│   │   │   │   └── Return stats dict
│   │   │   │
│   │   │   ├── _format_stats_response(stats)
│   │   │   └── Return stats response
│   │   │
│   │   ├── ELSE: (search, summary, recommendation, general)
│   │   │   ├── _search_content(query, limit=5)     # Semantic search
│   │   │   │   ├── chroma_client.get_or_create_collection("enhanced_content")
│   │   │   │   ├── _expand_query(query)            # Query expansion
│   │   │   │   │   ├── Basic expansion with original query
│   │   │   │   │   ├── Hashtag variations if applicable
│   │   │   │   │   ├── Automation term expansion:
│   │   │   │   │   │   ├── 'automation' → ['workflow', 'n8n', 'zapier']
│   │   │   │   │   │   ├── 'workflow' → ['automation', 'n8n', 'process']
│   │   │   │   │   │   ├── 'n8n' → ['automation', 'workflow', 'integration']
│   │   │   │   │   │   └── etc.
│   │   │   │   │   └── Return expanded queries (max 3)
│   │   │   │   │
│   │   │   │   ├── FOR each expanded query:
│   │   │   │   │   ├── collection.query() with semantic search
│   │   │   │   │   ├── Parse JSON metadata (keywords, hashtags, mentions)
│   │   │   │   │   ├── Calculate relevance_score = 1 - distance
│   │   │   │   │   ├── Avoid duplicates with seen_content_ids
│   │   │   │   │   └── Collect results
│   │   │   │   │
│   │   │   │   ├── Sort by relevance_score (highest first)
│   │   │   │   └── Return top results
│   │   │   │
│   │   │   ├── IF no search results:
│   │   │   │   └── Return "no content found" response
│   │   │   │
│   │   │   ├── FOR each result: add relevance explanations
│   │   │   │   └── _explain_relevance(query, result)
│   │   │   │       ├── Check title relevance
│   │   │   │       ├── Check keyword matches
│   │   │   │       ├── Check category relevance
│   │   │   │       ├── Check hashtag matches
│   │   │   │       ├── Check search query used
│   │   │   │       └── Combine explanations
│   │   │   │
│   │   │   ├── generate_contextual_response_ollama(query, search_results, intent)
│   │   │   │   ├── logger.info("Starting contextual response generation...")
│   │   │   │   ├── Process context data for top 3 results
│   │   │   │   ├── Create enhanced context with relevance explanations
│   │   │   │   ├── Generate intent-specific prompt:
│   │   │   │   │   ├── 'summary': Comprehensive response with WHY explanations
│   │   │   │   │   ├── 'recommendation': Thoughtful recommendations with reasoning
│   │   │   │   │   └── 'general'/'search': Helpful response with connections
│   │   │   │   │
│   │   │   │   ├── requests.post(OLLAMA_API_URL) with prompt
│   │   │   │   │   ├── Validate response structure
│   │   │   │   │   ├── Extract message.content
│   │   │   │   │   ├── Clean response (remove artifacts)
│   │   │   │   │   └── Return generated text
│   │   │   │   │
│   │   │   │   └── Handle errors with fallback responses
│   │   │   │
│   │   │   ├── Prepare sources array (top 3 results)
│   │   │   └── Return structured response
│   │   │
│   │   └── Return final result with status, intent, response, data, sources
│   │
│   ├── Log processing results and timing
│   ├── Create ChatResponse object
│   └── Return response
│
└── Handle errors with HTTPException
```

---

## 4. CONTENT RETRIEVAL EVENTS

### Event: Get Content by ID
**Trigger**: `GET /api/content/{content_id}`

#### Function Call Chain:
```
GET /api/content/{content_id}
├── api/ingest.py:get_content(content_id)
│   ├── find_content_by_id(content_id)
│   │   ├── load_all_content()
│   │   │   ├── os.path.exists(CONTENT_FILE)
│   │   │   ├── json.load() from data/content.json
│   │   │   └── Return content list
│   │   │
│   │   ├── Search for matching content_id
│   │   └── Return content or None
│   │
│   ├── IF not found: HTTPException(404)
│   └── Return content data
│
└── Handle errors with HTTPException(500)
```

### Event: List All Content
**Trigger**: `GET /api/content/` with optional filters

#### Function Call Chain:
```
GET /api/content/
├── api/ingest.py:list_content(skip, limit, category, source_type)
│   ├── load_all_content()                          # Load from JSON
│   ├── Apply filters:
│   │   ├── IF category: filter by content.category
│   │   └── IF source_type: filter by content.source_type
│   │
│   ├── Calculate total count
│   ├── Apply pagination with skip/limit
│   ├── Sort by created_at (newest first)
│   └── Return paginated results with metadata
│
└── Handle errors with HTTPException(500)
```

---

## 5. CHAT SUPPORT EVENTS

### Event: Get Chat Suggestions
**Trigger**: `GET /api/chat/suggestions/`

#### Function Call Chain:
```
GET /api/chat/suggestions/
├── api/chat.py:get_chat_suggestions()
│   ├── load_content_from_file()
│   │   ├── os.path.exists(content_file)
│   │   ├── json.load() from data/content.json
│   │   └── Return content list
│   │
│   ├── IF empty content: return default suggestions
│   │
│   ├── ELSE: analyze content for dynamic suggestions
│   │   ├── Count categories: {category: count}
│   │   ├── Extract and count keywords
│   │   ├── Get top 3 categories by frequency
│   │   ├── Get top 5 keywords by frequency
│   │   │
│   │   ├── Generate suggestions:
│   │   │   ├── Base suggestions: ["What's an overview...", "Show me recent..."]
│   │   │   ├── Category suggestions: "Tell me about my {category} content"
│   │   │   ├── Keyword suggestions: "Find content about {keyword}"
│   │   │   └── General suggestions: ["Recommend similar...", "Main themes..."]
│   │   │
│   │   └── Limit to 8 suggestions
│   │
│   └── Return suggestions array
│
└── Handle errors with fallback suggestions
```

### Event: Get Supported Intents
**Trigger**: `GET /api/chat/intents/`

#### Function Call Chain:
```
GET /api/chat/intents/
├── api/chat.py:get_supported_intents()
│   └── Return static intent definitions:
│       ├── 'summary': descriptions and examples
│       ├── 'search': descriptions and examples  
│       ├── 'recommendation': descriptions and examples
│       └── 'stats': descriptions and examples
│
└── No error handling needed (static data)
```

### Event: Submit Chat Feedback
**Trigger**: `POST /api/chat/feedback/`

#### Function Call Chain:
```
POST /api/chat/feedback/
├── api/chat.py:submit_chat_feedback(response_helpful, response_id, feedback_text)
│   ├── logger.info() feedback details
│   ├── Log feedback for analysis (file/database in production)
│   └── Return acknowledgment message
│
└── Handle errors with fallback message
```

### Event: Test Ollama Connection
**Trigger**: `GET /api/chat/test-ollama/`

#### Function Call Chain:
```
GET /api/chat/test-ollama/
├── api/chat.py:test_ollama_endpoint()
│   ├── services/llama_agent.py:test_ollama_connection()
│   │   ├── requests.get(f"{base_url}/api/tags") # Check if Ollama running
│   │   ├── Parse available models
│   │   ├── Check if OLLAMA_MODEL in available models
│   │   │
│   │   ├── IF model available:
│   │   │   ├── Send test request: {"messages": [{"role": "user", "content": "Say hello"}]}
│   │   │   ├── requests.post(OLLAMA_API_URL) with test data
│   │   │   ├── Validate response structure
│   │   │   └── Return success with test response
│   │   │
│   │   └── ELSE: return error with available models
│   │
│   └── Handle connection errors and timeouts
│
└── Return test results
```

---

## 6. SYSTEM INFO EVENTS

### Event: Health Check
**Trigger**: `GET /health`

#### Function Call Chain:
```
GET /health
├── main.py:health_check()
│   └── Return static health status:
│       ├── status: "healthy"
│       ├── version: "2.0.0"
│       └── features: [list of application features]
│
└── No dependencies or error handling needed
```

### Event: API Info
**Trigger**: `GET /api/info`

#### Function Call Chain:
```
GET /api/info
├── main.py:api_info()
│   └── Return static API documentation:
│       ├── endpoints: {processing, chat}
│       ├── supported_platforms: [Instagram, YouTube, TikTok, Twitter]
│       └── features: {content_processing, storage, chat_interface}
│
└── No dependencies or error handling needed
```

### Event: Serve Web Interface
**Trigger**: `GET /`

#### Function Call Chain:
```
GET /
├── main.py:read_root()
│   ├── FileResponse("static/index.html")
│   └── Return HTML interface
│
└── FastAPI static file serving
```

### Event: Serve Static Files
**Trigger**: `GET /static/*`

#### Function Call Chain:
```
GET /static/*
├── FastAPI StaticFiles middleware
│   ├── File system lookup in static/ directory
│   └── Return file or 404
│
└── No custom handling needed
```

---

## 7. ERROR HANDLING EVENTS

### Event: Processing Failures
**Scenarios & Handlers**:

1. **Whisper Model Loading Failure**
   ```
   services/content_processor.py:
   ├── try: whisper.load_model()
   ├── except: logger.error() + whisper_model = None
   └── Later checks: if not self.whisper_model: raise ValueError()
   ```

2. **Audio Download Failure**
   ```
   download_audio():
   ├── try: yt_dlp.YoutubeDL.extract_info()
   ├── except: logger.error() + raise
   └── Cleanup: remove temp files
   ```

3. **Transcription Failure**
   ```
   transcribe_audio():
   ├── Validate file exists and size > 0
   ├── try: whisper_model.transcribe()
   ├── except: logger.error() + raise
   └── Caller handles with failed status
   ```

4. **Ollama API Failures**
   ```
   call_ollama_api() / generate_contextual_response_ollama():
   ├── try: requests.post()
   ├── except requests.exceptions.Timeout: "AI service taking too long"
   ├── except requests.exceptions.ConnectionError: "Couldn't connect to AI service"
   ├── except requests.exceptions.RequestException: "Network error"
   ├── except KeyError: "Unexpected response format"
   └── except Exception: "Couldn't generate response"
   ```

5. **ChromaDB Failures**
   ```
   store_in_chromadb():
   ├── try: collection operations
   ├── except: logger.error() + continue processing
   └── Don't raise - ChromaDB failure shouldn't break main flow
   
   _search_content():
   ├── try: collection.query()
   ├── except: logger.error() + return []
   └── Caller handles empty results gracefully
   ```

6. **File I/O Failures**
   ```
   load_all_content():
   ├── if not os.path.exists(): return []
   ├── try: json.load()
   ├── except: return []
   └── Always return list (empty if failed)
   
   save_all_content():
   ├── try: json.dump()
   ├── except: may raise - critical failure
   └── Used in atomic operations
   ```

7. **Instagram Extraction Failures**
   ```
   extract_enhanced_metadata():
   ├── try: enhanced extraction with web scraping
   ├── except: logger.warning() + _fallback_extraction()
   └── Always return dict with platform_type
   ```

---

## Data Flow Summary

### 1. Content Processing Flow
```
URL → Metadata Extraction → Audio Download → Transcription → 
AI Analysis (Keywords, Summary, Title, Category, Sentiment) → 
JSON Storage → ChromaDB Vector Storage → Response
```

### 2. Chat Query Flow
```
User Query → Intent Classification → Query Expansion → 
ChromaDB Semantic Search → Relevance Explanation → 
Ollama AI Response Generation → Structured Response
```

### 3. Data Storage Flow
```
Processed Content → JSON File (data/content.json) + ChromaDB (./chroma_db/)
├── JSON: Full structured content with metadata
└── ChromaDB: Vectorized documents with metadata for semantic search
```

### 4. Error Recovery Flow
```
Service Failure → Log Error → Fallback Strategy → Continue Processing OR Return Error Response
├── Non-critical failures: Log + Continue (ChromaDB, Instagram extraction)
└── Critical failures: Log + Return Error (Whisper, Core processing)
```

This workflow map provides complete visibility into how the application processes events and handles all possible scenarios, including error conditions and fallback strategies.