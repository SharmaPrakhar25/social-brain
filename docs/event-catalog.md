# Complete Event Catalog - Instagram AI Agent

## Event Classification System

Events are classified by:
- **Trigger Source**: User, System, External Service
- **Event Type**: HTTP Request, File Operation, Service Call, Error
- **Criticality**: Critical, Important, Normal, Debug
- **Data Flow**: Input → Processing → Output

---

## 1. USER-TRIGGERED EVENTS

### 1.1 Web Interface Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| U001 | Load Web Interface | `GET /` | `main.py:read_root()` | ✓ Critical |
| U002 | Load Static Assets | `GET /static/*` | FastAPI StaticFiles | Normal |
| U003 | Process Content URL | Form submission in UI | `POST /api/process/` | ✓ Critical |
| U004 | Send Chat Message | Chat input in UI | `POST /api/chat/` | ✓ Critical |
| U005 | Request Chat Suggestions | Page load/refresh | `GET /api/chat/suggestions/` | Normal |

### 1.2 Direct API Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| U101 | Health Check Request | `GET /health` | `main.py:health_check()` | Normal |
| U102 | API Info Request | `GET /api/info` | `main.py:api_info()` | Normal |
| U103 | List Content Request | `GET /api/content/` | `api/ingest.py:list_content()` | Important |
| U104 | Get Content by ID | `GET /api/content/{id}` | `api/ingest.py:get_content()` | Important |
| U105 | Get Chat Intents | `GET /api/chat/intents/` | `api/chat.py:get_supported_intents()` | Normal |
| U106 | Submit Chat Feedback | `POST /api/chat/feedback/` | `api/chat.py:submit_chat_feedback()` | Normal |
| U107 | Test Ollama Connection | `GET /api/chat/test-ollama/` | `api/chat.py:test_ollama_endpoint()` | Debug |

---

## 2. SYSTEM-TRIGGERED EVENTS

### 2.1 Application Lifecycle Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| S001 | Application Startup | `uvicorn main:app` | `main.py` module load | ✓ Critical |
| S002 | Service Initialization | App startup | Various `__init__` methods | ✓ Critical |
| S003 | Model Loading | Service init | `whisper.load_model()` | ✓ Critical |
| S004 | Database Connection | Service init | `PersistentClient()` | Important |
| S005 | Environment Loading | Module import | `load_dotenv()` | Important |
| S006 | Logging Configuration | App startup | `logging.basicConfig()` | Normal |

### 2.2 File System Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| S101 | Content File Read | Content operations | `load_all_content()` | Important |
| S102 | Content File Write | Content storage | `save_all_content()` | ✓ Critical |
| S103 | Temp Directory Creation | Audio processing | `tempfile.mkdtemp()` | Important |
| S104 | Audio File Download | Content processing | `yt_dlp.extract_info()` | ✓ Critical |
| S105 | Audio File Cleanup | Processing completion | `os.remove()` | Normal |
| S106 | Data Directory Creation | App startup | `os.makedirs()` | Important |

### 2.3 Background Processing Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| S201 | Content Processing Start | URL submission | `content_processor.process_content()` | ✓ Critical |
| S202 | Metadata Extraction | Content processing | `extract_metadata_from_url()` | Important |
| S203 | Audio Transcription | Content processing | `transcribe_audio()` | ✓ Critical |
| S204 | Keyword Extraction | Content processing | `extract_keywords()` | Important |
| S205 | Content Categorization | Content processing | `categorize_content()` | Important |
| S206 | Sentiment Analysis | Content processing | `analyze_sentiment()` | Normal |
| S207 | Vector Storage | Content processing | `store_in_chromadb()` | Important |

---

## 3. EXTERNAL SERVICE EVENTS

### 3.1 Ollama API Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| E001 | Summary Generation | Content processing | `call_ollama_api()` for summary | Important |
| E002 | Title Generation | Content processing | `call_ollama_api()` for title | Important |
| E003 | Chat Response Generation | Chat query | `generate_contextual_response_ollama()` | Important |
| E004 | Ollama Connection Test | Debug request | `test_ollama_connection()` | Debug |
| E005 | Ollama Model Check | Service startup | API availability check | Normal |

### 3.2 External Content Services

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| E101 | Instagram Page Fetch | Instagram URL processing | `requests.get()` in instagram_extractor | Important |
| E102 | YouTube-DL Metadata | Content processing | `yt_dlp.extract_info()` | Important |
| E103 | Video/Audio Download | Content processing | `yt_dlp` download | ✓ Critical |
| E104 | Instagram Metadata Parse | Instagram processing | `_parse_page_content()` | Important |

### 3.3 AI Model Events

| Event ID | Event Name | Trigger | Entry Point | Critical Path |
|----------|------------|---------|-------------|---------------|
| E201 | Whisper Transcription | Audio processing | `whisper_model.transcribe()` | ✓ Critical |
| E202 | TF-IDF Vectorization | Keyword extraction | `TfidfVectorizer.fit_transform()` | Important |
| E203 | Semantic Search | Chat queries | `collection.query()` | Important |
| E204 | Vector Document Storage | Content storage | `collection.add()` | Important |

---

## 4. ERROR EVENTS

### 4.1 Service Failures

| Event ID | Event Name | Trigger | Entry Point | Severity |
|----------|------------|---------|-------------|----------|
| E401 | Whisper Model Load Fail | App startup | Model loading | Critical |
| E402 | Ollama Connection Fail | API calls | `requests.post()` | Important |
| E403 | ChromaDB Connection Fail | Service init | `PersistentClient()` | Important |
| E404 | yt-dlp Download Fail | Content processing | Download operations | Important |
| E405 | Audio Transcription Fail | Processing | `whisper_model.transcribe()` | Important |

### 4.2 Data Errors

| Event ID | Event Name | Trigger | Entry Point | Severity |
|----------|------------|---------|-------------|----------|
| E501 | Content File Read Error | File operations | `json.load()` | Important |
| E502 | Content File Write Error | Storage operations | `json.dump()` | Critical |
| E503 | Invalid URL Error | Content processing | URL validation | Normal |
| E504 | Malformed JSON Error | Data parsing | Various JSON operations | Important |
| E505 | Missing Content Error | Content retrieval | `find_content_by_id()` | Normal |

### 4.3 Network Errors

| Event ID | Event Name | Trigger | Entry Point | Severity |
|----------|------------|---------|-------------|----------|
| E601 | Instagram Fetch Timeout | Instagram extraction | `requests.get()` | Normal |
| E602 | Ollama API Timeout | AI operations | Ollama API calls | Important |
| E603 | Ollama Connection Refused | AI operations | Network connection | Important |
| E604 | Video Download Timeout | Content processing | yt-dlp operations | Important |
| E605 | Network Unreachable | External requests | Various network calls | Important |

### 4.4 Processing Errors

| Event ID | Event Name | Trigger | Entry Point | Severity |
|----------|------------|---------|-------------|----------|
| E701 | Audio File Empty | Transcription | File size check | Important |
| E702 | Transcription Too Short | Content analysis | Text length check | Normal |
| E703 | Keyword Extraction Fail | Processing | TF-IDF operations | Normal |
| E704 | Category Classification Fail | Processing | Categorization logic | Normal |
| E705 | Title Generation Fail | Processing | AI title generation | Normal |

---

## 5. DATA FLOW EVENTS

### 5.1 Content Data Flow

```
Event Flow: URL → Metadata → Audio → Transcription → Analysis → Storage
```

| Event ID | Event Name | Data Transform | Input Format | Output Format |
|----------|------------|----------------|--------------|---------------|
| D001 | URL Input | User input → Validation | String | Validated URL |
| D002 | Metadata Extraction | URL → Platform metadata | URL | Dict[metadata] |
| D003 | Audio Extraction | Video → Audio file | Video URL | Audio file path |
| D004 | Speech-to-Text | Audio → Text | Audio file | Transcription text |
| D005 | Text Analysis | Text → Structured data | Text | Keywords, sentiment, category |
| D006 | Content Storage | Structured data → Persistent storage | Dict | JSON + Vector DB |

### 5.2 Chat Data Flow

```
Event Flow: Query → Intent → Search → Context → AI Response → Formatted Output
```

| Event ID | Event Name | Data Transform | Input Format | Output Format |
|----------|------------|----------------|--------------|---------------|
| D101 | Query Input | User message → Clean text | String | Cleaned query |
| D102 | Intent Classification | Query → Intent type | String | Intent enum |
| D103 | Query Expansion | Query → Multiple queries | String | List[String] |
| D104 | Semantic Search | Query → Relevant content | String | List[ContentMatch] |
| D105 | Context Preparation | Content → AI prompt | List[ContentMatch] | Formatted prompt |
| D106 | AI Response | Prompt → Generated text | Prompt | AI response |
| D107 | Response Formatting | AI text → Structured response | String | ChatResponse |

---

## 6. MONITORING EVENTS

### 6.1 Performance Events

| Event ID | Event Name | Metric | Threshold | Alert Level |
|----------|------------|--------|-----------|-------------|
| M001 | Content Processing Duration | Processing time | >60s | Warning |
| M002 | Chat Response Time | Response time | >10s | Warning |
| M003 | File Operation Duration | I/O time | >1s | Info |
| M004 | AI Model Response Time | API latency | >30s | Warning |
| M005 | Memory Usage | RAM consumption | >1GB | Warning |

### 6.2 Usage Events

| Event ID | Event Name | Metric | Collection Method |
|----------|------------|--------|-------------------|
| M101 | Content Items Processed | Count | Counter increment |
| M102 | Chat Queries Received | Count | Counter increment |
| M103 | Successful Processing Rate | Percentage | Success/Total ratio |
| M104 | Error Rate | Percentage | Error/Total ratio |
| M105 | User Sessions | Count | Session tracking |

### 6.3 System Health Events

| Event ID | Event Name | Check | Frequency | Alert Condition |
|----------|------------|-------|-----------|-----------------|
| M201 | File System Health | Disk space | Startup | <1GB free |
| M202 | Service Availability | Endpoint health | Per request | Service down |
| M203 | External Service Health | Ollama connectivity | Per AI call | Connection failed |
| M204 | Data Integrity | File validation | Daily | Corrupted files |
| M205 | Model Availability | AI model status | Startup | Model not loaded |

---

## 7. DEBUGGING EVENTS

### 7.1 Trace Events

| Event ID | Event Name | Trigger | Log Level | Information Captured |
|----------|------------|---------|-----------|----------------------|
| T001 | Function Entry | Function call | DEBUG | Function name, parameters |
| T002 | Function Exit | Function return | DEBUG | Function name, return value |
| T003 | Variable State | State changes | DEBUG | Variable name, value |
| T004 | Loop Iteration | Loop execution | DEBUG | Iteration count, current item |
| T005 | Condition Evaluation | If/else branches | DEBUG | Condition result, branch taken |

### 7.2 Diagnostic Events

| Event ID | Event Name | Purpose | Data Collected | Usage |
|----------|------------|---------|----------------|-------|
| T101 | Request Correlation | Track request flow | Request ID, timestamps | Performance analysis |
| T102 | Error Context | Debug errors | Stack trace, variables | Error debugging |
| T103 | Performance Profiling | Identify bottlenecks | Execution times | Optimization |
| T104 | Data Validation | Verify data integrity | Input/output validation | Quality assurance |
| T105 | Service Dependencies | Track external calls | Service status, latency | Dependency monitoring |

---

## Event Trigger Summary

### By Frequency (Production Environment)
- **High Frequency** (>10/hour): Chat queries, content retrieval, health checks
- **Medium Frequency** (1-10/hour): Content processing, suggestions, static assets
- **Low Frequency** (<1/hour): System startup, configuration changes, debug endpoints
- **Event-Driven**: Error events (as needed), monitoring alerts (threshold-based)

### By Business Impact
- **Critical**: Content processing pipeline, chat functionality
- **Important**: Content storage, search accuracy, AI responses
- **Normal**: UI interactions, API documentation, system health
- **Debug**: Development tools, diagnostic endpoints, trace logging

### By Response Time Requirements
- **Real-time** (<100ms): Health checks, static assets, basic API calls
- **Interactive** (<3s): Chat responses, content retrieval, suggestions
- **Background** (<60s): Content processing, AI analysis, file operations
- **Batch** (>60s): Large file processing, bulk operations, maintenance tasks