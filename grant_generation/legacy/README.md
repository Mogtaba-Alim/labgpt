# Legacy Grant Generation Files

This directory contains deprecated files from the original grant generation pipeline (v1.0). These files have been superseded by the modern v2.0 architecture.

## Deprecated Files and Their Replacements

### 1. app.py (Deprecated)
**Replaced by:** `enhanced_app.py` (v2.0)

**Why deprecated:**
- Used legacy `rag_service.py` instead of production `RAG/pipeline.py`
- Direct transformers pipeline instead of `inference.py` message format
- Pickle-based session storage instead of SQLite database
- No async document indexing
- No multi-turn conversation support
- No draft versioning or lineage tracking

**Modern equivalent:**
- `enhanced_app.py` - Flask app with SQLite, async indexing, modern RAG

---

### 2. enhanced_grant_service.py (Deprecated)
**Replaced by:** `enhanced_grant_service_v2.py`

**Why deprecated:**
- Used legacy `rag_service` for RAG queries
- Used direct `transformers.pipeline()` for generation
- No citation management or tracking
- No conversation history for multi-turn refinement
- No database persistence

**Modern equivalent:**
- `enhanced_grant_service_v2.py` - Modern service with RAGPipeline + inference.py
- Includes citation extraction, multi-turn conversations, quality assessment

---

### 3. rag_service.py (Deprecated)
**Replaced by:** `../RAG/pipeline.py` + `inference_adapter.py`

**Why deprecated:**
- Static FAISS index with no hybrid retrieval
- No BM25 sparse retrieval
- No cross-encoder reranking
- No query expansion (PRF)
- No cited span extraction
- No per-document management
- No embedding cache

**Modern equivalent:**
- `RAG/pipeline.py` - Production RAG with hybrid retrieval, reranking, expansion
- `inference_adapter.py` - Wrapper for grant-specific use with consistent params

---

### 4. enhanced_document_processor.py (Deprecated)
**Replaced by:** `RAG/pipeline.py` document ingestion + `corpus_manager.py`

**Why deprecated:**
- Extracted sections from uploaded documents for template filling
- Not needed with modern RAG approach (documents indexed, retrieved as needed)
- Limited document format support

**Modern equivalent:**
- `corpus_manager.py` - Manages per-project RAG indices
- `RAG/pipeline.py` - Handles document ingestion with semantic chunking

---

### 5. session_manager.py (Deprecated)
**Replaced by:** `database.py`

**Why deprecated:**
- Pickle-based file storage
- No versioning or lineage tracking
- No multi-user support
- Poor scalability

**Modern equivalent:**
- `database.py` - SQLite database with proper schema
- Tables: projects, documents, drafts, feedback, telemetry
- Draft versioning with parent_draft_id
- Conversation history storage

---

### 6. test_session.py (Deprecated)
**Replaced by:** (To be created) `test_grant_pipeline.py`

**Why deprecated:**
- Tests for old pickle-based session manager
- No longer relevant with database architecture

**Modern equivalent:**
- Will create comprehensive test suite for new pipeline

---

### 7. demo_personalized_prompts.py (Deprecated)
**Replaced by:** N/A (Demo file)

**Why deprecated:**
- Demo script showing prompt customization
- Not part of production pipeline

---

## Architecture Changes (v1.0 → v2.0)

### v1.0 Architecture (Deprecated)
```
User → Flask app.py → session_manager (pickle files)
                    → enhanced_document_processor (extract sections)
                    → enhanced_grant_service
                        → rag_service (static FAISS)
                        → transformers.pipeline (direct)
```

### v2.0 Architecture (Current)
```
User → Flask enhanced_app.py → SQLite database
                             → Celery async tasks
                             → corpus_manager (per-project RAG)
                             → enhanced_grant_service_v2
                                 → RAG/pipeline.py (hybrid retrieval)
                                 → inference_adapter
                                     → inference.py (trained LabGPT)
```

## Key Improvements in v2.0

1. **Modern RAG System**
   - Hybrid retrieval (FAISS + BM25 with RRF fusion)
   - Cross-encoder reranking for precision
   - PRF query expansion
   - Cited span extraction for attribution
   - Embedding cache (95%+ hit rate on re-indexing)

2. **Trained Model Integration**
   - Uses `inference.py` message format with fine-tuned Llama 3.1 8B (LabGPT)
   - Proper chat template via `tokenizer.apply_chat_template()`
   - Consistent parameters across all sections

3. **Database Persistence**
   - SQLite with proper relational schema
   - Draft versioning with parent lineage
   - Conversation history for multi-turn refinement
   - Telemetry tracking

4. **Async Processing**
   - Celery + Redis for background document indexing
   - Progress tracking and status polling
   - Non-blocking user experience

5. **Citation Management**
   - MLA 9th edition formatting
   - Automatic Works Cited generation
   - Citation tracking across draft versions

6. **Enhanced Prompts**
   - Detailed section-specific instructions
   - Content requirements and writing guidelines
   - Better quality and consistency

## Migration Notes

If you need to migrate data from v1.0 to v2.0:

1. Old pickle session files are incompatible
2. Documents must be re-indexed using RAG/pipeline.py
3. No automatic migration path - start fresh with v2.0

## Restoration

These files are kept for reference only. To restore v1.0:

```bash
# NOT RECOMMENDED - v1.0 is deprecated
cd grant_generation
mv legacy/app.py .
mv legacy/enhanced_grant_service.py .
mv legacy/rag_service.py .
# ... etc
```

However, v2.0 is significantly superior and should be used for all new work.

---

**Last Updated:** 2025-10-30
**Deprecated Version:** v1.0
**Current Version:** v2.0
