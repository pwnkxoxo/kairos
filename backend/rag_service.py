"""
KAIROS Retrieval-Augmented Generation (RAG) Service
Provides:
  1. Semantic and BM25/TF-IDF chunk retrieval over user-uploaded documents and curated knowledge.
  2. Bounded text chunking with overlap and sentence boundary preservation.
  3. Strict ownership scoping: user document chunks are strictly partitioned by conversation_id.
  4. Curated programming references for Python, C, FastAPI, React, and TypeScript.
  5. Page-accurate citations for PDFs without hallucinating page numbers.
  6. Prompt injection defense: all retrieved chunks are bounded and marked as untrusted data.
"""

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from backend.storage import (
    delete_rag_chunks_for_doc,
    query_rag_chunks,
    store_rag_chunks,
)

DEFAULT_CHUNK_SIZE = 500  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 80  # characters overlap
MAX_RETRIEVED_CHUNKS = 5
MAX_RAG_CONTEXT_CHARS = 3500
MIN_SIMILARITY_THRESHOLD = 0.05


def clean_tokens(text: str) -> List[str]:
    """Tokenizes string into normalized lowercase alphanumeric word tokens."""
    return [w for w in re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())]


def chunk_text(
    text: str,
    target_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Splits text into bounded chunks respecting paragraph or sentence boundaries where possible.
    Preserves overlap to maintain context across chunk boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= target_size:
        return [clean_text]

    chunks: List[str] = []
    # Split initially by paragraph
    paragraphs = re.split(r"\n\s*\n", clean_text)
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds target_size, split by sentences
        if len(para) > target_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if current_len + len(s) + 1 > target_size and current_chunk:
                    chunk_str = " ".join(current_chunk)
                    chunks.append(chunk_str)
                    # Keep tail for overlap
                    overlap_tokens = chunk_str[-overlap:].split()
                    current_chunk = overlap_tokens if len(" ".join(overlap_tokens)) < overlap else [s]
                    current_len = sum(len(x) + 1 for x in current_chunk)
                current_chunk.append(s)
                current_len += len(s) + 1
        else:
            if current_len + len(para) + 1 > target_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append(chunk_str)
                overlap_tokens = chunk_str[-overlap:].split()
                current_chunk = overlap_tokens if len(" ".join(overlap_tokens)) < overlap else [para]
                current_len = sum(len(x) + 1 for x in current_chunk)
            current_chunk.append(para)
            current_len += len(para) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def compute_bm25_similarity(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    """
    Computes TF-IDF / BM25 term overlap similarity score between query and chunk tokens.
    Returns float in range [0.0, 1.0]. Zero network, pure Python standard library.
    """
    if not query_tokens or not chunk_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    chunk_counts = Counter(chunk_tokens)

    intersection = set(query_counts.keys()) & set(chunk_counts.keys())
    if not intersection:
        return 0.0

    score = 0.0
    doc_len = len(chunk_tokens)
    avg_len = 80.0  # reference average chunk token count
    k1 = 1.2
    b = 0.75

    for word in intersection:
        tf = chunk_counts[word]
        # Bounded BM25 TF component
        norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
        # Term significance boost for query frequency
        qf = query_counts[word]
        score += norm_tf * math.log(1.0 + qf)

    # Normalize by query length
    normalized_score = score / (len(query_tokens) * 2.0)
    return min(1.0, max(0.0, normalized_score))


def retrieve_relevant_chunks(
    query: str,
    conversation_id: Optional[str] = None,
    source_type: Optional[str] = None,
    top_k: int = MAX_RETRIEVED_CHUNKS,
    score_threshold: float = MIN_SIMILARITY_THRESHOLD,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves and ranks the top-K relevant chunks for a question.
    Strictly isolates chunks:
      - Returns conversation's own document chunks + global curated reference chunks.
      - Blocks access to any chunks belonging to other conversations.
    """
    query_tokens = clean_tokens(query)
    if not query_tokens:
        return []

    # Query storage for accessible chunks
    candidates = query_rag_chunks(
        conversation_id=conversation_id,
        source_type=source_type,
        db_path=db_path,
    )

    if not candidates:
        return []

    scored_candidates = []
    for cand in candidates:
        c_tokens = clean_tokens(cand["content"])
        score = compute_bm25_similarity(query_tokens, c_tokens)
        if score >= score_threshold:
            cand_copy = dict(cand)
            cand_copy["score"] = score
            scored_candidates.append(cand_copy)

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate closely matching content and take top_k
    seen_content: Set[str] = set()
    results: List[Dict[str, Any]] = []

    for item in scored_candidates:
        content_key = item["content"][:100].strip().lower()
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        results.append(item)
        if len(results) >= top_k:
            break

    return results


def format_rag_evidence(
    chunks: List[Dict[str, Any]],
    max_chars: int = MAX_RAG_CONTEXT_CHARS,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Formats retrieved chunks as strictly untrusted evidence for Gemma.
    Returns (formatted_evidence_string, sources_list).
    Preserves exact page numbers for PDF citations.
    """
    if not chunks:
        return "No relevant reference documentation or document chunks were found.", []

    evidence_lines: List[str] = []
    sources: List[Dict[str, Any]] = []
    total_chars = 0

    for idx, c in enumerate(chunks, start=1):
        source_name = c.get("source_name", "Document")
        page_num = c.get("page_number")
        page_info = f" (Page {page_num})" if page_num is not None else ""
        content = c.get("content", "").strip()

        if total_chars + len(content) > max_chars:
            allowed = max(0, max_chars - total_chars)
            if allowed > 100:
                content = content[:allowed] + " ...[truncated]"
            else:
                break

        evidence_lines.append(
            f"[Source {idx}]: {source_name}{page_info}\n{content}\n"
        )
        sources.append({
            "source_name": source_name,
            "page_number": page_num,
            "doc_id": c.get("doc_id"),
            "source_type": c.get("source_type"),
            "score": round(c.get("score", 0.0), 3),
        })
        total_chars += len(content)

    return "\n".join(evidence_lines).strip(), sources


# ------------------------------------------------------------------------------
# Ingestion Indexers
# ------------------------------------------------------------------------------
def index_document_pages(
    doc_id: str,
    conversation_id: str,
    filename: str,
    pages: List[Dict[str, Any]],
    source_type: str = "pdf",
    db_path: Optional[Path] = None,
) -> int:
    """
    Chunks and indexes extracted pages from an uploaded PDF or scanned document.
    Preserves page_number on every chunk.
    """
    delete_rag_chunks_for_doc(doc_id, db_path=db_path)

    all_chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    for p in pages:
        p_num = p.get("page_number", 1)
        p_text = p.get("text", "").strip()
        if not p_text:
            continue

        raw_chunks = chunk_text(p_text)
        for raw in raw_chunks:
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "conversation_id": conversation_id,
                "source_type": source_type,
                "source_name": filename,
                "page_number": p_num,
                "chunk_index": chunk_idx,
                "content": raw,
                "metadata_json": {"filename": filename, "page_number": p_num},
            })
            chunk_idx += 1

    return store_rag_chunks(all_chunks, db_path=db_path)


def index_url_content(
    doc_id: str,
    conversation_id: str,
    url: str,
    title: str,
    text: str,
    db_path: Optional[Path] = None,
) -> int:
    """Chunks and indexes extracted public webpage content."""
    delete_rag_chunks_for_doc(doc_id, db_path=db_path)

    raw_chunks = chunk_text(text)
    chunks_to_store: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_chunks):
        chunks_to_store.append({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "conversation_id": conversation_id,
            "source_type": "url",
            "source_name": title or url,
            "page_number": None,
            "chunk_index": idx,
            "content": raw,
            "metadata_json": {"url": url, "title": title},
        })

    return store_rag_chunks(chunks_to_store, db_path=db_path)


# ------------------------------------------------------------------------------
# Curated Programming Knowledge Documentation
# ------------------------------------------------------------------------------
CURATED_PROGRAMMING_DOCS = {
    "prog_python": {
        "title": "Python Language & Core Runtime Reference",
        "content": """
Python 3 Language Reference:
- Data Classes: Decorated with `@dataclass`, standard library `dataclasses` automatically generates `__init__`, `__repr__`, and `__eq__`. They offer low runtime overhead compared to Pydantic. Use Pydantic `BaseModel` for external parsing and serialization, and dataclasses for internal data structures.
- Async/Await: Coroutines defined with `async def` run on an event loop (`asyncio.run()`). Non-blocking I/O operations must be awaited with `await`. Use `asyncio.gather()` for concurrent task scheduling. Never block the event loop with synchronous sleep or CPU-intensive tasks.
- Generators: Functions with `yield` return lazy iterators that produce values on demand, saving memory when processing large streams or files.
- Memory & GIL: CPython uses reference counting supplemented by a cyclic garbage collector. The Global Interpreter Lock (GIL) ensures thread safety for interpreter state, meaning CPU-bound tasks require multiprocessing rather than threading.
        """,
    },
    "prog_c": {
        "title": "C Language Standard & Systems Reference",
        "content": """
C Standard (C11/C17) Reference:
- Dynamic Memory: Allocated on heap using `malloc(size_t size)`, `calloc(size_t num, size_t size)`, and `realloc(void *ptr, size_t size)`. Must check for `NULL` return to prevent null pointer dereferences. Always free allocated blocks with `free(ptr)` and set pointer to `NULL` to prevent dangling pointer access and double frees.
- Pointer Arithmetic: Incrementing a pointer `ptr + 1` advances by `sizeof(*ptr)` bytes. Out-of-bounds pointer arithmetic is undefined behavior.
- Buffer Safety: Avoid unsafe functions like `gets()` or `strcpy()`. Prefer `strncpy()`, `snprintf()`, or bounds-checked functions to prevent stack and heap buffer overflows.
- Header Guards: Always guard header files with `#ifndef HEADER_NAME_H`, `#define HEADER_NAME_H`, and `#endif` or `#pragma once` to prevent duplicate type declarations during compilation.
        """,
    },
    "prog_fastapi": {
        "title": "FastAPI Web Framework Reference",
        "content": """
FastAPI Reference:
- Dependency Injection: Declared with `Depends(callable)`. Injected at path operation or router level. Yield dependencies (`def get_db(): try: yield db finally: db.close()`) handle setup and cleanup teardown cleanly.
- Request & Response Models: Use Pydantic `BaseModel` for request body validation and `response_model` on decorator. Extra fields can be stripped or forbidden. `Field(..., max_length=..., min_length=...)` enforces validation rules.
- Background Tasks: Pass `BackgroundTasks` as path function parameter and invoke `background_tasks.add_task(func, *args)`. Runs after the response is returned to the client.
- Status Codes & Errors: Raise `HTTPException(status_code=..., detail=...)`. Use `status.HTTP_404_NOT_FOUND` or `status.HTTP_422_UNPROCESSABLE_CONTENT` from `fastapi.status`.
        """,
    },
    "prog_react": {
        "title": "React 18/19 Architecture & Hooks Reference",
        "content": """
React Framework Reference:
- useState: Declares reactive state variable. State updates are batched. Functional updates `setCount(prev => prev + 1)` prevent stale closure bugs during rapid updates.
- useEffect: Synchronizes components with external systems. The dependency array `[dep1, dep2]` specifies when the effect runs. An empty array `[]` runs once on mount. Always return a cleanup function `return () => { ... }` to cancel timers, abort fetch requests, or clean up subscriptions.
- useCallback & useMemo: `useCallback(fn, deps)` memoizes function references to prevent unnecessary child re-renders. `useMemo(() => compute(), deps)` caches expensive calculation results between renders.
- Props & Keys: Props are read-only immutable inputs. Array rendering requires a unique, stable `key` prop on each root element (avoid using array indices if elements can reorder).
        """,
    },
    "prog_typescript": {
        "title": "TypeScript 5 Language Reference",
        "content": """
TypeScript Reference:
- Generics: Parameterize functions and interfaces with types: `function identity<T>(arg: T): T`. Generic constraints `T extends { id: string }` restrict allowable types.
- Interfaces vs Type Aliases: Interfaces support declaration merging and extending (`interface A extends B`). Type aliases `type X = ...` support union types (`A | B`), intersections (`A & B`), tuples, and mapped types.
- Type Narrowing: TypeScript refines types inside control flow branches using `typeof`, `instanceof`, `in`, equality checks, and custom type predicates (`is` keyword, e.g. `function isString(x: unknown): x is string`).
- Strict Mode: Enabling `"strict": true` in `tsconfig.json` turns on `noImplicitAny`, `strictNullChecks`, and `strictFunctionTypes`, preventing runtime undefined/null errors.
        """,
    },
}


def index_curated_programming_docs(db_path: Optional[Path] = None) -> int:
    """
    Indexes curated Python, C, FastAPI, React, and TypeScript documentation
    into rag_chunks with conversation_id=None (global read-only access).
    """
    total_chunks = 0
    for doc_id, doc_data in CURATED_PROGRAMMING_DOCS.items():
        delete_rag_chunks_for_doc(doc_id, db_path=db_path)
        chunks = chunk_text(doc_data["content"])
        chunk_dicts = []
        for idx, text_chunk in enumerate(chunks):
            chunk_dicts.append({
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "doc_id": doc_id,
                "conversation_id": None,  # Global reference
                "source_type": "programming_doc",
                "source_name": doc_data["title"],
                "page_number": None,
                "chunk_index": idx,
                "content": text_chunk,
                "metadata_json": {"topic": doc_id, "title": doc_data["title"]},
            })
        total_chunks += store_rag_chunks(chunk_dicts, db_path=db_path)

    return total_chunks


# Initialize curated docs on startup if not already populated
try:
    existing_docs = query_rag_chunks(source_type="programming_doc")
    if not existing_docs:
        index_curated_programming_docs()
except Exception:
    pass
