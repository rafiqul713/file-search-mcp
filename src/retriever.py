"""
=============================================================================
retriever.py — The RAG Query / Search Layer
=============================================================================

WHAT THIS FILE DOES:
--------------------
This file handles Phase 2 of the RAG pipeline: taking a natural language
query, converting it into a vector using the same embedding model used
during indexing, and asking ChromaDB to return the most similar chunks.

This is the "R" in RAG — Retrieval.

HOW SEMANTIC SEARCH WORKS:
---------------------------
1. The user types a query: "what did I write about neural networks?"
2. We pass that query through the same embedding model used at index time.
   This produces a 384-dimensional vector (a list of 384 floats).
3. ChromaDB compares that vector against every stored chunk vector using
   cosine similarity — a mathematical measure of how "close" two vectors
   are in high-dimensional space.
4. The top-k most similar chunks are returned, along with a relevance score.

Cosine similarity of 1.0 = identical meaning
Cosine similarity of 0.0 = completely unrelated

PERFORMANCE TRICK — LAZY LOADING:
----------------------------------
The embedding model and vector store are expensive to initialise
(~1-2 seconds). We use module-level variables (_embeddings, _vector_store)
and only initialise them on the first call. Subsequent calls reuse the
already-loaded objects instantly. This pattern is called lazy loading.
=============================================================================
"""

import os
from dotenv import load_dotenv

# HuggingFaceEmbeddings — same model used in indexer.py.
# CRITICAL: you MUST use the same model for indexing and querying.
# If you index with model A and query with model B, the vectors live in
# different "spaces" and similarity scores will be meaningless.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma — the vector database client.
# Here we only READ from it (similarity search), not write.
from langchain_community.vectorstores import Chroma

# Load environment variables from .env
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR",      "./chroma_db")  # must match indexer.py
COLLECTION = os.getenv("COLLECTION_NAME", "my_files")     # must match indexer.py

# Must be the SAME model used in indexer.py — different model = broken search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
# These start as None and are initialised on the first call to _get_store().
# Module-level variables persist for the lifetime of the Python process,
# so the model and store are loaded once and reused for every search call.
_embeddings   = None   # will hold the HuggingFaceEmbeddings instance
_vector_store = None   # will hold the Chroma instance


def _get_store():
    """
    Return the (lazily initialised) vector store.

    On the first call: loads the embedding model and opens ChromaDB.
    On subsequent calls: returns the already-loaded objects immediately.

    This is a private function (leading underscore convention) —
    only called from within this module.

    Returns:
        Chroma: the ready-to-query vector store
    """
    # "global" keyword lets us reassign the module-level variables from inside
    # a function. Without it, Python would create local variables instead.
    global _embeddings, _vector_store

    # Only initialise if we haven't done so yet in this process
    if _vector_store is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vector_store = Chroma(
            collection_name=COLLECTION,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )

    return _vector_store


# ── Main search function ───────────────────────────────────────────────────────

def search(query: str, k: int = 5) -> list[dict]:
    """
    Perform a semantic search against the indexed files.

    Steps:
      1. Embed the query string into a 384-dim vector
      2. Ask ChromaDB for the k most similar stored vectors
      3. Return results as clean dicts (not raw LangChain objects)

    The returned relevance score is a cosine similarity value between 0 and 1.
    Higher = more relevant. In practice:
      > 0.7  = strong match
      0.4-0.7 = reasonable match
      < 0.4  = weak, possibly off-topic

    Args:
        query: the user's natural language search query
        k:     number of results to return (default 5, more = broader results)

    Returns:
        list[dict]: each dict has keys:
            "content"  — the actual text of the chunk
            "source"   — full file path
            "filename" — just the filename (e.g. "notes.md")
            "filetype" — ".md", ".pdf", etc.
            "score"    — relevance score 0.0–1.0
    """
    store = _get_store()

    try:
        # similarity_search_with_relevance_scores returns:
        #   list of (Document, float) tuples
        # where the float is the cosine similarity score (0.0 to 1.0)
        results = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as e:
        # Could happen if ChromaDB files are corrupted or index is empty
        print(f"[Retriever] Search error: {e}")
        return []

    # ── Convert LangChain Document objects into plain dicts ───────────────────
    # We don't want to expose LangChain internals to the MCP server layer.
    # Plain dicts are simpler to work with and serialise.
    output = []
    for doc, score in results:
        output.append({
            "content":  doc.page_content,                        # the chunk text
            "source":   doc.metadata.get("source",   "unknown"), # full path
            "filename": doc.metadata.get("filename", "unknown"), # just the name
            "filetype": doc.metadata.get("filetype", ""),        # extension
            "score":    round(score, 3),                         # round to 3 decimal places
        })

    return output


# ── Utility: list all indexed filenames ───────────────────────────────────────

def list_indexed_files() -> list[str]:
    """
    Return a sorted list of all unique filenames currently in the index.

    Useful for the list_files MCP tool, so the LLM (and the user) can
    see what documents are available to search before making a query.

    How it works:
      - ChromaDB's .get() with no filters returns ALL stored chunks.
      - Each chunk has a metadata dict with a "filename" field.
      - We collect those into a set (to deduplicate) and sort alphabetically.

    Returns:
        list[str]: sorted list of filenames, e.g. ["notes.md", "report.pdf"]
    """
    store = _get_store()
    try:
        # .get() with no arguments fetches all documents in the collection.
        # Returns a dict with keys: "ids", "embeddings", "documents", "metadatas"
        all_docs = store.get()

        # Use a set to automatically deduplicate — a file with 50 chunks
        # only appears once in the final list.
        filenames = set()
        for meta in all_docs.get("metadatas", []):
            # meta can be None for chunks with no metadata (defensive check)
            if meta and meta.get("filename"):
                filenames.add(meta["filename"])

        return sorted(filenames)   # alphabetical order for readability

    except Exception as e:
        print(f"[Retriever] List error: {e}")
        return []