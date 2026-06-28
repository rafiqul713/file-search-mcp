"""
=============================================================================
indexer.py — The RAG Indexing Pipeline
=============================================================================

WHAT THIS FILE DOES:
--------------------
This file handles Phase 1 of the RAG pipeline: reading files from disk,
breaking them into small chunks, converting those chunks into numerical
vectors (embeddings), and storing everything in ChromaDB so it can be
searched later.

Think of it like building a library catalogue — except instead of keywords,
every piece of text is stored as a vector so we can find it by meaning.

FLOW:
  Your files (.txt / .md / .pdf)
      ↓
  LangChain document loaders  (read raw text from disk)
      ↓
  RecursiveCharacterTextSplitter  (cut text into ~500-char chunks)
      ↓
  HuggingFaceEmbeddings  (convert each chunk to a 384-dim vector)
      ↓
  ChromaDB  (store vectors + original text on disk)

WHY CHUNKS?
-----------
LLMs have a limited context window. A 50,000-token file can't be fed in
whole. By chunking, we retrieve only the 5-6 most relevant pieces,
keeping context small and focused.

WHY EMBEDDINGS?
---------------
Embeddings turn text into numbers that capture semantic meaning.
"AI impacts jobs" and "artificial intelligence affects employment"
produce similar vectors even though the words differ — that's what
makes search semantic rather than keyword-based.
=============================================================================
"""

import os
import hashlib                    # used to fingerprint files for change detection
from pathlib import Path          # cleaner cross-platform file path handling
from dotenv import load_dotenv    # reads key=value pairs from .env file into os.environ

# LangChain loaders — each one reads a different file format into Document objects.
# A Document has two fields: page_content (the text) and metadata (a dict of info).
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# RecursiveCharacterTextSplitter — intelligently splits text into chunks.
# It tries splitting at paragraph breaks first, then sentences, then words,
# so chunks are more coherent than simply cutting every N characters.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFaceEmbeddings — runs a local sentence-transformer model that converts
# text strings into dense float vectors.
# "all-MiniLM-L6-v2" is small (90MB), fast, free, and works well for search.
# Runs 100% locally — no API key, no internet needed after first download.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma — the vector database that stores text chunks + their embeddings on disk.
# ChromaDB is embedded (no server needed), stores files in a local folder,
# and survives restarts.
from langchain_community.vectorstores import Chroma

# Load .env file so os.getenv() can read our configuration values
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
# os.getenv("KEY", "default") reads from environment/.env, with a fallback default.
# This pattern lets you change paths without editing source code.

FILES_DIR  = os.getenv("FILES_DIR",       "./data")       # where your documents live
CHROMA_DIR = os.getenv("CHROMA_DIR",      "./chroma_db")  # where ChromaDB saves its files
COLLECTION = os.getenv("COLLECTION_NAME", "my_files")     # logical collection name inside ChromaDB

# The embedding model name. "all-MiniLM-L6-v2" produces 384-dimensional vectors.
# Downloaded once from HuggingFace and cached in ~/.cache/huggingface/
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Helper: load the embedding model ──────────────────────────────────────────

def get_embeddings():
    """
    Load and return the sentence-transformer embedding model.

    Why HuggingFace instead of Gemini embeddings?
    Because this runs 100% locally with no API cost. We embed potentially
    thousands of chunks during indexing, so free local embeddings matter.

    The model is downloaded once (~90MB) and cached. All subsequent calls
    load it from the local cache instantly.

    Returns:
        HuggingFaceEmbeddings: LangChain-compatible embedding model
    """
    print("[Indexer] Loading embedding model (first run downloads ~90MB)...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL   # tells HuggingFace which model weights to load
    )


# ── Helper: get or create the ChromaDB vector store ───────────────────────────

def get_vector_store(embeddings):
    """
    Return the ChromaDB vector store, creating it on disk if it doesn't exist.

    ChromaDB stores three things per chunk:
      1. The original text (so we can return readable results)
      2. The embedding vector (so we can do similarity search)
      3. Metadata dict (filename, source path, file hash, filetype)

    persist_directory means the data survives between Python sessions —
    you don't have to re-index every time you restart.

    Args:
        embeddings: the embedding model — used both when adding docs and
                    when embedding search queries later

    Returns:
        Chroma: LangChain-compatible vector store
    """
    return Chroma(
        collection_name=COLLECTION,     # logical namespace — keeps collections separate
        embedding_function=embeddings,  # how to convert text → vector
        persist_directory=CHROMA_DIR,   # folder where ChromaDB writes its SQLite + bin files
    )


# ── Helper: fingerprint a file for change detection ───────────────────────────

def file_hash(filepath: str) -> str:
    """
    Compute the MD5 hash of a file's raw bytes.

    This fingerprint lets us check at index time whether a file has changed
    since we last indexed it. We store the hash as metadata in ChromaDB.
    On subsequent runs, if the stored hash matches the current file hash,
    we skip re-indexing — saving time and avoiding duplicate chunks.

    MD5 isn't cryptographically secure, but for change detection it's
    fast and more than sufficient.

    Args:
        filepath: path to the file

    Returns:
        str: 32-character hex string, e.g. "d41d8cd98f00b204e9800998ecf8427e"
    """
    with open(filepath, "rb") as f:         # "rb" = read binary, works for PDFs too
        return hashlib.md5(f.read()).hexdigest()


# ── Helper: load one file into LangChain Document objects ─────────────────────

def load_file(filepath: str):
    """
    Load a single file using the appropriate LangChain loader.

    Each loader returns a list of Document objects:
      - TextLoader  → one Document for the whole file
      - PyPDFLoader → one Document per page (useful for large PDFs)

    We choose the loader based on file extension.

    Args:
        filepath: path to the file

    Returns:
        list[Document]: loaded documents, or [] on failure
    """
    ext = Path(filepath).suffix.lower()  # ".pdf", ".txt", ".md", etc.

    try:
        if ext == ".pdf":
            # PyPDFLoader extracts text from each PDF page separately.
            # metadata will include {"page": 0, "source": "..."}
            loader = PyPDFLoader(filepath)
        else:
            # TextLoader reads the entire file as one Document.
            # encoding="utf-8" supports most languages including German (ä, ö, ü, ß).
            loader = TextLoader(filepath, encoding="utf-8")

        return loader.load()   # triggers actual file read

    except Exception as e:
        # Graceful failure — log the error but don't crash the whole indexer.
        # One bad file shouldn't stop all others from being indexed.
        print(f"[Indexer] Could not load {filepath}: {e}")
        return []


# ── Main function: index an entire directory ───────────────────────────────────

def index_directory(files_dir: str = FILES_DIR):
    """
    Walk the files directory, chunk every supported file, and upsert into ChromaDB.

    This is the primary function you call to build or update the search index.
    It uses file hashes to skip files that haven't changed, so repeated runs
    only process new or modified files.

    Supported types: .txt  .md  .pdf

    Args:
        files_dir: path to the directory containing your documents

    Returns:
        Chroma | None: the vector store object, or None if no files were found
    """
    supported = {".txt", ".md", ".pdf"}
    files_dir = Path(files_dir)

    # Create the data directory if the user hasn't made it yet
    if not files_dir.exists():
        print(f"[Indexer] Creating files directory: {files_dir}")
        files_dir.mkdir(parents=True)   # parents=True creates intermediate directories too

    # rglob("*") is a recursive glob — finds all files in this folder AND subfolders.
    # The list comprehension filters to supported extensions only.
    all_files = [
        f for f in files_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ]

    if not all_files:
        print(f"[Indexer] No supported files found in {files_dir}")
        print("[Indexer] Add .txt / .md / .pdf files to the data/ folder and re-run.")
        return

    # ── Set up the embedding model and vector store ───────────────────────────
    embeddings   = get_embeddings()
    vector_store = get_vector_store(embeddings)

    # ── Configure the text splitter ───────────────────────────────────────────
    # chunk_size=500     → each chunk is at most 500 characters long.
    #                      ~500 chars ≈ 100-125 tokens — a good size for retrieval.
    # chunk_overlap=50   → consecutive chunks share 50 characters at their boundary.
    #                      Overlap prevents important context from being split exactly
    #                      at a chunk edge and lost during retrieval.
    # separators         → the splitter tries these in order, preferring natural breaks:
    #                        "\n\n" = paragraph break (most natural)
    #                        "\n"   = line break
    #                        "."    = sentence end
    #                        " "    = word boundary (last resort)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )

    indexed = 0   # counter: files newly indexed this run
    skipped = 0   # counter: files skipped because they haven't changed

    # ── Process each file ─────────────────────────────────────────────────────
    for filepath in all_files:

        # Step 1: compute this file's fingerprint
        fhash = file_hash(str(filepath))

        # Step 2: check if ChromaDB already has chunks from this exact file version.
        # "where" is a metadata filter — ChromaDB only returns docs whose
        # "file_hash" metadata field equals fhash.
        # limit=1 means we stop as soon as we find one match (fast).
        existing = vector_store.get(
            where={"file_hash": fhash},
            limit=1,
        )
        if existing and existing.get("ids"):
            # This file version is already in the index — nothing to do
            skipped += 1
            continue

        print(f"[Indexer] Indexing: {filepath.name}")

        # Step 3: load the file into LangChain Document objects
        docs = load_file(str(filepath))
        if not docs:
            continue   # skip this file if loading failed

        # Step 4: split all Documents into chunks.
        # split_documents handles lists of Documents and respects page
        # boundaries in multi-page PDFs.
        chunks = splitter.split_documents(docs)

        # Step 5: enrich each chunk's metadata.
        # Metadata stored here is saved in ChromaDB alongside the embedding.
        # It's used for:
        #   - Showing the user which file a result came from
        #   - Filtering searches to specific files or types
        #   - Detecting file changes (file_hash)
        for chunk in chunks:
            chunk.metadata["source"]    = str(filepath)           # full absolute path
            chunk.metadata["filename"]  = filepath.name           # "notes.md"
            chunk.metadata["file_hash"] = fhash                   # MD5 fingerprint
            chunk.metadata["filetype"]  = filepath.suffix.lower() # ".md", ".pdf", etc.

        # Step 6: add chunks to ChromaDB.
        # Internally, ChromaDB:
        #   a) calls the embedding model on each chunk's text
        #   b) stores the resulting vector + text + metadata in its database
        vector_store.add_documents(chunks)
        indexed += 1

    print(f"[Indexer] Done — {indexed} file(s) indexed, {skipped} unchanged.")
    return vector_store


# ── Single-file indexer (called by the file watcher on changes) ───────────────

def index_single_file(filepath: str):
    """
    Re-index a single file — called automatically by watcher.py when
    a file is created or modified in the data directory.

    Important: before indexing, we DELETE the old chunks for this file
    to avoid duplicate content building up in ChromaDB over time.

    Args:
        filepath: path to the file that changed
    """
    filepath = Path(filepath)

    # Ignore unsupported file types that the OS might create (.DS_Store, etc.)
    if filepath.suffix.lower() not in {".txt", ".md", ".pdf"}:
        return

    print(f"[Indexer] Re-indexing changed file: {filepath.name}")

    embeddings   = get_embeddings()
    vector_store = get_vector_store(embeddings)
    splitter     = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # ── Delete old chunks for this file ───────────────────────────────────────
    # We search ChromaDB for all chunks whose "source" metadata matches this
    # file path. Then we delete them by their IDs before adding fresh chunks.
    # Without this step, every file edit would ADD duplicate chunks, degrading
    # search quality over time.
    existing = vector_store.get(
        where={"source": str(filepath)},
    )
    if existing and existing.get("ids"):
        vector_store.delete(ids=existing["ids"])
        print(f"[Indexer] Removed {len(existing['ids'])} stale chunks for {filepath.name}")

    # Load, chunk, and re-index with fresh content
    docs   = load_file(str(filepath))
    chunks = splitter.split_documents(docs)
    fhash  = file_hash(str(filepath))   # new hash reflects updated file

    for chunk in chunks:
        chunk.metadata["source"]    = str(filepath)
        chunk.metadata["filename"]  = filepath.name
        chunk.metadata["file_hash"] = fhash
        chunk.metadata["filetype"]  = filepath.suffix.lower()

    vector_store.add_documents(chunks)
    print(f"[Indexer] Re-indexed {len(chunks)} chunks from {filepath.name}")