"""
=============================================================================
server.py — The MCP Server (Main Entry Point)
=============================================================================

WHAT THIS FILE DOES:
--------------------
This is the heart of the project. It creates an MCP (Model Context Protocol)
server that exposes 4 tools any compatible LLM client can call:

  1. search_files     — semantic search across all indexed files
  2. read_file        — read the full content of a specific file
  3. summarise_topic  — search + LLM-powered synthesis with Gemini
  4. list_files       — list all currently indexed files

WHAT IS MCP?
------------
MCP (Model Context Protocol) is an open standard created by Anthropic.
It defines a contract between an LLM client (Claude Desktop, LangGraph, etc.)
and a "tool server" (this file). The LLM can discover available tools,
call them with arguments, and receive structured results — all via a
standardised protocol over stdio (standard input/output).

Think of it like a REST API, but designed specifically for LLMs:
  - REST:  client sends HTTP request → server returns JSON
  - MCP:   LLM sends tool call → server returns text/structured data

WHY FASTMCP?
------------
FastMCP is the high-level interface from the official MCP Python SDK.
It reads your Python function signatures and docstrings to automatically
generate the JSON schema that tells the LLM what arguments each tool expects.
You write normal Python functions — FastMCP handles the protocol layer.

TRANSPORT: STDIO
----------------
This server communicates via stdio (standard input/output streams).
When Claude Desktop launches it as a subprocess, it writes JSON-RPC
messages to stdin and reads responses from stdout. This is the simplest
MCP transport — no ports, no HTTP, no auth required.
=============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# FastMCP — the high-level MCP server class from Anthropic's official Python SDK.
# It handles:
#   - Tool discovery (listing tools and their schemas to the client)
#   - Message routing (calling the right function when the LLM uses a tool)
#   - Protocol serialisation (converting Python return values to MCP responses)
#   - Transport management (stdio communication with the LLM client)
from mcp.server.fastmcp import FastMCP

# Our retrieval functions from the RAG layer.
# The MCP tools call these to do the actual work.
from src.retriever import search, list_indexed_files

# Load API keys and config from .env
load_dotenv()

FILES_DIR = os.getenv("FILES_DIR", "./data")  # where user's files live on disk

# ── Create the MCP server instance ────────────────────────────────────────────
# The name and description are shown in Claude Desktop's tool panel,
# so the user (and the LLM) know what this server is for.
mcp = FastMCP(
    "Personal File Search",
    description=(
        "Search, read, and summarise your local files using natural language. "
        "Files must be indexed first with run_indexer.py."
    ),
)


# =============================================================================
# TOOL 1: search_files
# =============================================================================

@mcp.tool()
def search_files(query: str, max_results: int = 5) -> str:
    """
    Search your local files semantically using natural language.

    Returns the most relevant text chunks with their source filenames
    and relevance scores. Use this when the user asks to 'find',
    'search', or 'look for' something in their files.

    Unlike keyword search (grep), this understands meaning — searching
    for "machine learning" will also find chunks about "neural networks"
    and "AI models" even if those exact words aren't in the query.

    Args:
        query:       What to search for, described in plain English.
                     Example: "notes about Python decorators"
        max_results: How many results to return. Default 5. Max 10.
                     More results = broader coverage, more noise.
                     Fewer results = higher precision.

    Returns:
        str: Formatted search results with filename, relevance score,
             and the matching text content for each result.
    """
    # Cap at 10 to avoid flooding the LLM context window with low-quality results
    max_results = min(max_results, 10)

    # Call our retriever — returns list of dicts with content, filename, score
    results = search(query, k=max_results)

    # Handle the case where the index is empty or no matches found
    if not results:
        return (
            "No results found. Make sure files are indexed by running "
            "'python run_indexer.py' first, and that the data/ folder "
            "contains .txt, .md, or .pdf files."
        )

    # ── Format results as readable text ───────────────────────────────────────
    # We return plain text (not JSON) because:
    #   a) The LLM reads plain text naturally
    #   b) It's easier to display in Claude Desktop's chat interface
    lines = [f"Found {len(results)} result(s) for: '{query}'\n"]

    for i, r in enumerate(results, 1):
        lines.append(
            f"--- Result {i} ---\n"
            f"File:      {r['filename']}\n"
            f"Relevance: {r['score']}\n"   # 0.0–1.0, higher is more relevant
            f"Content:\n{r['content']}\n"
        )

    # "\n".join() combines all lines into one string with newlines between them
    return "\n".join(lines)


# =============================================================================
# TOOL 2: read_file
# =============================================================================

@mcp.tool()
def read_file(filename: str) -> str:
    """
    Read the full content of a specific file from the data directory.

    Use this after search_files to read a complete document when the
    user wants to see the full context, not just a matching chunk.
    Use list_files first if you are unsure of the exact filename.

    Args:
        filename: The name of the file to read.
                  Examples: 'notes.md', 'report.pdf', 'ideas.txt'
                  Do not include the full path — just the filename.

    Returns:
        str: The full text content of the file, prefixed with the filename
             as a heading. Truncated at 20,000 characters for very large files.
    """
    files_dir = Path(FILES_DIR)

    # rglob searches the data directory and all subdirectories for a file
    # with this name. This means the user doesn't need to know the subfolder.
    matches = list(files_dir.rglob(filename))

    if not matches:
        # Helpful error message: show what files ARE available so the LLM
        # can correct itself and try again with the right filename.
        available = [f.name for f in files_dir.rglob("*") if f.is_file()]
        return (
            f"File '{filename}' not found in {FILES_DIR}.\n"
            f"Available files: {', '.join(available) if available else 'none — run the indexer first'}"
        )

    # Take the first match if multiple files share the same name
    filepath = matches[0]
    ext      = filepath.suffix.lower()

    try:
        if ext == ".pdf":
            # PDFs need special handling — we can't just open them as text.
            # PyPDFLoader extracts text from each page and returns Documents.
            # We join all pages into one string separated by double newlines.
            from langchain_community.document_loaders import PyPDFLoader
            docs    = PyPDFLoader(str(filepath)).load()
            content = "\n\n".join(d.page_content for d in docs)
        else:
            # .txt and .md are plain text — read directly with Python's built-in
            # Path.read_text(). encoding="utf-8" handles special characters.
            content = filepath.read_text(encoding="utf-8")

        # ── Safety: truncate very large files ─────────────────────────────────
        # LLMs have limited context windows. Sending a 200,000-char file
        # would exceed the context limit and likely cause errors.
        # 20,000 chars ≈ ~5,000 tokens — a safe limit.
        if len(content) > 20_000:
            content = content[:20_000] + "\n\n[... file truncated at 20,000 chars ...]"

        # Return with a markdown heading so the LLM knows which file this is
        return f"# {filepath.name}\n\n{content}"

    except Exception as e:
        return f"Error reading '{filename}': {e}"


# =============================================================================
# TOOL 3: summarise_topic
# =============================================================================

@mcp.tool()
def summarise_topic(topic: str) -> str:
    """
    Search your files for a topic and return a Gemini-written summary.

    More powerful than search_files — instead of returning raw chunks,
    this tool finds relevant content AND asks Gemini to synthesise it
    into a coherent, readable answer.

    Use this when the user asks:
      - "What do my notes say about X?"
      - "Summarise everything I have on Y"
      - "Give me an overview of Z from my files"

    Two-step process:
      Step 1: RAG retrieval — find the 6 most relevant chunks from ChromaDB
      Step 2: LLM synthesis — Gemini reads those chunks and writes a summary

    Args:
        topic: The subject or question to research across the user's files.
               Examples: "machine learning projects", "meeting notes from Q1"

    Returns:
        str: A 150-200 word summary written by Gemini, citing which files
             the information came from. Falls back to raw chunks if Gemini fails.
    """
    # ── Step 1: retrieve relevant chunks from ChromaDB ────────────────────────
    # k=6 gives us slightly more context than search_files (k=5) since we're
    # passing it to Gemini which can handle more text than a single LLM call.
    results = search(topic, k=6)

    if not results:
        return f"No content found about '{topic}' in your indexed files."

    # ── Step 2: build the context string to send to Gemini ───────────────────
    # We format each chunk with its source filename so Gemini can mention
    # which files the information came from in its summary.
    context_parts = []
    for r in results:
        context_parts.append(
            f"[From: {r['filename']}]\n{r['content']}"
        )
    # Join chunks with a visible separator so Gemini knows where one ends
    # and the next begins
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: call Gemini to synthesise a summary ───────────────────────────
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # temperature=0.2 → low creativity, high factual accuracy.
        # For summarisation, we want the LLM to stick to the provided text,
        # not hallucinate additional information.
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # The prompt explicitly tells Gemini to:
        #   a) Base its answer only on the provided excerpts (no hallucination)
        #   b) Cite the source filenames
        #   c) Stay concise (150-200 words)
        prompt = (
            f"The user wants to know about: '{topic}'\n\n"
            f"Here are relevant excerpts from their personal files:\n\n"
            f"{context}\n\n"
            f"Write a clear, concise summary (150-200 words) that answers "
            f"the user's question based ONLY on the excerpts above. "
            f"Do not add information that isn't in the excerpts. "
            f"Mention which files the information came from."
        )

        response = llm.invoke(prompt)
        return response.content   # the generated summary text

    except Exception as e:
        # Graceful fallback: if Gemini fails (network error, rate limit, etc.)
        # return the raw search results so the user still gets something useful.
        print(f"[Server] Gemini summarisation failed: {e}")
        return (
            f"Gemini summarisation failed ({e}).\n\n"
            f"Here are the raw search results instead:\n\n{context}"
        )


# =============================================================================
# TOOL 4: list_files
# =============================================================================

@mcp.tool()
def list_files() -> str:
    """
    List all files currently indexed and available to search.

    Use this at the start of a conversation to understand what documents
    are available, or before calling read_file to get the exact filename.

    Returns:
        str: A formatted list of all indexed filenames, with a total count.
             Returns a helpful message if no files have been indexed yet.
    """
    # Calls retriever.py's list_indexed_files() which queries ChromaDB
    # for all unique "filename" metadata values across the collection.
    files = list_indexed_files()

    if not files:
        return (
            "No files are indexed yet.\n"
            "To get started:\n"
            "  1. Add .txt, .md, or .pdf files to the data/ directory\n"
            "  2. Run: python run_indexer.py\n"
            "  3. Then try this tool again."
        )

    # Format as a bullet list — easy to read in Claude Desktop's chat interface
    lines = [f"Indexed files ({len(files)} total):\n"]
    for f in files:
        lines.append(f"  • {f}")

    return "\n".join(lines)


# =============================================================================
# Entry point — runs when you execute: python src/server.py
# =============================================================================

if __name__ == "__main__":
    print("[MCP] Starting Personal File Search server...")
    print("[MCP] Registered tools:")
    print("[MCP]   • search_files     — semantic search across your files")
    print("[MCP]   • read_file        — read a complete file")
    print("[MCP]   • summarise_topic  — search + Gemini summary")
    print("[MCP]   • list_files       — see what's indexed")
    print("[MCP] Waiting for connections via stdio...\n")

    # transport="stdio" means communication happens via standard input/output.
    # Claude Desktop launches this script as a subprocess and communicates
    # by writing JSON-RPC messages to its stdin and reading from its stdout.
    # No ports, no HTTP server, no authentication needed.
    mcp.run(transport="stdio")