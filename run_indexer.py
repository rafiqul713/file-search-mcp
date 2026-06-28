"""
=============================================================================
run_indexer.py — One-Time Index Builder (CLI script)
=============================================================================

WHAT THIS FILE DOES:
--------------------
This is the script you run ONCE (and whenever you add new files) to build
the search index. It scans the data/ directory, processes every supported
file, and stores the embeddings in ChromaDB.

You must run this BEFORE starting the MCP server, otherwise there will
be nothing to search.

WHEN TO RUN IT:
---------------
  1. First time setup (after adding your files to data/)
  2. After adding new files to data/
  3. After manually editing files (or let the watcher handle this automatically)

USAGE:
------
  source .venv/bin/activate
  python run_indexer.py
=============================================================================
"""

from dotenv import load_dotenv

# Import the main indexing function from our indexer module
from src.indexer import index_directory

# Load environment variables (.env) so index_directory can read FILES_DIR etc.
load_dotenv()

if __name__ == "__main__":
    # Guard clause: only run when this file is executed directly.
    # This prevents accidental execution if another file imports this module.

    print("=" * 55)
    print("  Personal File Search — Indexer")
    print("=" * 55)
    print("Scanning data/ directory for .txt, .md, .pdf files...")
    print("(First run downloads the embedding model ~90MB)\n")

    # index_directory() does all the heavy lifting:
    #   1. Scans FILES_DIR (./data by default) for supported files
    #   2. Skips files already indexed (uses file hash comparison)
    #   3. Loads, chunks, embeds, and stores new/changed files in ChromaDB
    index_directory()

    print("\n" + "=" * 55)
    print("  Indexing complete!")
    print("  You can now run: python src/server.py")
    print("  Or test with:    python test_search.py")
    print("=" * 55)