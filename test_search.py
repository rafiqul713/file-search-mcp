"""
=============================================================================
test_search.py — Local Test Script (No LLM Client Needed)
=============================================================================

WHAT THIS FILE DOES:
--------------------
Tests all 4 MCP tools directly in the terminal, without needing to start
Claude Desktop or any other MCP client. Useful for:

  - Verifying the index was built correctly
  - Checking that search returns sensible results
  - Debugging issues before connecting to Claude Desktop
  - Quickly trying out your data without setting up a full MCP client

USAGE:
------
  source .venv/bin/activate
  python test_search.py

You will be prompted to enter a search query. Press Enter to use a demo query.
=============================================================================
"""

from dotenv import load_dotenv

# Import the tool functions directly from the server module.
# In production, Claude Desktop calls these via the MCP protocol.
# Here we call them directly as normal Python functions — same code, no protocol.
from src.server import search_files, list_files, summarise_topic, read_file

# Load environment variables from .env before calling any tools
load_dotenv()


def divider(title: str):
    """Print a formatted section divider for readable terminal output."""
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


if __name__ == "__main__":
    print("\nPersonal File Search — Local Test")
    print("Tests all 4 MCP tools without needing Claude Desktop.\n")

    # ── Test 1: list_files ────────────────────────────────────────────────────
    # This is always the first thing to check — confirms the index isn't empty.
    divider("TOOL 1: list_files()")
    print(list_files())
    # Expected output: a bullet list of filenames
    # If you see "No files indexed yet" → run python run_indexer.py first

    # ── Test 2: search_files ──────────────────────────────────────────────────
    divider("TOOL 2: search_files()")
    query = input("\nEnter a search query (press Enter for 'main ideas'): ").strip()
    if not query:
        query = "main ideas"   # sensible default if user just presses Enter

    print(f"\nSearching for: '{query}'")
    print("-" * 40)
    # Call with max_results=3 so the terminal output stays manageable
    print(search_files(query, max_results=3))

    # ── Test 3: summarise_topic ───────────────────────────────────────────────
    # This makes a real Gemini API call — requires GOOGLE_API_KEY in .env
    divider("TOOL 3: summarise_topic()")
    print(f"Summarising topic: '{query}'")
    print("(This calls Gemini — may take 5-10 seconds...)")
    print("-" * 40)
    print(summarise_topic(query))

    # ── Test 4: read_file ─────────────────────────────────────────────────────
    divider("TOOL 4: read_file()")
    filename = input("\nEnter a filename to read (e.g. notes.md), or press Enter to skip: ").strip()
    if filename:
        print(f"\nReading: {filename}")
        print("-" * 40)
        print(read_file(filename))
    else:
        print("Skipped.")

    print("\n" + "=" * 55)
    print("  All tests complete!")
    print("=" * 55)