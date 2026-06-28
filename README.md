# Personal File Search MCP Server 🔍

A locally-running MCP (Model Context Protocol) server that turns your personal files into a searchable knowledge base. Ask questions in plain English and get answers pulled directly from your own documents — no cloud storage, no data leaving your machine.

Built to demonstrate RAG (Retrieval-Augmented Generation), vector search, and MCP server development using the official Anthropic SDK.

---

## What it actually does

Drop any `.txt`, `.md`, or `.pdf` files into the `data/` folder. The indexer reads them, breaks them into chunks, converts each chunk into a numerical vector using a local AI model, and stores everything in ChromaDB — a local vector database.

Once indexed, you can connect the server to Claude Desktop and ask things like:

- *"What do my notes say about machine learning?"*
- *"Summarise everything I have on Project Apollo"*
- *"Find my meeting notes from last quarter"*
- *"Read the full content of report.pdf"*

The search works by **meaning**, not keywords. So searching for "AI impacts employment" will find a document that says "artificial intelligence is changing the job market" — even though none of those exact words match.

---

## How it works under the hood

### Phase 1 — Indexing (run once)

```
Your files (.txt / .md / .pdf)
    ↓
LangChain loaders      read raw text from disk
    ↓
Text splitter          cut into ~500-character chunks with overlap
    ↓
Embedding model        convert each chunk into a 384-dim vector (runs locally)
    ↓
ChromaDB               store vectors + text + metadata on disk
```

### Phase 2 — Search (every query)

```
Your question (plain English)
    ↓
Embedding model        convert query into a vector
    ↓
ChromaDB               find the most similar stored vectors (cosine similarity)
    ↓
Gemini 2.5 Flash       optionally synthesise results into a clean summary
    ↓
Answer                 returned to Claude Desktop via MCP protocol
```

---

## The 4 tools exposed via MCP

| Tool | What it does |
|---|---|
| `search_files` | Semantic search — returns the most relevant chunks with relevance scores |
| `read_file` | Reads the complete content of a specific file |
| `summarise_topic` | Searches files AND synthesises results into a Gemini-written summary |
| `list_files` | Lists all files currently indexed and available to search |

Claude Desktop automatically decides which tool to call based on the conversation. You just ask naturally.

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| MCP server | `mcp[cli]` (Anthropic SDK) | Official protocol — works with any MCP-compatible LLM |
| RAG framework | `LangChain` | Document loaders, text splitters, vector store interface |
| Vector database | `ChromaDB` | Local, no server needed, persists to disk |
| Embedding model | `sentence-transformers` (all-MiniLM-L6-v2) | Runs 100% locally, no API cost, 90MB one-time download |
| LLM (summarise) | `Gemini 2.5 Flash` | Free tier, used only for the summarise tool |
| File watcher | `watchdog` | Auto re-indexes when files change |
| Config | `python-dotenv` | Keeps API keys out of source code |

Everything runs locally except the Gemini summarise call.

---

## Project structure

```
file-search-mcp/
├── src/
│   ├── __init__.py      marks src/ as a Python package
│   ├── indexer.py       RAG indexing pipeline — load, chunk, embed, store
│   ├── retriever.py     semantic search against ChromaDB
│   ├── server.py        MCP server — defines and exposes the 4 tools
│   └── watcher.py       file system watcher for auto re-indexing
├── data/                put your .txt / .md / .pdf files here
├── chroma_db/           ChromaDB database (auto-created, do not commit)
├── run_indexer.py       run this to build or update the search index
├── test_search.py       test all tools locally without Claude Desktop
├── requirements.txt     Python dependencies
├── .env         copy to .env and fill in your API keys
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites

- macOS, Linux, or Windows (WSL recommended on Windows)
- Python 3.11 or higher
- An internet connection (for first-time model download and Gemini calls)

Check your Python version:

```bash
python3 --version
```

If it shows 3.9 or lower, install 3.11 via Homebrew (macOS):

```bash
brew install python@3.11
```

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/rafiqul713/file-search-mcp.git
cd file-search-mcp
```

---

### Step 2 — Create a virtual environment

A virtual environment is an isolated Python installation just for this project. It keeps dependencies completely separate from your system Python and from other projects — no version conflicts, no broken installs.

```bash
# Create the virtual environment in a folder called .venv
python3.11 -m venv .venv
```

Now activate it. You must do this every time you open a new terminal:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

When the virtual environment is active, your terminal prompt will show `(.venv)` at the start:

```
(.venv) rafiqul@Mac file-search-mcp %
```

That `(.venv)` tells you that any `python` or `pip` command runs inside the isolated environment, not your system Python.

To deactivate (when you're done working):
```bash
deactivate
```

---

### Step 3 — Install dependencies

With the virtual environment active:

```bash
# Upgrade pip first (avoids some install issues)
pip install --upgrade pip

# Install all dependencies
pip install --timeout 120 -r requirements.txt
```

The first time this runs, it will download the `sentence-transformers` embedding model (~90MB from HuggingFace). This only happens once — it's cached locally after that.

If any package times out, install it alone with a longer timeout:
```bash
pip install --timeout 300 sentence-transformers
```

---

### Step 4 — Get your free API keys

You only need one API key for the summarise feature. Everything else (search, read, list) runs completely locally.

**Google Gemini (free)**

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **Create API key**
4. Copy the key — it starts with `AIza...`

---

### Step 5 — Configure your environment

```

Open `.env` in your editor:

```bash
nano .env        # terminal editor
# or
code .env        # VS Code
```

Fill in your values:

```
GOOGLE_API_KEY=AIzaSy...your_key_here...
FILES_DIR=./data
CHROMA_DIR=./chroma_db
COLLECTION_NAME=my_files
```

Save and close. Your `.env` is listed in `.gitignore` so it will never be accidentally committed to GitHub.

---

### Step 6 — Add your files

Put any `.txt`, `.md`, or `.pdf` files you want to search into the `data/` folder:

```bash
# The folder is created automatically, but you can make it manually
mkdir -p data

# Copy some files in
cp ~/Documents/notes.md data/
cp ~/Downloads/report.pdf data/
```

Or create a test file to try right away:

```bash
cat > data/sample.md << 'EOF'
# My Notes

## LangGraph
LangGraph is a library for building stateful multi-agent AI applications.
It uses a graph approach where nodes are functions and edges define flow.

## RAG
Retrieval Augmented Generation combines semantic search with LLM generation.
Instead of relying on training data, the LLM gets relevant context injected.

## MCP
Model Context Protocol is an open standard by Anthropic.
It lets LLMs connect to external tools in a standardised way.
EOF
```

---

### Step 7 — Build the index

```bash
python run_indexer.py
```

You will see output like:

```
=======================================================
  Personal File Search — Indexer
=======================================================
Scanning data/ directory for .txt, .md, .pdf files...
(First run downloads the embedding model ~90MB)

[Indexer] Loading embedding model (first run downloads ~90MB)...
[Indexer] Indexing: sample.md
[Indexer] Done — 1 file(s) indexed, 0 unchanged.
```

Running it again later only re-indexes new or changed files — unchanged files are skipped.

---

### Step 8 — Test that everything works

Before setting up Claude Desktop, verify search is working locally:

```bash
python test_search.py
```

This tests all 4 tools directly in your terminal. You should see your indexed files listed and get real search results back.

---

### Step 9 — Connect to Claude Desktop

Download Claude Desktop from [claude.ai/download](https://claude.ai/download) if you haven't already.

Open the Claude Desktop configuration file:

```bash
# macOS
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Add this configuration (replace the paths with your actual project location):

```json
{
  "mcpServers": {
    "file-search": {
      "command": "/path/to/file-search-mcp/.venv/bin/python",
      "args": [
        "/path/to/file-search-mcp/src/server.py"
      ],
      "env": {
        "GOOGLE_API_KEY": "your_gemini_key_here",
        "FILES_DIR": "/path/to/file-search-mcp/data",
        "CHROMA_DIR": "/path/to/file-search-mcp/chroma_db",
        "COLLECTION_NAME": "my_files"
      }
    }
  }
}
```

To find your exact project path:
```bash
pwd
```

Restart Claude Desktop. You should see a hammer icon (🔨) in the chat input area — that confirms your MCP tools are connected.

Now try asking Claude:

> *"What files do you have access to?"*

> *"Search my files for anything about LangGraph"*

> *"Summarise what my notes say about RAG"*

---

## Keeping the index up to date

**Option A — Manual** (run after adding or editing files):

```bash
source .venv/bin/activate
python run_indexer.py
```

**Option B — Automatic watcher** (runs in the background, re-indexes on every save):

```bash
source .venv/bin/activate
python -c "from src.watcher import start_watcher; start_watcher()"
```

Leave this running in a terminal while you work. Every time you save a file to `data/`, it re-indexes automatically within seconds.

---

## Common issues

**`(.venv)` not showing in terminal**

The virtual environment isn't active. Run:
```bash
source .venv/bin/activate
```

**`ModuleNotFoundError: No module named 'src'`**

You're running the script from the wrong directory. Always run commands from the project root (where `run_indexer.py` lives):
```bash
cd /path/to/file-search-mcp
python run_indexer.py
```

**`GOOGLE_API_KEY not found`**

Your `.env` file is missing or in the wrong place. It must be in the project root (same folder as `run_indexer.py`). Double-check:
```bash
ls -la | grep .env
cat .env
```

**Gemini rate limit error in summarise**

The free tier allows ~15 requests per minute. Wait a moment and try again. The `search_files` and `read_file` tools don't call Gemini, so they always work instantly.

---

## Requirements

- Python 3.11+
- macOS, Linux, or Windows (WSL)
- ~500MB disk space (ChromaDB index + embedding model cache)
- Internet connection for first setup and Gemini summarise calls

---

## License

MIT — do whatever you want with this.

---
