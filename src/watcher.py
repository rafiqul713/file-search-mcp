"""
=============================================================================
watcher.py — Automatic File Re-indexing via Watchdog
=============================================================================

WHAT THIS FILE DOES:
--------------------
Watches the data/ directory for file changes using the watchdog library.
Whenever a file is created or modified, it automatically re-indexes that
file so the search index stays in sync without manual intervention.

WHY THIS IS USEFUL:
-------------------
Without a watcher, you would have to manually run "python run_indexer.py"
every time you edit a file. With the watcher running in the background,
your index is always up to date — save a file, search it seconds later.

HOW WATCHDOG WORKS:
-------------------
The watchdog library uses OS-level file system events (inotify on Linux,
FSEvents on macOS, ReadDirectoryChangesW on Windows) to get notified of
changes without polling. It's efficient — it doesn't repeatedly scan the
directory, it just listens for OS notifications.

Key components:
  Observer     — the background thread that watches the filesystem
  EventHandler — our custom class that decides what to do when an event fires
=============================================================================
"""

import os
import time                        # used for the blocking loop (time.sleep)
from pathlib import Path           # cleaner file path handling
from dotenv import load_dotenv

# watchdog.observers.Observer — the main watcher object.
# Runs in a background thread, fires events when files change.
from watchdog.observers import Observer

# FileSystemEventHandler — base class we inherit from.
# We override on_created and on_modified to react to file changes.
from watchdog.events import FileSystemEventHandler

# Our indexer function that handles a single file.
# Called every time watchdog detects a relevant change.
from src.indexer import index_single_file

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
FILES_DIR = os.getenv("FILES_DIR", "./data")

# Only react to these file types — ignore .DS_Store, .log, temp files, etc.
SUPPORTED = {".txt", ".md", ".pdf"}


# ── Event handler class ────────────────────────────────────────────────────────

class FileChangeHandler(FileSystemEventHandler):
    """
    Custom watchdog event handler.

    Inherits from FileSystemEventHandler (watchdog's base class) and
    overrides two methods:
      - on_created  → called when a new file appears in the watched directory
      - on_modified → called when an existing file is saved/changed

    We don't override on_deleted because deleting a file from disk doesn't
    require updating ChromaDB — the old chunks become stale but harmless.
    (You could add deletion support later as an enhancement.)
    """

    def on_created(self, event):
        """
        Called by watchdog when a new file is created in the watched directory.

        event.is_directory — True if a folder was created (we ignore those)
        event.src_path     — full path to the new file

        We check the file extension and only index supported types.
        """
        # Skip directory creation events — we only care about files
        if not event.is_directory:
            path = Path(event.src_path)

            # Only process supported file types to avoid wasting time
            # on system files like .DS_Store (macOS) or .swp (vim temp files)
            if path.suffix.lower() in SUPPORTED:
                print(f"[Watcher] New file detected: {path.name}")
                # Re-index this specific file — much faster than re-indexing everything
                index_single_file(str(path))

    def on_modified(self, event):
        """
        Called by watchdog when an existing file is saved or modified.

        Works exactly like on_created — same logic, same result.
        index_single_file() handles both new files and updated files:
        it deletes old chunks before adding new ones.
        """
        if not event.is_directory:
            path = Path(event.src_path)
            if path.suffix.lower() in SUPPORTED:
                print(f"[Watcher] File modified: {path.name}")
                index_single_file(str(path))


# ── Main watcher function ──────────────────────────────────────────────────────

def start_watcher(files_dir: str = FILES_DIR):
    """
    Start watching the files directory for changes.

    This function BLOCKS — it runs in an infinite loop until you press Ctrl+C.
    In practice, you'd run this in a separate terminal while the MCP server
    runs in another, or wrap it in a background thread.

    How it works:
      1. Create our FileChangeHandler (tells watchdog what to do on events)
      2. Create an Observer (the background watcher thread)
      3. Schedule: "watch files_dir with this handler, recursively"
      4. Start the observer thread
      5. Sleep in a loop until Ctrl+C (KeyboardInterrupt)
      6. On Ctrl+C: stop the observer cleanly and join the thread

    Args:
        files_dir: the directory to watch (default: ./data from .env)
    """
    # Make sure the directory exists before trying to watch it
    Path(files_dir).mkdir(parents=True, exist_ok=True)

    # Instantiate our custom event handler
    handler = FileChangeHandler()

    # Observer manages the background watcher thread
    observer = Observer()

    # schedule() tells the observer:
    #   - which handler to use for events in this directory
    #   - which directory to watch
    #   - recursive=True means it watches subdirectories too
    observer.schedule(handler, files_dir, recursive=True)

    # Start the background thread — the OS will now deliver file events to it
    observer.start()
    print(f"[Watcher] Watching {files_dir} for changes. Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive so the background observer can keep running.
        # We sleep 2 seconds at a time to avoid busy-waiting (wasting CPU).
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        # User pressed Ctrl+C — gracefully shut down the observer thread
        print("\n[Watcher] Stopping...")
        observer.stop()

    # join() waits for the observer thread to fully finish before we exit.
    # This prevents "thread still running" errors at shutdown.
    observer.join()
    print("[Watcher] Stopped.")