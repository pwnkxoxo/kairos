"""
KAIROS Local Database Storage - SQLite Persistence
Handles persistent local storage for conversations, messages, and document references.
Stored safely in an OS-appropriate data directory (or KAIROS_DATA_DIR / ECHO_DATA_DIR / workspace/data).

Guarantees:
  - Thread-safe SQLite transactions and parameterized queries.
  - Automatic table creation and versioned migration support.
  - No secrets, credentials, or full raw sensitive documents stored in conversation logs.
  - Survives backend restarts.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import uuid

# ------------------------------------------------------------------------------
# 1. Database Location & Directory Configuration
# ------------------------------------------------------------------------------
def get_data_dir() -> Path:
    """
    Returns an OS-appropriate application data directory for KAIROS.
    Precedence:
      1. KAIROS_DATA_DIR or ECHO_DATA_DIR environment variable (useful for testing or custom setups).
      2. Linux / BSD: $XDG_DATA_HOME/kairos or ~/.local/share/kairos (or legacy echo dir if present)
      3. macOS: ~/Library/Application Support/kairos (or legacy echo dir if present)
      4. Windows: %APPDATA%\\kairos or %LOCALAPPDATA%\\kairos
      5. Fallback: <project_root>/workspace/data
    """
    override = os.getenv("KAIROS_DATA_DIR") or os.getenv("ECHO_DATA_DIR")
    if override:
        path = Path(override).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    home = Path.home()
    if os.name == "nt":
        base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        path = Path(base) / "kairos" if base else home / "AppData" / "Roaming" / "kairos"
    elif os.uname().sysname == "Darwin":
        path = home / "Library" / "Application Support" / "kairos"
    else:
        xdg_data = os.getenv("XDG_DATA_HOME")
        path = (Path(xdg_data) / "kairos") if xdg_data else (home / ".local" / "share" / "kairos")

    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        # Safe fallback inside workspace/data
        fallback = Path(__file__).resolve().parent.parent / "workspace" / "data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def get_db_path() -> Path:
    """
    Returns the full path to the SQLite database file.
    Deterministic database selection rules:
      1. If legacy 'echo_history.db' exists and 'kairos_history.db' does not exist,
         use 'echo_history.db' for backward compatibility with existing legacy installations.
      2. If 'kairos_history.db' exists, use 'kairos_history.db' (takes precedence if both exist;
         neither database is merged, deleted, renamed, or overwritten).
      3. On a fresh installation (neither database exists), default to 'kairos_history.db'.
    """
    data_dir = get_data_dir()
    kairos_db = data_dir / "kairos_history.db"
    echo_db = data_dir / "echo_history.db"
    if echo_db.exists() and not kairos_db.exists():
        return echo_db
    return kairos_db


from contextlib import contextmanager

# ------------------------------------------------------------------------------
# 2. Connection Helper & Schema Migration
# ------------------------------------------------------------------------------
@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """
    Context manager yielding a SQLite connection with row_factory set to sqlite3.Row,
    foreign keys enabled, and automatic closing upon exit.
    """
    path = db_path or get_db_path()
    conn = sqlite3.connect(str(path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initializes database tables and indexes safely without dropping or destroying existing data.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,          -- 'user', 'assistant', 'system'
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,      -- JSON string for workflow, plan, action_id, pdf citations, etc.
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 1,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                sha256_prefix TEXT NOT NULL,
                ocr_available INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_documents (
                conversation_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                attached_at TEXT NOT NULL,
                PRIMARY KEY (conversation_id, doc_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,          -- 'pending_approval', 'executing', 'completed', 'rejected', 'failed', 'expired'
                mode TEXT NOT NULL,            -- 'gemma', 'demo'
                model_name TEXT,
                user_message TEXT NOT NULL,
                plan_json TEXT NOT NULL,       -- JSON serialized steps list
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            """
        )

        # Performance indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_docs_conv ON conversation_documents(conversation_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_docs_doc ON conversation_documents(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_conv ON actions(conversation_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status, expires_at);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overwrite_confirmations (
                token TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                target_file TEXT NOT NULL,
                file_content TEXT NOT NULL,
                action_desc TEXT NOT NULL,
                mode TEXT NOT NULL,
                model_name TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                consumed_at TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_overwrite_confirmations_token ON overwrite_confirmations(token);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_overwrite_confirmations_conv ON overwrite_confirmations(conversation_id);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                conversation_id TEXT,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_conv ON rag_chunks(conversation_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON rag_chunks(source_type);")

        cur = conn.execute("SELECT MAX(version) AS cur_ver FROM schema_version;")
        row = cur.fetchone()
        cur_version = row["cur_ver"] if row and row["cur_ver"] is not None else 0

        now = datetime.now(timezone.utc).isoformat()
        if cur_version < 1:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (1, ?);",
                (now,),
            )
        if cur_version < 2:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (2, ?);",
                (now,),
            )
        if cur_version < 3:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (3, ?);",
                (now,),
            )
        conn.commit()


# ------------------------------------------------------------------------------
# 3. Conversation Operations (CRUD)
# ------------------------------------------------------------------------------
def create_conversation(title: str = "New Chat", conv_id: Optional[str] = None, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Creates a new conversation record."""
    cid = conv_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    clean_title = (title or "New Chat").strip()[:150] or "New Chat"

    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
            (cid, clean_title, now, now),
        )
        conn.commit()

    return {
        "id": cid,
        "title": clean_title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
        "attached_doc_ids": [],
    }


def list_conversations(query: Optional[str] = None, limit: int = 100, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Lists conversations ordered by recency, optionally filtered by search query."""
    with get_connection(db_path) as conn:
        if query and query.strip():
            term = f"%{query.strip().lower()}%"
            # Matches title or message text
            sql = """
                SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE LOWER(c.title) LIKE ? OR LOWER(m.content) LIKE ?
                ORDER BY c.updated_at DESC
                LIMIT ?;
            """
            cur = conn.execute(sql, (term, term, limit))
        else:
            sql = """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT ?;
            """
            cur = conn.execute(sql, (limit,))

        rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "message_count": r["msg_count"],
        })
    return results


def get_conversation(conv_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves conversation details along with all messages and attached documents."""
    with get_connection(db_path) as conn:
        c_cur = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?;",
            (conv_id,),
        )
        c_row = c_cur.fetchone()
        if not c_row:
            return None

        m_cur = conn.execute(
            "SELECT id, role, content, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC;",
            (conv_id,),
        )
        messages = []
        for mr in m_cur.fetchall():
            meta = {}
            if mr["metadata"]:
                try:
                    meta = json.loads(mr["metadata"])
                except Exception:
                    meta = {}
            messages.append({
                "id": mr["id"],
                "role": mr["role"],
                "content": mr["content"],
                "metadata": meta,
                "created_at": mr["created_at"],
            })

        d_cur = conn.execute(
            """
            SELECT d.doc_id, d.filename, d.storage_path, d.page_count, d.size_bytes, d.ocr_available
            FROM documents d
            JOIN conversation_documents cd ON cd.doc_id = d.doc_id
            WHERE cd.conversation_id = ?
            ORDER BY cd.attached_at ASC;
            """,
            (conv_id,),
        )
        docs = []
        for dr in d_cur.fetchall():
            docs.append({
                "doc_id": dr["doc_id"],
                "filename": dr["filename"],
                "storage_path": dr["storage_path"],
                "page_count": dr["page_count"],
                "size_bytes": dr["size_bytes"],
                "ocr_available": bool(dr["ocr_available"]),
            })

    return {
        "id": c_row["id"],
        "title": c_row["title"],
        "created_at": c_row["created_at"],
        "updated_at": c_row["updated_at"],
        "messages": messages,
        "documents": docs,
    }


def rename_conversation(conv_id: str, new_title: str, db_path: Optional[Path] = None) -> bool:
    """Renames a conversation title."""
    clean_title = (new_title or "").strip()[:150]
    if not clean_title:
        return False
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?;",
            (clean_title, now, conv_id),
        )
        conn.commit()
        return cur.rowcount > 0


def delete_conversation(conv_id: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Deletes a conversation and its messages.
    Deletes referenced documents ONLY if they are not referenced by any other conversation.
    """
    deleted_files: List[str] = []
    with get_connection(db_path) as conn:
        # Find candidate documents attached to this conversation
        cur_docs = conn.execute(
            "SELECT doc_id FROM conversation_documents WHERE conversation_id = ?;",
            (conv_id,),
        )
        candidate_doc_ids = [row["doc_id"] for row in cur_docs.fetchall()]

        # Delete conversation record (CASCADE removes messages and conversation_documents rows)
        del_cur = conn.execute("DELETE FROM conversations WHERE id = ?;", (conv_id,))
        conv_deleted = del_cur.rowcount > 0

        # Delete conversation-scoped RAG chunks
        conn.execute("DELETE FROM rag_chunks WHERE conversation_id = ?;", (conv_id,))

        # Check which candidate documents are now orphaned (not referenced anywhere)
        for did in candidate_doc_ids:
            ref_check = conn.execute(
                "SELECT COUNT(*) AS cnt FROM conversation_documents WHERE doc_id = ?;",
                (did,),
            )
            count = ref_check.fetchone()["cnt"]
            if count == 0:
                # Retrieve storage path before deletion
                dinfo = conn.execute("SELECT storage_path FROM documents WHERE doc_id = ?;", (did,)).fetchone()
                if dinfo and dinfo["storage_path"]:
                    fpath = Path(dinfo["storage_path"])
                    if fpath.exists():
                        try:
                            fpath.unlink()
                            deleted_files.append(str(fpath))
                        except Exception:
                            pass
                conn.execute("DELETE FROM documents WHERE doc_id = ?;", (did,))
                conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?;", (did,))

        conn.commit()

    return {
        "conversation_id": conv_id,
        "deleted": conv_deleted,
        "deleted_files_count": len(deleted_files),
    }


# ------------------------------------------------------------------------------
# 4. Message Operations
# ------------------------------------------------------------------------------
def add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    msg_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Adds a message to an existing conversation and updates conversation's updated_at timestamp."""
    mid = msg_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata or {})

    with get_connection(db_path) as conn:
        # Ensure conversation exists; if not, create it with auto title
        c_check = conn.execute("SELECT id, title, (SELECT COUNT(*) FROM messages WHERE conversation_id = ?) AS mcnt FROM conversations WHERE id = ?;", (conversation_id, conversation_id)).fetchone()
        if not c_check:
            # Auto-title from first 40 chars of user message if role == 'user'
            title = (content[:45] + "...") if len(content) > 45 else content
            title = title.replace("\n", " ").strip() or "New Chat"
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (conversation_id, title, now, now),
            )
        elif role == "user" and c_check["mcnt"] == 0 and c_check["title"] in ["New Chat", "Untitled"]:
            # Update title from first user prompt
            new_title = (content[:45] + "...") if len(content) > 45 else content
            new_title = new_title.replace("\n", " ").strip() or "New Chat"
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?;", (new_title, conversation_id))

        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (mid, conversation_id, role, content, meta_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?;",
            (now, conversation_id),
        )
        conn.commit()

    return {
        "id": mid,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": now,
    }


def get_conversation_context_slice(
    conversation_id: str,
    max_messages: int = 6,
    db_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Returns a bounded, recent slice of conversation messages for model context follow-up.
    Formats as [{'role': 'user'|'assistant', 'content': '...'}].
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (conversation_id, max_messages),
        )
        rows = cur.fetchall()

    # Reverse so they are in chronological order
    slice_msgs = []
    for r in reversed(rows):
        if r["role"] in ["user", "assistant"]:
            slice_msgs.append({"role": r["role"], "content": r["content"]})
    return slice_msgs


# ------------------------------------------------------------------------------
# 5. Document Persistence & Conversation Linkage
# ------------------------------------------------------------------------------
def store_document_record(
    doc_id: str,
    filename: str,
    storage_path: str,
    page_count: int,
    size_bytes: int,
    sha256_prefix: str,
    ocr_available: bool = True,
    db_path: Optional[Path] = None,
) -> None:
    """Saves or updates document metadata in the documents table."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (doc_id, filename, storage_path, page_count, size_bytes, sha256_prefix, ocr_available, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                filename=excluded.filename,
                storage_path=excluded.storage_path,
                page_count=excluded.page_count,
                size_bytes=excluded.size_bytes,
                ocr_available=excluded.ocr_available;
            """,
            (doc_id, filename, storage_path, page_count, size_bytes, sha256_prefix, int(ocr_available), now),
        )
        conn.commit()


def link_document_to_conversation(
    conversation_id: str,
    doc_id: str,
    db_path: Optional[Path] = None,
) -> None:
    """Links an uploaded document to a specific conversation."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        c_check = conn.execute("SELECT id FROM conversations WHERE id = ?;", (conversation_id,)).fetchone()
        if not c_check:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (conversation_id, "New Chat", now, now),
            )
        conn.execute(
            """
            INSERT OR IGNORE INTO conversation_documents (conversation_id, doc_id, attached_at)
            VALUES (?, ?, ?);
            """,
            (conversation_id, doc_id, now),
        )
        conn.commit()


def get_conversation_documents(conversation_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns all documents attached to a conversation."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT d.doc_id, d.filename, d.storage_path, d.page_count, d.size_bytes, d.ocr_available, d.sha256_prefix
            FROM documents d
            JOIN conversation_documents cd ON cd.doc_id = d.doc_id
            WHERE cd.conversation_id = ?
            ORDER BY cd.attached_at ASC;
            """,
            (conversation_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "storage_path": r["storage_path"],
            "page_count": r["page_count"],
            "size_bytes": r["size_bytes"],
            "ocr_available": bool(r["ocr_available"]),
            "sha256_prefix": r["sha256_prefix"],
        }
        for r in rows
    ]


def get_document_record(doc_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Returns stored metadata for a single document ID from the database, or None if not found."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT doc_id, filename, storage_path, page_count, size_bytes, ocr_available, sha256_prefix, created_at
            FROM documents
            WHERE doc_id = ?;
            """,
            (doc_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "storage_path": r["storage_path"],
            "page_count": r["page_count"],
            "size_bytes": r["size_bytes"],
            "ocr_available": bool(r["ocr_available"]),
            "sha256_prefix": r["sha256_prefix"],
            "created_at": r["created_at"],
        }


# ------------------------------------------------------------------------------
# 6. Action Persistence & Lifecycle Management
# ------------------------------------------------------------------------------
DEFAULT_ACTION_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def store_action(
    action_id: str,
    conversation_id: str,
    user_message: str,
    steps: List[Dict[str, Any]],
    mode: str = "demo",
    model_name: Optional[str] = None,
    ttl_seconds: int = DEFAULT_ACTION_TTL_SECONDS,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Persists a newly planned action in 'pending_approval' status with an explicit expiration timestamp.
    """
    now = datetime.now(timezone.utc)
    expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc)
    created_str = now.isoformat()
    expires_str = expires.isoformat()
    plan_json = json.dumps(steps)

    with get_connection(db_path) as conn:
        # Ensure conversation exists to satisfy FOREIGN KEY constraint
        c_check = conn.execute("SELECT id FROM conversations WHERE id = ?;", (conversation_id,)).fetchone()
        if not c_check:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (conversation_id, "New Chat", created_str, created_str),
            )

        conn.execute(
            """
            INSERT INTO actions (
                id, conversation_id, status, mode, model_name, user_message, plan_json, created_at, expires_at
            ) VALUES (?, ?, 'pending_approval', ?, ?, ?, ?, ?, ?);
            """,
            (action_id, conversation_id, mode, model_name, user_message, plan_json, created_str, expires_str),
        )
        conn.commit()

    return {
        "action_id": action_id,
        "conversation_id": conversation_id,
        "status": "pending_approval",
        "mode": mode,
        "model_name": model_name,
        "user_message": user_message,
        "created_at": created_str,
        "expires_at": expires_str,
        "steps": steps,
    }


def get_action(action_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves an action record by ID.
    If the action is pending_approval but past its expires_at time, updates status to 'expired'.
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT id, conversation_id, status, mode, model_name, user_message, plan_json,
                   created_at, expires_at, completed_at, error_message
            FROM actions
            WHERE id = ?;
            """,
            (action_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        status = row["status"]
        expires_at = row["expires_at"]
        now_str = datetime.now(timezone.utc).isoformat()

        # Check and lazily transition to expired if currently pending or stale executing past expiration
        if status in ("pending_approval", "executing") and expires_at and now_str > expires_at:
            conn.execute("UPDATE actions SET status = 'expired' WHERE id = ?;", (action_id,))
            conn.commit()
            status = "expired"

        steps = []
        if row["plan_json"]:
            try:
                steps = json.loads(row["plan_json"])
            except Exception:
                steps = []

        return {
            "action_id": row["id"],
            "conversation_id": row["conversation_id"],
            "status": status,
            "mode": row["mode"],
            "model_name": row["model_name"],
            "user_message": row["user_message"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
            "steps": steps,
        }


def claim_action_for_execution(action_id: str, db_path: Optional[Path] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Atomically transitions an action from 'pending_approval' (or retryable 'failed') to 'executing'
    within a short SQLite transaction to prevent duplicate or concurrent approvals.
    Returns (success, action_dict, reason_message).
    """
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT id, conversation_id, status, mode, model_name, user_message, plan_json,
                   created_at, expires_at, completed_at, error_message
            FROM actions
            WHERE id = ?;
            """,
            (action_id,),
        )
        row = cur.fetchone()
        if not row:
            return False, None, "not_found"

        status = row["status"]
        expires_at = row["expires_at"]

        # Check expiration
        if expires_at and now_str > expires_at:
            conn.execute("UPDATE actions SET status = 'expired' WHERE id = ?;", (action_id,))
            conn.commit()
            return False, None, "expired"

        # Check eligibility: allowed from 'pending_approval' or 'failed'
        if status not in ("pending_approval", "failed"):
            return False, None, status

        # Atomically claim
        claim_cur = conn.execute(
            """
            UPDATE actions
            SET status = 'executing', error_message = NULL
            WHERE id = ? AND status IN ('pending_approval', 'failed');
            """,
            (action_id,),
        )
        conn.commit()

        if claim_cur.rowcount == 0:
            # Raced with another concurrent claim
            return False, None, "conflict"

        steps = []
        if row["plan_json"]:
            try:
                steps = json.loads(row["plan_json"])
            except Exception:
                steps = []

        claimed_action = {
            "action_id": row["id"],
            "conversation_id": row["conversation_id"],
            "status": "executing",
            "mode": row["mode"],
            "model_name": row["model_name"],
            "user_message": row["user_message"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "steps": steps,
        }
        return True, claimed_action, "claimed"


def complete_action(action_id: str, db_path: Optional[Path] = None) -> bool:
    """Marks an executing action as completed."""
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE actions
            SET status = 'completed', completed_at = ?, error_message = NULL
            WHERE id = ?;
            """,
            (now_str, action_id),
        )
        conn.commit()
        return cur.rowcount > 0


def fail_action(action_id: str, error_message: str, db_path: Optional[Path] = None) -> bool:
    """Reverts an executing action to 'failed' status so it can be investigated or safely retried."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE actions
            SET status = 'failed', error_message = ?
            WHERE id = ?;
            """,
            (error_message[:500], action_id),
        )
        conn.commit()
        return cur.rowcount > 0


def reject_action_record(action_id: str, db_path: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Transitions a pending action to 'rejected' status.
    Returns (success, current_status_or_reason).
    """
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT status, expires_at FROM actions WHERE id = ?;",
            (action_id,),
        )
        row = cur.fetchone()
        if not row:
            return False, "not_found"

        status = row["status"]
        expires_at = row["expires_at"]

        if expires_at and now_str > expires_at:
            conn.execute("UPDATE actions SET status = 'expired' WHERE id = ?;", (action_id,))
            conn.commit()
            return False, "expired"

        if status != "pending_approval":
            return False, status

        upd = conn.execute(
            "UPDATE actions SET status = 'rejected' WHERE id = ? AND status = 'pending_approval';",
            (action_id,),
        )
        conn.commit()
        return (upd.rowcount > 0), "rejected" if upd.rowcount > 0 else "conflict"


def expire_stale_actions(db_path: Optional[Path] = None) -> int:
    """Marks all expired actions as 'expired'. Returns count of updated actions."""
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE actions
            SET status = 'expired'
            WHERE status IN ('pending_approval', 'executing', 'failed') AND expires_at < ?;
            """,
            (now_str,),
        )
        conn.commit()
        return cur.rowcount


def recover_interrupted_actions(db_path: Optional[Path] = None) -> int:
    """
    Recovers actions that were interrupted in 'executing' status (e.g. process crash or restart).
    Transitions stranded 'executing' actions to 'failed' with an explicit recovery error message.
    Crucially: NEVER automatically re-executes actions. Does not revive expired, completed, or rejected actions.
    Returns the count of recovered actions.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    msg = "Execution interrupted by process termination or restart. Manual approval retry required."
    with get_connection(db_path) as conn:
        # First expire any executing actions that have exceeded their 24h deadline
        conn.execute(
            """
            UPDATE actions
            SET status = 'expired'
            WHERE status = 'executing' AND expires_at < ?;
            """,
            (now_str,),
        )
        # Transition remaining unexpired 'executing' actions to 'failed'
        cur = conn.execute(
            """
            UPDATE actions
            SET status = 'failed', error_message = ?
            WHERE status = 'executing';
            """,
            (msg,),
        )
        conn.commit()
        return cur.rowcount


# ------------------------------------------------------------------------------
# 7. Overwrite Confirmation Lifecycle Management
# ------------------------------------------------------------------------------
DEFAULT_CONFIRMATION_TTL_SECONDS = 10 * 60  # 10 minutes


def create_overwrite_confirmation(
    token: str,
    conversation_id: str,
    target_file: str,
    file_content: str,
    action_desc: str,
    mode: str = "demo",
    model_name: Optional[str] = None,
    ttl_seconds: int = DEFAULT_CONFIRMATION_TTL_SECONDS,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Persists a pending overwrite confirmation bound strictly to a target file and content.
    Expires after ttl_seconds (default 10 minutes).
    """
    now = datetime.now(timezone.utc)
    expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc)
    created_str = now.isoformat()
    expires_str = expires.isoformat()

    with get_connection(db_path) as conn:
        c_check = conn.execute("SELECT id FROM conversations WHERE id = ?;", (conversation_id,)).fetchone()
        if not c_check:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (conversation_id, "New Chat", created_str, created_str),
            )
        conn.execute(
            """
            INSERT INTO overwrite_confirmations (
                token, conversation_id, target_file, file_content, action_desc,
                mode, model_name, created_at, expires_at, consumed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL);
            """,
            (token, conversation_id, target_file, file_content, action_desc, mode, model_name, created_str, expires_str),
        )
        conn.commit()

    return {
        "token": token,
        "conversation_id": conversation_id,
        "target_file": target_file,
        "created_at": created_str,
        "expires_at": expires_str,
    }


def get_and_consume_overwrite_confirmation(
    token: str,
    target_file: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Validates and atomically consumes a pending overwrite confirmation.
    Checks existence, expiration, and ensures path binding matches target_file if provided.
    Cannot be consumed more than once.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT token, conversation_id, target_file, file_content, action_desc,
                   mode, model_name, created_at, expires_at, consumed_at
            FROM overwrite_confirmations
            WHERE token = ?;
            """,
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return False, None, "not_found"

        if row["consumed_at"] is not None:
            return False, None, "already_consumed"

        if row["expires_at"] < now:
            return False, None, "expired"

        if target_file and row["target_file"] != target_file:
            return False, None, "path_mismatch"

        # Atomically mark as consumed
        update_cur = conn.execute(
            """
            UPDATE overwrite_confirmations
            SET consumed_at = ?
            WHERE token = ? AND consumed_at IS NULL;
            """,
            (now, token),
        )
        conn.commit()
        if update_cur.rowcount == 0:
            return False, None, "already_consumed"

        record = {
            "token": row["token"],
            "conversation_id": row["conversation_id"],
            "target_file": row["target_file"],
            "file_content": row["file_content"],
            "action_desc": row["action_desc"],
            "mode": row["mode"],
            "model_name": row["model_name"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "consumed_at": now,
        }
        return True, record, "ok"


# ------------------------------------------------------------------------------
# 8. RAG Knowledge & Document Chunk Operations
# ------------------------------------------------------------------------------
def store_rag_chunks(chunks: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """
    Persists document/knowledge chunks in the rag_chunks table.
    Each chunk dict must have:
      chunk_id, doc_id, conversation_id (optional), source_type, source_name,
      page_number (optional), chunk_index, content, metadata_json (optional).
    """
    if not chunks:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for c in chunks:
        cid = c.get("chunk_id") or str(uuid.uuid4())
        meta = c.get("metadata_json")
        if isinstance(meta, dict):
            meta_str = json.dumps(meta)
        elif isinstance(meta, str):
            meta_str = meta
        else:
            meta_str = "{}"

        rows.append((
            cid,
            c["doc_id"],
            c.get("conversation_id"),
            c.get("source_type", "pdf"),
            c.get("source_name", "document"),
            c.get("page_number"),
            c.get("chunk_index", 0),
            c["content"],
            meta_str,
            now,
        ))

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO rag_chunks (
                chunk_id, doc_id, conversation_id, source_type, source_name,
                page_number, chunk_index, content, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def delete_rag_chunks_for_doc(doc_id: str, db_path: Optional[Path] = None) -> int:
    """Deletes all chunks associated with a specific doc_id."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?;", (doc_id,))
        conn.commit()
        return cur.rowcount


def delete_rag_chunks_for_conversation(conv_id: str, db_path: Optional[Path] = None) -> int:
    """Deletes all chunks attached to a specific conversation."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM rag_chunks WHERE conversation_id = ?;", (conv_id,))
        conn.commit()
        return cur.rowcount


def query_rag_chunks(
    conversation_id: Optional[str] = None,
    source_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves accessible RAG chunks with strict access-control boundaries:
      - If conversation_id is provided: returns chunks belonging to that conversation
        PLUS global reference chunks (conversation_id IS NULL).
      - If conversation_id is None: returns ONLY global reference chunks.
      - Never leaks chunks belonging to a different conversation.
    """
    with get_connection(db_path) as conn:
        conditions = []
        params: List[Any] = []

        if conversation_id:
            conditions.append("(conversation_id = ? OR conversation_id IS NULL)")
            params.append(conversation_id)
        else:
            conditions.append("conversation_id IS NULL")

        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT chunk_id, doc_id, conversation_id, source_type, source_name,
                   page_number, chunk_index, content, metadata_json, created_at
            FROM rag_chunks
            {where_clause}
            ORDER BY source_name ASC, chunk_index ASC;
        """
        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()

    return [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "conversation_id": r["conversation_id"],
            "source_type": r["source_type"],
            "source_name": r["source_name"],
            "page_number": r["page_number"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "metadata_json": r["metadata_json"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# Auto-init on module import
init_db()

