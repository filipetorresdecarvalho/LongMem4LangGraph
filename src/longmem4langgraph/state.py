"""StateManager — Combined state, history, and audit in one class.

Implements the PipelineStore ABC interface. Wraps SqliteSaver
(checkpoints), HistoryStore (agent memory), and recovery (crash recovery)
with pipeline-level operations.

All components share a single SqliteConnection to avoid write
contention and enable VACUUM.

Use this when you want a single entry point for all persistence needs.
Use the individual components (SqliteSaver, HistoryStore) when you
want only specific functionality.

Usage:
    from longmem4langgraph import StateManager

    with StateManager("pipeline.db") as sm:
        sm.save_state("req-1", {"status": "processing"})
        summary = sm.get_summary("req-1")
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from .connection import SqliteConnection
from .saver import SqliteSaver
from .history import HistoryStore
from . import recovery
from .migration import ensure_migrated

logger = logging.getLogger(__name__)

_SQLITE_OK = 0
_SQLITE_DENY = 8
_SQLITE_READ = 21
_SQLITE_PRAGMA = 19


def _read_only_authorizer(action_code, arg1, arg2, db_name, trigger_name):
    """SQLite authorizer that blocks write operations.

    Allows all read and pragma operations. Denies INSERT, UPDATE,
    DELETE, DROP, ALTER, CREATE, ATTACH, and DETACH.
    """
    if action_code == _SQLITE_READ:
        return _SQLITE_OK
    if action_code == _SQLITE_PRAGMA:
        return _SQLITE_OK
    return _SQLITE_DENY


_WRITE_KEYWORDS = frozenset([
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "REINDEX", "ATTACH", "DETACH", "VACUUM",
])


def _is_read_only_query(sql: str) -> bool:
    """Check if a SQL query is read-only by examining the first keyword."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word not in _WRITE_KEYWORDS


class StateManager:
    """SQLite-backed state manager for persistent LangGraph pipelines.

    Provides everything needed for persistent LangGraph pipelines:
    - Checkpoint persistence (via SqliteSaver)
    - AGNO-style history (via HistoryStore)
    - Crash recovery (via recovery module)
    - Pipeline summaries, audit, and read-only queries

    Implements the PipelineStore ABC interface. All sub-components
    share a single SqliteConnection.

    Args:
        db_path: Path to SQLite file. Use same path for all components.
        search_backend: Optional SearchBackend for semantic search.
            Passed through to HistoryStore.
    """

    def __init__(
        self,
        db_path: str = "pipeline.db",
        search_backend=None,
    ):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self._conn = SqliteConnection(self.db_path)

        self.checkpointer = SqliteSaver(
            self.db_path, connection=self._conn,
        )
        self.history = HistoryStore(
            connection=self._conn,
            search_backend=search_backend,
        )

        self._lock = threading.Lock()
        ensure_migrated(self._conn)

    # =====================================================================
    # Pipeline State
    # =====================================================================

    def save_state(self, request_id: str, state: dict) -> dict:
        """Save/update the current pipeline state. Returns merged state."""
        existing = self._conn.execute(
            "SELECT data FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()

        if existing:
            current = json.loads(existing["data"])
            current.update(state)
        else:
            current = state

        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO pipeline_states
                   (request_id, status, current_node, data, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (
                    request_id,
                    state.get("status", "unknown"),
                    state.get("current_node"),
                    json.dumps(current),
                    state.get("error"),
                ),
            )
            self._conn.commit()

        return current

    def get_state(self, request_id: str) -> Optional[dict]:
        """Get current pipeline state, or None if not found."""
        row = self._conn.execute(
            "SELECT data FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def save_pipeline_state(self, request_id: str, state: dict) -> dict:
        """Alias for save_state() — preserved for backward compatibility."""
        return self.save_state(request_id, state)

    def get_pipeline_state(self, request_id: str) -> Optional[dict]:
        """Alias for get_state() — preserved for backward compatibility."""
        return self.get_state(request_id)

    # =====================================================================
    # Node Tracking
    # =====================================================================

    def start_node(self, request_id: str, node_name: str) -> int:
        """Record the start of a node execution. Returns the node record id."""
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO pipeline_nodes
                   (request_id, node_name, status, started_at)
                   VALUES (?, ?, 'running', datetime('now'))""",
                (request_id, node_name),
            )
            self._conn.commit()
            return cursor.lastrowid

    def finish_node(
        self,
        node_id: int,
        status: str,
        output: Optional[dict] = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Record the completion of a node execution."""
        with self._lock:
            self._conn.execute(
                """UPDATE pipeline_nodes SET
                   status = ?,
                   finished_at = datetime('now'),
                   output_snapshot = ?,
                   error = ?,
                   duration_ms = ?
                   WHERE id = ?""",
                (
                    status,
                    json.dumps(output) if output else None,
                    error,
                    duration_ms,
                    node_id,
                ),
            )
            self._conn.commit()

    def track_node(self, request_id: str, node_name: str):
        """Context manager that auto-tracks node execution.

        Usage:
            with sm.track_node("req-1", "parse_files"):
                do_work()
        """
        return _NodeTracker(self, request_id, node_name)

    # =====================================================================
    # Pipeline Summary
    # =====================================================================

    def get_summary(self, request_id: str) -> dict:
        """Get a complete summary of a pipeline run."""
        pipeline = self._conn.execute(
            "SELECT * FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()

        nodes = [
            dict(r) for r in self._conn.execute(
                "SELECT * FROM pipeline_nodes WHERE request_id = ? ORDER BY started_at",
                (request_id,),
            ).fetchall()
        ]

        history = self.history.get_history(request_id)
        cost = self.history.get_cost_summary(request_id)

        return {
            "pipeline": dict(pipeline) if pipeline else None,
            "nodes": nodes,
            "history_turns": len(history),
            "total_cost_cents": cost.get("total_cost_cents", 0),
            "total_tokens": cost.get("total_tokens", 0),
        }

    def get_pipeline_summary(self, request_id: str) -> dict:
        """Alias for get_summary() — preserved for backward compatibility."""
        return self.get_summary(request_id)

    # =====================================================================
    # Recovery
    # =====================================================================

    def find_stalled(self, stalled_minutes: int = 5) -> List[dict]:
        """Find stalled pipelines that may need recovery."""
        return recovery.find_stalled_pipelines(
            stalled_minutes=stalled_minutes,
            connection=self._conn,
        )

    def mark_completed(self, request_id: str) -> None:
        """Mark pipeline as completed — updates BOTH pipeline_states AND checkpoints."""
        self.save_state(request_id, {"status": "completed"})
        thread_id = self._conn.execute(
            "SELECT thread_id FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if thread_id and thread_id["thread_id"]:
            recovery.mark_completed(
                thread_id=thread_id["thread_id"],
                connection=self._conn,
                db_path=self.db_path,
            )

    def mark_failed(self, request_id: str, error: str) -> None:
        """Mark pipeline as failed — updates BOTH tables."""
        self.save_state(request_id, {"status": "failed", "error": error})
        thread_id = self._conn.execute(
            "SELECT thread_id FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if thread_id and thread_id["thread_id"]:
            recovery.mark_failed(
                thread_id=thread_id["thread_id"],
                error=error,
                connection=self._conn,
                db_path=self.db_path,
            )

    # =====================================================================
    # Query (read-only, SQL injection protected)
    # =====================================================================

    def query(self, sql: str, params=()) -> List[dict]:
        """Run a read-only query against the state database.

        Blocks write operations by checking the SQL statement's first
        keyword. Raises ValueError for any write attempt.

        Useful for custom analytics, dashboards, or debugging.
        All tables are queryable: checkpoints, agent_history,
        pipeline_states, pipeline_nodes.

        Raises:
            ValueError: If the query attempts a write operation.
        """
        if not _is_read_only_query(sql):
            raise ValueError(
                f"Only read-only queries allowed. Query starts with: "
                f"{sql.strip().split()[0].upper() if sql.strip() else '(empty)'}"
            )
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # =====================================================================
    # Maintenance
    # =====================================================================

    def vacuum(self) -> None:
        """Reclaim disk space. Run periodically via cron."""
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def checkpoint_wal(self) -> None:
        """Flush WAL to main database file. Call before backup."""
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        """Graceful shutdown — close the shared connection."""
        self.checkpoint_wal()
        self._conn.close()

    def __enter__(self) -> "StateManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class _NodeTracker:
    """Context manager for automatic node tracking.

    Uses consistent time sources (SQLite datetime('now') for storage,
    Python datetime for duration calculation).
    """

    def __init__(self, sm: StateManager, request_id: str, node_name: str):
        self._sm = sm
        self._request_id = request_id
        self._node_name = node_name
        self._node_id = None

    def __enter__(self):
        self._node_id = self._sm.start_node(self._request_id, self._node_name)
        self._start_time = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = int(
            (datetime.now(timezone.utc) - self._start_time).total_seconds() * 1000
        )
        if exc_type is None:
            self._sm.finish_node(self._node_id, "success", duration_ms=duration)
        else:
            self._sm.finish_node(
                self._node_id, "failed", error=str(exc_val), duration_ms=duration
            )
