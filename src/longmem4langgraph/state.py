"""StateManager — Combined state, history, and audit in one Singleton.

This is the 'everything in one place' class from the RTGo V5 design.
It wraps SqliteSaver (checkpoints), HistoryStore (agent memory),
recovery (crash recovery), and adds pipeline-level operations.

Use this when you want a single entry point for all persistence needs.
Use the individual components (SqliteSaver, HistoryStore) when you
want only specific functionality.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict

from .connection import SqliteConnection
from .saver import SqliteSaver
from .history import HistoryStore
from . import recovery


class StateManager:
    """Singleton-ish SQLite-backed state manager.

    Provides everything needed for persistent LangGraph pipelines:
    - Checkpoint persistence (via SqliteSaver)
    - AGNO-style history (via HistoryStore)
    - Crash recovery (via recovery module)
    - Pipeline summaries, audit, and queries

    While not a true Singleton (you can create instances), sharing
    the same db_path means all components share the same SQLite file.

    Args:
        db_path: Path to SQLite file. Use same path for all components.
    """

    def __init__(self, db_path: str = "pipeline.db"):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self._conn = SqliteConnection(self.db_path)

        # Sub-components sharing the same database
        self.checkpointer = SqliteSaver(self.db_path)
        self.history = HistoryStore(self.db_path)

        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Extended tables for pipeline management."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pipeline_states (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT,
                status TEXT NOT NULL DEFAULT 'received',
                current_node TEXT,
                data TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS pipeline_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                thread_id TEXT,
                node_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT,
                finished_at TEXT,
                duration_ms INTEGER,
                input_snapshot TEXT,
                output_snapshot TEXT,
                error TEXT,
                retry_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_request
                ON pipeline_nodes(request_id);

            CREATE TABLE IF NOT EXISTS skills_generated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                thread_id TEXT,
                skill_name TEXT NOT NULL,
                skill_path TEXT,
                source TEXT,
                tags TEXT DEFAULT '[]',
                use_count INTEGER DEFAULT 0,
                quality_score REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    # =====================================================================
    # Pipeline State
    # =====================================================================

    def save_pipeline_state(self, request_id: str, state: dict) -> dict:
        """Save/update the current pipeline state."""
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

    def get_pipeline_state(self, request_id: str) -> Optional[dict]:
        """Get current pipeline state."""
        row = self._conn.execute(
            "SELECT data FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return json.loads(row["data"]) if row else None

    # =====================================================================
    # Node Tracking
    # =====================================================================

    def start_node(self, request_id: str, node_name: str) -> int:
        """Record the start of a node execution.

        Returns the node record id.
        """
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO pipeline_nodes
                   (request_id, node_name, status, started_at)
                   VALUES (?, ?, 'running', datetime('now'))
                   RETURNING id""",
                (request_id, node_name),
            )
            self._conn.commit()
            return cursor.fetchone()["id"]

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
                (status, json.dumps(output) if output else None, error, duration_ms, node_id),
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

    def get_pipeline_summary(self, request_id: str) -> dict:
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

        skills = self._conn.execute(
            "SELECT * FROM skills_generated WHERE request_id = ?",
            (request_id,),
        ).fetchall()

        cost = self.history.get_cost_summary(request_id)

        return {
            "pipeline": dict(pipeline) if pipeline else None,
            "nodes": nodes,
            "history_turns": len(history),
            "skills_generated": len(skills),
            "total_cost_cents": cost.get("total_cost_cents", 0),
            "total_tokens": cost.get("total_tokens", 0),
        }

    # =====================================================================
    # Skills
    # =====================================================================

    def register_skill(self, request_id: str, skill_name: str,
                       skill_path: str, source: str, tags: List[str]) -> int:
        """Register a skill created during a pipeline run."""
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO skills_generated
                   (request_id, skill_name, skill_path, source, tags)
                   VALUES (?, ?, ?, ?, ?)""",
                (request_id, skill_name, skill_path, source, json.dumps(tags)),
            )
            self._conn.commit()
            return cursor.lastrowid

    def increment_skill_use(self, skill_name: str) -> None:
        """Increment use count when a skill is applied."""
        with self._lock:
            self._conn.execute(
                "UPDATE skills_generated SET use_count = use_count + 1 WHERE skill_name = ?",
                (skill_name,),
            )
            self._conn.commit()

    # =====================================================================
    # Recovery
    # =====================================================================

    def find_stalled(self, stalled_minutes: int = 5) -> List[dict]:
        """Find stalled pipelines that may need recovery."""
        return recovery.find_stalled_pipelines(
            self.db_path, stalled_minutes
        )

    def mark_completed(self, request_id: str) -> None:
        """Mark pipeline as completed."""
        self.save_pipeline_state(request_id, {"status": "completed"})
        # Also mark in checkpointer
        thread_id = self._conn.execute(
            "SELECT thread_id FROM pipeline_states WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if thread_id and thread_id["thread_id"]:
            recovery.mark_completed(self.db_path, thread_id["thread_id"])

    # =====================================================================
    # Query
    # =====================================================================

    def query(self, sql: str, params=()) -> List[dict]:
        """Run a raw query against the state database.

        Useful for custom analytics, dashboards, or debugging.
        All tables are queryable: checkpoints, agent_history,
        pipeline_states, pipeline_nodes, skills_generated.
        """
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

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
        """Graceful shutdown."""
        self.checkpoint_wal()
        self._conn.close()


class _NodeTracker:
    """Context manager for automatic node tracking."""

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
