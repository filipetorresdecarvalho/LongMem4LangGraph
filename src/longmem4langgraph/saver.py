"""LangGraph BaseCheckpointSaver implementation backed by SQLite.

This is the core persistence layer — it replaces LangGraph's default
in-memory checkpointer with a SQLite-backed one that survives restarts.

What it stores:
    - Full graph state at each checkpoint step
    - Thread/run metadata for multi-session support
    - Timestamps for all transitions
"""

import json
import threading
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .connection import SqliteConnection


class SqliteSaver(BaseCheckpointSaver):
    """LangGraph checkpointer backed by SQLite.

    Drop-in replacement for LangGraph's MemorySaver or any other
    BaseCheckpointSaver. Use it when compiling your StateGraph:

        from longmem4langgraph import SqliteSaver
        graph = builder.compile(checkpointer=SqliteSaver("state.db"))

    Args:
        db_path: Path to SQLite file. Created automatically if it doesn't exist.
        serde: Serializer (defaults to JsonPlusSerializer).
    """

    def __init__(
        self,
        db_path: str,
        serde: Optional[Any] = None,
    ):
        super().__init__(serde=serde or JsonPlusSerializer())
        self._conn = SqliteConnection(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create checkpoint tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );

            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT,
                value BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
                ON checkpoints(thread_id, checkpoint_ns, checkpoint_id);
            CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread
                ON checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id);
        """)
        self._conn.commit()

    # =====================================================================
    # Required BaseCheckpointSaver overrides
    # =====================================================================

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Get the latest checkpoint for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        row = self._conn.execute(
            """SELECT checkpoint, metadata, parent_checkpoint_id, created_at
               FROM checkpoints
               WHERE thread_id = ? AND checkpoint_ns = ?
               ORDER BY created_at DESC LIMIT 1""",
            (thread_id, checkpoint_ns),
        ).fetchone()

        if row is None:
            return None

        checkpoint = self._deserialize_checkpoint(row["checkpoint"])
        metadata = self._deserialize_metadata(row["metadata"])

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": str(checkpoint["id"]),
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": str(row["parent_checkpoint_id"]),
                    }
                }
                if row["parent_checkpoint_id"]
                else None
            ),
        )

    def list(
        self,
        config: dict,
        *,
        filter: Optional[dict] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread, optionally filtered."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        query = """SELECT checkpoint, metadata, parent_checkpoint_id, created_at
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?"""
        params = [thread_id, checkpoint_ns]

        if before is not None:
            query += " AND created_at < ?"
            params.append(before)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        rows = self._conn.execute(query, params).fetchall()

        for row in rows:
            checkpoint = self._deserialize_checkpoint(row["checkpoint"])
            metadata = self._deserialize_metadata(row["metadata"])

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": str(checkpoint["id"]),
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": str(row["parent_checkpoint_id"]),
                        }
                    }
                    if row["parent_checkpoint_id"]
                    else None
                ),
            )

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> dict:
        """Save a checkpoint. Called by LangGraph after each node execution."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = str(checkpoint["id"])

        parent_id = None
        if config["configurable"].get("checkpoint_id"):
            parent_id = config["configurable"]["checkpoint_id"]

        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO checkpoints
                   (thread_id, checkpoint_ns, checkpoint_id,
                    parent_checkpoint_id, type, checkpoint, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_id,
                    checkpoint.get("type", ""),
                    json.dumps(checkpoint),
                    json.dumps(metadata),
                ),
            )
            self._conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: list,
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint (for async tasks)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._lock:
            for idx, write in enumerate(writes):
                channel, value = write
                self._conn.execute(
                    """INSERT OR REPLACE INTO checkpoint_writes
                       (thread_id, checkpoint_ns, checkpoint_id,
                        task_id, idx, channel, type, value)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        self.SERIALIZER.typetuple(value),
                        self.SERIALIZER.dumps(value),
                    ),
                )
            self._conn.commit()

    # =====================================================================
    # Serialization helpers
    # =====================================================================

    def _deserialize_checkpoint(self, raw: str) -> dict:
        if isinstance(raw, str):
            return json.loads(raw)
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    def _deserialize_metadata(self, raw: str) -> dict:
        if not raw or raw == "{}":
            return {}
        if isinstance(raw, str):
            return json.loads(raw)
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    # =====================================================================
    # Utility
    # =====================================================================

    def get_state_summary(self, thread_id: str) -> dict:
        """Get a quick summary of a thread's state."""
        row = self._conn.execute(
            """SELECT checkpoint_id, parent_checkpoint_id, metadata, created_at
               FROM checkpoints
               WHERE thread_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()

        if row is None:
            return {"thread_id": thread_id, "status": "not_found"}

        count = self._conn.execute(
            "SELECT COUNT(*) as c FROM checkpoints WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]

        return {
            "thread_id": thread_id,
            "status": "active",
            "checkpoints": count,
            "last_checkpoint": row["checkpoint_id"],
            "last_updated": row["created_at"],
        }
