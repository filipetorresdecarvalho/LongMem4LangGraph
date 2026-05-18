"""LangGraph BaseCheckpointSaver implementation backed by SQLite.

This is the core persistence layer — it replaces LangGraph's default
in-memory checkpointer with a SQLite-backed one that survives restarts.

Implements the CheckpointStore ABC interface for extensibility.
Users can create custom backends by implementing CheckpointStore.

What it stores:
    - Full graph state at each checkpoint step
    - Thread/run metadata for multi-session support
    - Timestamps for all transitions

Can accept a shared SqliteConnection to avoid multiple connections
to the same database file.

Usage:
    from longmem4langgraph import SqliteSaver

    graph = builder.compile(checkpointer=SqliteSaver("state.db"))
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
from .migration import ensure_migrated


class SqliteSaver(BaseCheckpointSaver):
    """LangGraph checkpointer backed by SQLite.

    Drop-in replacement for LangGraph's MemorySaver or any other
    BaseCheckpointSaver. Use it when compiling your StateGraph:

        from longmem4langgraph import SqliteSaver
        graph = builder.compile(checkpointer=SqliteSaver("state.db"))

    Implements the CheckpointStore ABC interface. Users who want a
    different database backend (PostgreSQL, MySQL, etc.) should
    implement CheckpointStore instead of subclassing this class.

    Use with a shared connection to avoid write contention:

        conn = SqliteConnection("state.db")
        saver = SqliteSaver(connection=conn)
        history = HistoryStore(connection=conn)

    Args:
        db_path: Path to SQLite file (only used if connection is None).
        serde: Serializer (defaults to JsonPlusSerializer).
        connection: Optional shared SqliteConnection.
    """

    def __init__(
        self,
        db_path: str = None,
        serde: Optional[Any] = None,
        connection: SqliteConnection = None,
    ):
        super().__init__(serde=serde or JsonPlusSerializer())

        if connection:
            self._conn = connection
        else:
            self._conn = SqliteConnection(db_path or "pipeline.db")

        self._lock = threading.Lock()
        ensure_migrated(self._conn)

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
        config: Optional[dict] = None,
        *,
        filter: Optional[dict] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread, optionally filtered."""
        if config is None:
            return iter([])

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
        new_versions: Any,
    ) -> dict:
        """Save a checkpoint. Called by LangGraph after each node execution."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = str(checkpoint["id"])

        parent_id = None
        if config["configurable"].get("checkpoint_id"):
            parent_id = config["configurable"]["checkpoint_id"]

        checkpoint_data = self._serialize_checkpoint(checkpoint)
        metadata_data = self._serialize_metadata(metadata)

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
                    checkpoint_data,
                    metadata_data,
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
        task_path: str = "",
    ) -> None:
        """Store pending writes for a checkpoint (for async tasks)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._lock:
            for idx, write in enumerate(writes):
                channel, value = write
                type_str, value_bytes = self.serde.dumps_typed(value)
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
                        type_str,
                        value_bytes,
                    ),
                )
            self._conn.commit()

    # =====================================================================
    # Serialization helpers — uses serde for proper type handling
    # =====================================================================

    def _serialize_checkpoint(self, checkpoint: dict) -> bytes:
        type_str, data = self.serde.dumps_typed(checkpoint)
        return data

    def _deserialize_checkpoint(self, raw: Any) -> dict:
        if not raw:
            return {}
        if isinstance(raw, str):
            raw = raw.encode()
        if isinstance(raw, (bytes, memoryview)):
            return self.serde.loads_typed(("msgpack", bytes(raw)))
        return raw

    def _serialize_metadata(self, metadata: dict) -> bytes:
        type_str, data = self.serde.dumps_typed(metadata)
        return data

    def _deserialize_metadata(self, raw: Any) -> dict:
        if not raw:
            return {}
        if isinstance(raw, str):
            raw = raw.encode()
        if isinstance(raw, (bytes, memoryview)):
            return self.serde.loads_typed(("msgpack", bytes(raw)))
        return raw

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
