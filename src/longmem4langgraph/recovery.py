"""LangGraph Crash Recovery — detect and resume interrupted pipelines.

When a LangGraph pipeline crashes (server restart, exception, OOM),
the checkpoint data is safe in SQLite but the pipeline is in
'processing' limbo. This module detects those pipelines and
provides resume capabilities.

Usage:
    from longmem4langgraph import recover_pipelines

    # On system startup:
    graph = build_my_graph()
    await recover_pipelines("state.db", graph)
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any

from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from .connection import SqliteConnection

logger = logging.getLogger(__name__)


def find_stalled_pipelines(
    db_path: str,
    stalled_minutes: int = 5,
    status_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Find pipelines that are stuck in 'processing' state.

    Looks for checkpoints that were started but never completed.
    A pipeline is considered stalled if its most recent checkpoint
    is older than `stalled_minutes` and has no completion marker.

    Args:
        db_path: Path to the SQLite state file.
        stalled_minutes: Age threshold for considering a pipeline stalled.
        status_filter: Only check these statuses. Default: ['processing', 'running'].

    Returns:
        List of dicts with thread_id and last_checkpoint details.
    """
    conn = SqliteConnection(db_path)
    filters = status_filter or ["processing", "running"]

    try:
        rows = conn.execute(
            """SELECT c.thread_id, c.checkpoint_id, c.checkpoint,
                      c.metadata, c.created_at
               FROM checkpoints c
               WHERE c.created_at < datetime('now', ?)
               AND (c.metadata LIKE ?)
               ORDER BY c.created_at DESC""",
            (f"-{stalled_minutes} minutes", f"%{filters[0]}%"),
        ).fetchall()

        # Check for multiple statuses
        if len(filters) > 1:
            all_rows = []
            for f in filters:
                r = conn.execute(
                    """SELECT c.thread_id, c.checkpoint_id, c.checkpoint,
                              c.metadata, c.created_at
                       FROM checkpoints c
                       WHERE c.created_at < datetime('now', ?)
                       AND c.metadata LIKE ?
                       ORDER BY c.created_at DESC""",
                    (f"-{stalled_minutes} minutes", f"%{f}%"),
                ).fetchall()
                all_rows.extend(r)
            rows = all_rows

    finally:
        conn.close()

    return [dict(r) for r in rows]


async def recover_pipelines(
    db_path: str,
    graph: CompiledGraph,
    stalled_minutes: int = 5,
    status_filter: Optional[List[str]] = None,
    max_recover: int = 10,
) -> List[str]:
    """Detect and resume stalled pipelines automatically.

    Call this on system startup. It finds pipelines that were
    interrupted, determines the last executed node, and resumes
    them from that point.

    Args:
        db_path: Path to the SQLite state file.
        graph: Compiled LangGraph to resume pipelines on.
        stalled_minutes: Age threshold for stalled detection.
        status_filter: Only recover these statuses.
        max_recover: Max pipelines to recover in one call.

    Returns:
        List of recovered thread_ids.
    """
    stalled = find_stalled_pipelines(db_path, stalled_minutes, status_filter)
    recovered = []

    for pipeline in stalled[:max_recover]:
        thread_id = pipeline["thread_id"]
        last_checkpoint_id = pipeline["checkpoint_id"]

        try:
            logger.info(
                "Recovering pipeline %s from checkpoint %s",
                thread_id,
                last_checkpoint_id,
            )

            # Build a config that resumes from the last checkpoint
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": last_checkpoint_id,
                }
            }

            # Resume the graph — LangGraph handles the rest
            # Uses the checkpointer's existing data
            state = await graph.arainvoke(
                input=None,  # Resume, don't restart
                config=config,
            )

            recovered.append(thread_id)
            logger.info("Successfully recovered pipeline %s", thread_id)

        except Exception as e:
            logger.error(
                "Failed to recover pipeline %s: %s",
                thread_id,
                str(e),
            )

    return recovered


def mark_completed(db_path: str, thread_id: str) -> None:
    """Mark a pipeline as completed (moved from processing to completed).

    Call this at the END node of your graph to mark completion,
    so recovery doesn't attempt to resume finished pipelines.

    Args:
        db_path: Path to the SQLite state file.
        thread_id: The pipeline thread to mark.
    """
    conn = SqliteConnection(db_path)
    try:
        row = conn.execute(
            """SELECT checkpoint_id, checkpoint FROM checkpoints
               WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()

        if row:
            metadata = {"status": "completed", "recovered": False}
            conn.execute(
                """UPDATE checkpoints SET metadata = ?
                   WHERE thread_id = ? AND checkpoint_id = ?""",
                (json.dumps(metadata), thread_id, row["checkpoint_id"]),
            )
            conn.commit()
    finally:
        conn.close()


def mark_failed(db_path: str, thread_id: str, error: str) -> None:
    """Mark a pipeline as failed (not recoverable).

    Args:
        db_path: Path to the SQLite state file.
        thread_id: The pipeline thread to mark.
        error: Error description.
    """
    conn = SqliteConnection(db_path)
    try:
        row = conn.execute(
            """SELECT checkpoint_id, checkpoint FROM checkpoints
               WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()

        if row:
            metadata = {"status": "failed", "error": error, "recovered": False}
            conn.execute(
                """UPDATE checkpoints SET metadata = ?
                   WHERE thread_id = ? AND checkpoint_id = ?""",
                (json.dumps(metadata), thread_id, row["checkpoint_id"]),
            )
            conn.commit()
    finally:
        conn.close()
