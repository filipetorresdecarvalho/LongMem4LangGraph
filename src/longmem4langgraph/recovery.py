"""LangGraph Crash Recovery — detect and resume interrupted pipelines.

When a LangGraph pipeline crashes (server restart, exception, OOM),
the checkpoint data is safe in SQLite but the pipeline is in
'processing' limbo. This module detects those pipelines and
provides resume capabilities.

Changes from original:
- Now queries pipeline_states table (not checkpoints) for stalled detection
- mark_completed/mark_failed merge with existing metadata (don't overwrite)
- Fixed graph.arainvoke() → graph.ainvoke()
- Uses shared connection when available

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

try:
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph
except ImportError:
    try:
        from langgraph.graph.graph import CompiledGraph
    except ImportError:
        CompiledGraph = Any

from .connection import SqliteConnection

logger = logging.getLogger(__name__)


def find_stalled_pipelines(
    db_path: str = None,
    stalled_minutes: int = 5,
    status_filter: Optional[List[str]] = None,
    connection: SqliteConnection = None,
) -> List[dict]:
    """Find pipelines that are stuck in 'processing' state.

    Queries the pipeline_states table (not checkpoints) for pipelines
    whose status is 'processing' or 'running' and haven't been updated
    within the staleness threshold.

    Args:
        db_path: Path to the SQLite state file (ignored if connection given).
        stalled_minutes: Age threshold for considering a pipeline stalled.
        status_filter: Statuses to consider stalled. Default: ['processing', 'running'].
        connection: Optional shared SqliteConnection.

    Returns:
        List of dicts with request_id, status, last updated, and thread info.
    """
    filters = status_filter or ["processing", "running"]
    placeholders = ",".join("?" for _ in filters)

    close_conn = False
    if connection:
        conn = connection
    else:
        conn = SqliteConnection(db_path or "pipeline.db")
        close_conn = True

    try:
        rows = conn.execute(
            f"""SELECT request_id, thread_id, status, current_node,
                       error, created_at, updated_at
                FROM pipeline_states
                WHERE status IN ({placeholders})
                AND updated_at < datetime('now', ?)
                ORDER BY updated_at""",
            (*filters, f"-{stalled_minutes} minutes"),
        ).fetchall()
    finally:
        if close_conn:
            conn.close()

    return [dict(r) for r in rows]


async def recover_pipelines(
    db_path: str,
    graph: CompiledGraph,
    stalled_minutes: int = 5,
    status_filter: Optional[List[str]] = None,
    max_recover: int = 10,
    connection: SqliteConnection = None,
) -> List[str]:
    """Detect and resume stalled pipelines automatically.

    Call this on system startup. It finds pipelines that were
    interrupted, determines the last known state, and marks them
    for resumption.

    Note: Full auto-resume (re-invoking the graph from last checkpoint)
    requires the graph to accept config with a thread_id. For complex
    LangGraph pipelines, manual review of stalled pipelines is
    recommended before automatic resumption.

    Args:
        db_path: Path to the SQLite state file.
        graph: Compiled LangGraph to resume pipelines on.
        stalled_minutes: Age threshold for stalled detection.
        status_filter: Only recover these statuses.
        max_recover: Max pipelines to recover in one call.
        connection: Optional shared SqliteConnection.

    Returns:
        List of recovered thread_ids.
    """
    stalled = find_stalled_pipelines(db_path, stalled_minutes, status_filter, connection)
    recovered = []

    for pipeline in stalled[:max_recover]:
        thread_id = pipeline.get("thread_id")
        request_id = pipeline.get("request_id")

        if not thread_id:
            logger.warning("Pipeline %s has no thread_id, skipping recovery", request_id)
            continue

        try:
            logger.info(
                "Attempting recovery of pipeline %s (thread %s)",
                request_id,
                thread_id,
            )

            # Build a config that resumes from the last checkpoint
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            # Resume the graph — LangGraph uses the checkpointer
            # to load the latest state from SQLite
            state = await graph.ainvoke(
                input=None,  # Resume, don't restart
                config=config,
            )

            recovered.append(request_id)
            logger.info("Successfully recovered pipeline %s", request_id)

        except Exception as e:
            logger.error(
                "Failed to recover pipeline %s: %s",
                request_id,
                str(e),
            )

    return recovered


def _merge_metadata(existing_raw: str, new_fields: dict) -> str:
    """Merge new fields into existing metadata JSON.

    Preserves existing LangGraph metadata fields and only adds/updates
    the new ones. If existing_raw is empty or unparseable, returns
    just the new fields.
    """
    if not existing_raw or existing_raw == "{}":
        return json.dumps(new_fields)

    try:
        existing = json.loads(existing_raw)
    except (json.JSONDecodeError, TypeError):
        existing = {}

    existing.update(new_fields)
    return json.dumps(existing)


def mark_completed(db_path: str, thread_id: str, connection: SqliteConnection = None) -> None:
    """Mark a pipeline as completed (moved from processing to completed).

    Merges with existing metadata — preserves LangGraph's own fields
    (source, step, writes, etc.) instead of overwriting them.

    Call this at the END node of your graph to mark completion,
    so recovery doesn't attempt to resume finished pipelines.

    Args:
        db_path: Path to the SQLite state file (ignored if connection given).
        thread_id: The pipeline thread to mark.
        connection: Optional shared SqliteConnection.
    """
    close_conn = False
    if connection:
        conn = connection
    else:
        conn = SqliteConnection(db_path or "pipeline.db")
        close_conn = True

    try:
        row = conn.execute(
            """SELECT checkpoint_id, metadata FROM checkpoints
               WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()

        if row:
            merged = _merge_metadata(
                row["metadata"],
                {"status": "completed", "recovered": False, "completed_at": str(datetime.now(timezone.utc))},
            )
            conn.execute(
                """UPDATE checkpoints SET metadata = ?
                   WHERE thread_id = ? AND checkpoint_id = ?""",
                (merged, thread_id, row["checkpoint_id"]),
            )
            conn.commit()
    finally:
        if close_conn:
            conn.close()


def mark_failed(db_path: str, thread_id: str, error: str, connection: SqliteConnection = None) -> None:
    """Mark a pipeline as failed (not recoverable).

    Merges with existing metadata instead of overwriting.

    Args:
        db_path: Path to the SQLite state file (ignored if connection given).
        thread_id: The pipeline thread to mark.
        error: Error description.
        connection: Optional shared SqliteConnection.
    """
    close_conn = False
    if connection:
        conn = connection
    else:
        conn = SqliteConnection(db_path or "pipeline.db")
        close_conn = True

    try:
        row = conn.execute(
            """SELECT checkpoint_id, metadata FROM checkpoints
               WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()

        if row:
            merged = _merge_metadata(
                row["metadata"],
                {"status": "failed", "error": error, "recovered": False, "failed_at": str(datetime.now(timezone.utc))},
            )
            conn.execute(
                """UPDATE checkpoints SET metadata = ?
                   WHERE thread_id = ? AND checkpoint_id = ?""",
                (merged, thread_id, row["checkpoint_id"]),
            )
            conn.commit()
    finally:
        if close_conn:
            conn.close()
