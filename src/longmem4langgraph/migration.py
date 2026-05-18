"""Schema migration system for LongMem4LangGraph.

Uses SQLite PRAGMA user_version for lightweight version tracking.
Each migration is a function that receives a connection and performs
schema changes. All migrations are idempotent (safe to run on existing
databases created by v0.1).

Usage:
    from longmem4langgraph.migration import ensure_migrated

    conn = SqliteConnection("pipeline.db")
    version = ensure_migrated(conn)

To add a new migration:
    from longmem4langgraph.migration import register_migration

    @register_migration(2)
    def _v2_my_feature(conn):
        conn.execute(\"CREATE INDEX IF NOT EXISTS ...\")
"""

import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

_MIGRATIONS: Dict[int, Callable] = {}


def register_migration(version: int):
    """Decorator to register a migration function.

    Migrations run in order. Each receives a connection and should
    use CREATE IF NOT EXISTS / INSERT OR IGNORE for idempotency.

    Args:
        version: Migration version number (must be sequential, starting from 1).

    Raises:
        ValueError: If a migration for this version is already registered.
    """
    def decorator(fn: Callable) -> Callable:
        if version in _MIGRATIONS:
            raise ValueError(f"Migration v{version} already registered")
        _MIGRATIONS[version] = fn
        return fn
    return decorator


def run_migrations(conn) -> int:
    """Run all pending migrations on the given connection.

    Returns:
        The new schema version after all migrations are applied.
    """
    current = _get_version(conn)
    pending = sorted(v for v in _MIGRATIONS if v > current)

    if not pending:
        return current

    for version in pending:
        logger.info("Running migration v%d", version)
        _MIGRATIONS[version](conn)

    new_version = max(pending)
    _set_version(conn, new_version)
    conn.commit()
    logger.info("Schema migrated to v%d", new_version)
    return new_version


def _get_version(conn) -> int:
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def _set_version(conn, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {version}")


def ensure_migrated(conn) -> int:
    """Ensure database schema is up to date. Call on module init.

    Safe to call multiple times — no-op if already at latest version.

    Args:
        conn: A SqliteConnection (or any object with execute/commit/executescript).

    Returns:
        The current schema version.
    """
    return run_migrations(conn)


@register_migration(1)
def _v1_initial_schema(conn) -> None:
    """Initial schema — all core tables for v0.1 compatibility.

    Uses CREATE TABLE IF NOT EXISTS so this is safe to run on
    databases created by v0.1 (tables already exist = no-op).
    """
    conn.executescript("""
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

        CREATE TABLE IF NOT EXISTS agent_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            thread_id TEXT,
            turn_number INTEGER NOT NULL,
            source TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'text',
            content TEXT NOT NULL,
            summary TEXT,
            metadata TEXT DEFAULT '{}',
            token_count INTEGER,
            cost_cents REAL DEFAULT 0.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_history_request
            ON agent_history(request_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_history_thread
            ON agent_history(thread_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_history_source
            ON agent_history(request_id, source);

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
    """)
