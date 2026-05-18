"""Thread-safe SQLite connection management with WAL mode.

Used as the foundation for all LongMem4LangGraph components.
Can be shared across checkpointer, history, and recovery modules.
"""

import sqlite3
import threading
import warnings
from pathlib import Path
from typing import Optional


class SqliteConnection:
    """Thread-safe SQLite connection with WAL mode.

    Pool-free design — SQLite with WAL handles concurrent reads
    natively. Writes are serialized via a lock.

    Supports context manager (with statement) for automatic cleanup.

    Usage:
        conn = SqliteConnection("pipeline.db")
        with conn.transaction():
            conn.execute("INSERT INTO ...")
        rows = conn.execute("SELECT * FROM ...")
    """

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=10,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64 MB cache
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row

        self._write_lock = threading.Lock()
        self._closed = False

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        """Execute a read query. Thread-safe (WAL allows concurrent reads)."""
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self._conn.executemany(sql, params_seq)

    def executescript(self, sql: str) -> None:
        self._conn.executescript(sql)

    def commit(self) -> None:
        with self._write_lock:
            self._conn.commit()

    def transaction(self):
        """Context manager for atomic writes. Auto-commits on success."""
        return _Transaction(self)

    def close(self) -> None:
        """Graceful shutdown — flush WAL to main file before closing."""
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self._conn.close()

    def __del__(self) -> None:
        """Destructor — ensures connection is closed even if close() not called."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    def __enter__(self) -> "SqliteConnection":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def total_changes(self) -> int:
        return self._conn.total_changes

    def __repr__(self) -> str:
        return f"<SqliteConnection {self.db_path}>"


class _Transaction:
    """Context manager for atomic writes."""

    def __init__(self, conn: SqliteConnection):
        self._conn = conn

    def __enter__(self) -> "SqliteConnection":
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn._conn.rollback()
