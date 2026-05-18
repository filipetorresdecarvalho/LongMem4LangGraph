"""AGNO-style full conversation history for agents.

Gives LangGraph agents access to the complete history of what
previous agents said — not just the current state.

In AGNO, agents have session.memory that persists across turns.
This module replicates that behavior with SQLite persistence,
so every agent call can read what came before.

Usage:
    from longmem4langgraph import HistoryStore

    history = HistoryStore("pipeline.db")

    def my_agent(state):
        # Read what previous agents wrote
        context = history.get_context(state["request_id"], last_n=10)

        # Write what this agent did
        history.add(state["request_id"], "agent3", result)
        ...
"""

import json
import threading
from datetime import datetime, timezone
from typing import List, Optional

from .connection import SqliteConnection


class HistoryStore:
    """AGNO-style conversation memory backed by SQLite.

    Each 'turn' in the conversation is stored with source, content,
    content_type, and metadata. Agents can query the full history
    as a formatted string for prompt injection.

    Args:
        db_path: Path to SQLite file. Can share with SqliteSaver.
    """

    def __init__(self, db_path: str):
        self._conn = SqliteConnection(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create history tables."""
        self._conn.executescript("""
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
        """)
        self._conn.commit()

    def add(
        self,
        request_id: str,
        source: str,
        content: str,
        content_type: str = "text",
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        token_count: Optional[int] = None,
        cost_cents: float = 0.0,
    ) -> int:
        """Add a conversation turn (like AGNO's session.memory).

        Args:
            request_id: Pipeline/request identifier.
            source: Who wrote this (agent1, agent2, user, system, etc.).
            content: The actual message/output.
            content_type: text, json, code, error, decision, etc.
            summary: Short summary for context when full content is too long.
            metadata: Extra structured data (key used, model, duration, etc.).

        Returns:
            The turn number (1-indexed).
        """
        # Auto-increment turn number
        turn = self._conn.execute(
            """SELECT COALESCE(MAX(turn_number), 0) + 1
               FROM agent_history WHERE request_id = ?""",
            (request_id,),
        ).fetchone()[0]

        with self._lock:
            self._conn.execute(
                """INSERT INTO agent_history
                   (request_id, thread_id, turn_number, source,
                    content_type, content, summary, metadata,
                    token_count, cost_cents)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    thread_id,
                    turn,
                    source,
                    content_type,
                    content,
                    summary,
                    json.dumps(metadata or {}),
                    token_count,
                    cost_cents,
                ),
            )
            self._conn.commit()

        return turn

    def get_history(
        self,
        request_id: str,
        last_n: Optional[int] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        """Get full conversation history as a list of dicts.

        This is like AGNO's session.memory — returns every turn
        in chronological order.

        Args:
            request_id: The pipeline to query.
            last_n: Only return the last N turns.
            source_filter: Only include these sources.

        Returns:
            List of history entries, oldest first.
        """
        query = "SELECT * FROM agent_history WHERE request_id = ?"
        params = [request_id]

        if source_filter:
            placeholders = ",".join("?" for _ in source_filter)
            query += f" AND source IN ({placeholders})"
            params.extend(source_filter)

        query += " ORDER BY turn_number"

        if last_n:
            # Subquery to get last N turns
            query = f"""
                SELECT * FROM (
                    SELECT * FROM agent_history WHERE request_id = ?
                    {"AND source IN (" + ",".join("?" for _ in source_filter) + ")" if source_filter else ""}
                    ORDER BY turn_number DESC
                    LIMIT {int(last_n)}
                ) ORDER BY turn_number
            """
            params = [request_id]
            if source_filter:
                params.extend(source_filter)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_context(
        self,
        request_id: str,
        last_n: int = 10,
        max_content_length: int = 2000,
        format_template: Optional[str] = None,
    ) -> str:
        """Get history formatted as a string for LLM prompt injection.

        This is the main method you'll use inside agent nodes.
        It returns a formatted string you can inject into any prompt.

        Args:
            request_id: The pipeline to query.
            last_n: How many recent turns to include.
            max_content_length: Truncate content to this many chars.
            format_template: Optional custom format. Default:
                "[{source.upper()}]: {content}"

        Returns:
            Formatted string, ready for prompt injection.
        """
        history = self.get_history(request_id, last_n=last_n)
        template = format_template or "[{source}]: {content}"

        parts = []
        for h in history:
            content = h["content"]
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n... [truncated]"

            source_tag = h["source"].upper()
            parts.append(template.format(source=source_tag, content=content))

        return "\n\n".join(parts)

    def count_turns(self, request_id: str) -> int:
        """Count total turns for a request."""
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM agent_history WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return row["c"]

    def get_cost_summary(self, request_id: str) -> dict:
        """Get total cost and token usage for a request."""
        row = self._conn.execute(
            """SELECT
                   COUNT(*) as turns,
                   COALESCE(SUM(cost_cents), 0) as total_cost_cents,
                   COALESCE(SUM(token_count), 0) as total_tokens,
                   COUNT(*) FILTER (WHERE source LIKE 'agent%') as agent_turns
               FROM agent_history WHERE request_id = ?""",
            (request_id,),
        ).fetchone()
        return dict(row)

    def search(self, request_id: str, query_text: str) -> List[dict]:
        """Basic text search over history content.

        For production, use Weaviate or other vector search.
        This is a simple LIKE-based search for quick debugging.
        """
        rows = self._conn.execute(
            """SELECT * FROM agent_history
               WHERE request_id = ? AND content LIKE ?
               ORDER BY turn_number""",
            (request_id, f"%{query_text}%"),
        ).fetchall()
        return [dict(r) for r in rows]
