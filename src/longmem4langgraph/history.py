"""AGNO-style full conversation history for agents.

Gives LangGraph agents access to the complete history of what
previous agents said — not just the current state.

Implements the MemoryStore ABC interface for extensibility.
Users can create custom backends by implementing MemoryStore.

In AGNO, agents have session.memory that persists across turns.
This module replicates that behavior with SQLite persistence,
so every agent call can read what came before.

Supports pluggable search backends:
- Default: SqliteLikeSearch (zero dependencies, LIKE-based)
- Optional: WeaviateSearch (semantic vector search via Weaviate)

Usage:
    from longmem4langgraph import HistoryStore
    from longmem4langgraph.search import WeaviateSearch

    history = HistoryStore("pipeline.db")

    # With semantic search
    history = HistoryStore("pipeline.db",
                           search_backend=WeaviateSearch())

    def my_agent(state):
        context = history.get_context(state["request_id"], last_n=10)
        history.add_turn(state["request_id"], "agent3", result)
"""

import json
import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional

from .connection import SqliteConnection
from .migration import ensure_migrated
from .search import SqliteLikeSearch, SearchBackend

logger = logging.getLogger(__name__)


def _validate_non_empty(value, name: str) -> None:
    if not value or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_positive_int(value, name: str) -> None:
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer, got {value}")


class HistoryStore:
    """AGNO-style conversation memory backed by SQLite.

    Implements the MemoryStore ABC interface. Each 'turn' in the
    conversation is stored with source, content, content_type, and
    metadata. Agents can query the full history as a formatted
    string for prompt injection.

    Supports pluggable search backends for semantic or keyword search.
    Defaults to SqliteLikeSearch (zero dependencies).

    Args:
        db_path: Path to SQLite file. Can share with SqliteSaver.
        connection: Optional shared SqliteConnection (avoids multiple connections).
        search_backend: Optional SearchBackend for semantic search.
            Defaults to SqliteLikeSearch (LIKE-based, zero deps).
            Pass WeaviateSearch() for semantic vector search.
    """

    def __init__(
        self,
        db_path: str = None,
        connection: SqliteConnection = None,
        search_backend: SearchBackend = None,
    ):
        if connection:
            self._conn = connection
        else:
            self._conn = SqliteConnection(db_path or "pipeline.db")

        self._lock = threading.Lock()
        self._search_backend = search_backend or SqliteLikeSearch(self._conn)
        ensure_migrated(self._conn)

    def add_turn(
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

        Also indexes the content in the configured search backend
        (non-fatal if indexing fails).

        Args:
            request_id: Pipeline/request identifier.
            source: Who wrote this (agent1, agent2, user, system, etc.).
            content: The actual message/output.
            content_type: text, json, code, error, decision, etc.
            summary: Short summary for context when full content is too long.
            metadata: Extra structured data (key used, model, duration, etc.).
            thread_id: Optional thread identifier for cross-thread queries.
            token_count: LLM token count for cost tracking.
            cost_cents: Cost in cents for cost tracking.

        Returns:
            The turn number (1-indexed).
        """
        _validate_non_empty(request_id, "request_id")
        _validate_non_empty(source, "source")

        with self._lock:
            self._conn.execute(
                """INSERT INTO agent_history
                   (request_id, thread_id, turn_number, source,
                    content_type, content, summary, metadata,
                    token_count, cost_cents)
                   SELECT ?, ?, COALESCE(MAX(turn_number), 0) + 1,
                          ?, ?, ?, ?, ?, ?, ?
                   FROM agent_history WHERE request_id = ?""",
                (
                    request_id,
                    thread_id,
                    source,
                    content_type,
                    content,
                    summary or content[:200],
                    json.dumps(metadata or {}),
                    token_count,
                    cost_cents,
                    request_id,
                ),
            )
            self._conn.commit()

            turn = self._conn.execute(
                "SELECT MAX(turn_number) FROM agent_history WHERE request_id = ?",
                (request_id,),
            ).fetchone()[0]

        if self._search_backend:
            try:
                self._search_backend.index(
                    request_id, source, content,
                    turn_number=turn,
                    content_type=content_type,
                    metadata=metadata,
                    thread_id=thread_id,
                )
            except Exception as e:
                logger.debug("Search backend index failed (non-fatal): %s", e)

        return turn

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
        """Alias for add_turn() — preserved for backward compatibility."""
        return self.add_turn(
            request_id, source, content,
            content_type=content_type,
            summary=summary,
            metadata=metadata,
            thread_id=thread_id,
            token_count=token_count,
            cost_cents=cost_cents,
        )

    def get_history(
        self,
        request_id: str,
        last_n: Optional[int] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        """Get full conversation history as a list of dicts.

        Args:
            request_id: The pipeline to query.
            last_n: Only return the last N turns.
            source_filter: Only include these sources.

        Returns:
            List of history entries, oldest first.
        """
        _validate_non_empty(request_id, "request_id")
        _validate_positive_int(last_n, "last_n")

        query = "SELECT * FROM agent_history WHERE request_id = ?"
        params: list = [request_id]

        if source_filter:
            placeholders = ",".join("?" for _ in source_filter)
            query += f" AND source IN ({placeholders})"
            params.extend(source_filter)

        query += " ORDER BY turn_number"

        if last_n:
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

        Args:
            request_id: The pipeline to query.
            last_n: How many recent turns to include.
            max_content_length: Truncate content to this many chars.
            format_template: Optional custom format. Default:
                "[{source.upper()}]: {content}"

        Returns:
            Formatted string, ready for prompt injection.
        """
        _validate_non_empty(request_id, "request_id")
        _validate_positive_int(last_n, "last_n")

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

    def search(self, request_id: str, query_text: str, limit: int = 10) -> List[dict]:
        """Search conversation history.

        Uses the configured search backend (WeaviateSearch for semantic,
        SqliteLikeSearch for keyword). Falls back to LIKE if the
        backend returns no results or fails.

        Args:
            request_id: The pipeline to search within.
            query_text: The search query.
            limit: Maximum results to return.

        Returns:
            List of matching history entries.
        """
        _validate_non_empty(request_id, "request_id")

        if self._search_backend:
            try:
                results = self._search_backend.search(query_text, request_id, limit)
                if results:
                    return results
            except Exception as e:
                logger.debug("Search backend failed, falling back to LIKE: %s", e)

        safe_query = query_text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._conn.execute(
            """SELECT * FROM agent_history
               WHERE request_id = ? AND content LIKE ?
               ORDER BY turn_number""",
            (request_id, f"%{safe_query}%"),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_turns(self, request_id: str) -> int:
        """Count total turns for a request."""
        _validate_non_empty(request_id, "request_id")
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM agent_history WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return row["c"]

    def get_cost_summary(self, request_id: str) -> dict:
        """Get total cost and token usage for a request.

        Uses SUM with CASE WHEN instead of PostgreSQL FILTER (WHERE ...)
        for cross-database compatibility (SQLite doesn't support FILTER).
        """
        _validate_non_empty(request_id, "request_id")
        row = self._conn.execute(
            """SELECT
                   COUNT(*) as turns,
                   COALESCE(SUM(cost_cents), 0) as total_cost_cents,
                   COALESCE(SUM(token_count), 0) as total_tokens,
                   SUM(CASE WHEN source LIKE 'agent%' THEN 1 ELSE 0 END) as agent_turns
               FROM agent_history WHERE request_id = ?""",
            (request_id,),
        ).fetchone()
        return dict(row)
