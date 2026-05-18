"""Abstract base classes for LongMem4LangGraph stores.

These ABCs define the interfaces that any database backend must implement.
Users can create custom backends (PostgreSQL, MySQL, etc.) by implementing
these classes. The library ships with SQLite implementations.

Usage:
    from longmem4langgraph.base import MemoryStore, SearchBackend

    class PostgresMemoryStore(MemoryStore):
        def add_turn(self, request_id, source, content, **kwargs):
            ...
        def get_history(self, request_id, **kwargs):
            ...
        def get_context(self, request_id, **kwargs):
            ...
        def search(self, request_id, query, **kwargs):
            ...
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional


class CheckpointStore(ABC):
    """Abstract interface for LangGraph checkpoint persistence.

    Implement this to support checkpoint saving on any database backend.
    The SQLite implementation (SqliteSaver) is provided out of the box.

    Methods match LangGraph's BaseCheckpointSaver interface so that
    any CheckpointStore can be used as a LangGraph checkpointer.
    """

    @abstractmethod
    def get_tuple(self, config: dict) -> Optional[Any]:
        """Get the latest checkpoint for a thread."""
        ...

    @abstractmethod
    def put(self, config: dict, checkpoint: Any, metadata: Any) -> dict:
        """Save a checkpoint. Returns the config with checkpoint_id."""
        ...

    @abstractmethod
    def list(
        self,
        config: dict,
        *,
        filter: Optional[dict] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Any]:
        """List checkpoints for a thread, optionally filtered."""
        ...


class MemoryStore(ABC):
    """Abstract interface for agent conversation memory (AGNO-style).

    Implement this to support agent history on any database backend.
    The SQLite implementation (HistoryStore) is provided out of the box.

    Gives agents access to the full history of what previous agents said,
    not just the current state — similar to AGNO's session.memory.
    """

    @abstractmethod
    def add_turn(self, request_id: str, source: str, content: str, **kwargs) -> int:
        """Add a conversation turn. Returns the turn number (1-indexed)."""
        ...

    @abstractmethod
    def get_history(
        self,
        request_id: str,
        last_n: Optional[int] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        """Get conversation history as a list of dicts, oldest first."""
        ...

    @abstractmethod
    def get_context(
        self,
        request_id: str,
        last_n: int = 10,
        max_content_length: int = 2000,
        format_template: Optional[str] = None,
    ) -> str:
        """Get history formatted as a string for LLM prompt injection."""
        ...

    @abstractmethod
    def search(self, request_id: str, query: str, limit: int = 10) -> List[dict]:
        """Search conversation history. Delegates to SearchBackend if configured."""
        ...


class PipelineStore(ABC):
    """Abstract interface for pipeline state management.

    Implement this to support pipeline tracking on any database backend.
    The SQLite implementation (StateManager) is provided out of the box.
    """

    @abstractmethod
    def save_state(self, request_id: str, state: dict) -> dict:
        """Save or update pipeline state. Returns the merged state dict."""
        ...

    @abstractmethod
    def get_state(self, request_id: str) -> Optional[dict]:
        """Get current pipeline state, or None if not found."""
        ...

    @abstractmethod
    def get_summary(self, request_id: str) -> dict:
        """Get complete pipeline summary (nodes, history, costs)."""
        ...


class SearchBackend(ABC):
    """Abstract interface for search backends.

    Implement this to support different search strategies:
    - SqliteLikeSearch: zero-dependency LIKE-based search (default)
    - WeaviateSearch: semantic vector search (optional)
    - Custom: any backend that implements search() and index()

    Usage:
        history = HistoryStore(db_path, search_backend=WeaviateSearch())
    """

    @abstractmethod
    def search(self, query: str, request_id: str, limit: int = 10) -> List[dict]:
        """Search for content matching the query within a request."""
        ...

    @abstractmethod
    def index(self, request_id: str, source: str, content: str, **kwargs) -> None:
        """Index a document for future searches. Non-fatal on failure."""
        ...
