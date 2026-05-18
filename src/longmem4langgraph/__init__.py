"""LongMem4LangGraph — Long-term memory for LangGraph and LangChain.

Provides persistent checkpoint saving, AGNO-style full history
for agent context, and crash recovery — backed by SQLite with
pluggable backends for any database.

Architecture:
    ABC interfaces (CheckpointStore, MemoryStore, PipelineStore, SearchBackend)
    define extension points. Users can implement these for PostgreSQL,
    MySQL, or any database. The library ships with SQLite implementations.

Key exports:
    SqliteSaver:       LangGraph BaseCheckpointSaver implementation
    HistoryStore:      AGNO-style full conversation memory
    StateManager:      Combined state, history, and audit
    recover_pipelines: Auto-detect and resume crashed pipelines
    CheckpointStore:   ABC for custom checkpoint backends
    MemoryStore:       ABC for custom memory backends
    PipelineStore:     ABC for custom pipeline backends
    SearchBackend:     ABC for pluggable search (LIKE, Weaviate, etc.)
    SqliteLikeSearch:  Zero-dependency LIKE-based search (default)
    WeaviateSearch:    Semantic vector search via Weaviate REST API
"""

from .base import CheckpointStore, MemoryStore, PipelineStore, SearchBackend
from .connection import SqliteConnection
from .saver import SqliteSaver
from .history import HistoryStore
from .state import StateManager
from .recovery import recover_pipelines
from .migration import ensure_migrated, register_migration
from .search import SqliteLikeSearch, WeaviateSearch

__version__ = "0.2.0"
__all__ = [
    "CheckpointStore",
    "MemoryStore",
    "PipelineStore",
    "SearchBackend",
    "StateManager",
    "SqliteSaver",
    "SqliteConnection",
    "HistoryStore",
    "SqliteLikeSearch",
    "WeaviateSearch",
    "recover_pipelines",
    "ensure_migrated",
    "register_migration",
]
