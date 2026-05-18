"""LongMem4LangGraph — Long-term SQLite-backed memory for LangGraph.

Provides persistent checkpoint saving, AGNO-style full history
for agent context, and crash recovery — all backed by a single
SQLite file. No external databases required.

Key exports:
    StateManager: Core Singleton — state, history, audit, queries
    SqliteSaver: LangGraph BaseCheckpointSaver implementation
    HistoryStore: AGNO-style full conversation memory
    recover_pipelines: Auto-detect and resume crashed pipelines
"""

from .connection import SqliteConnection
from .saver import SqliteSaver
from .history import HistoryStore
from .state import StateManager
from .recovery import recover_pipelines

__version__ = "0.1.0"
__all__ = [
    "StateManager",
    "SqliteSaver",
    "HistoryStore",
    "SqliteConnection",
    "recover_pipelines",
]
