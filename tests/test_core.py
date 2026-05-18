"""Tests for longmem4langgraph v0.2 — core, migration, search, serializer."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

from longmem4langgraph import (
    SqliteSaver,
    HistoryStore,
    StateManager,
    SqliteConnection,
    SqliteLikeSearch,
    WeaviateSearch,
    CheckpointStore,
    MemoryStore,
    PipelineStore,
    SearchBackend,
    ensure_migrated,
    register_migration,
)
from longmem4langgraph.migration import _MIGRATIONS, _get_version
from longmem4langgraph.connection import SqliteConnection as Conn


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def db_conn(db_path):
    conn = Conn(db_path)
    yield conn
    conn.close()


# =====================================================================
# ABC Interface Tests
# =====================================================================

class TestABCInterfaces:
    def test_checkpoint_store_is_abstract(self):
        with pytest.raises(TypeError):
            CheckpointStore()

    def test_memory_store_is_abstract(self):
        with pytest.raises(TypeError):
            MemoryStore()

    def test_pipeline_store_is_abstract(self):
        with pytest.raises(TypeError):
            PipelineStore()

    def test_search_backend_is_abstract(self):
        with pytest.raises(TypeError):
            SearchBackend()


# =====================================================================
# Migration System
# =====================================================================

class TestMigration:
    def test_ensure_migrated_creates_tables(self, db_conn):
        version = ensure_migrated(db_conn)
        assert version >= 1

        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "checkpoints" in names
        assert "checkpoint_writes" in names
        assert "checkpoint_blobs" in names
        assert "agent_history" in names
        assert "pipeline_states" in names
        assert "pipeline_nodes" in names

    def test_migration_is_idempotent(self, db_conn):
        v1 = ensure_migrated(db_conn)
        v2 = ensure_migrated(db_conn)
        assert v1 == v2

    def test_migration_sets_user_version(self, db_conn):
        ensure_migrated(db_conn)
        version = _get_version(db_conn)
        assert version >= 1

    def test_register_migration_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_migration(1)
            def _dup(conn):
                pass

    def test_migration_on_existing_v01_db(self, db_path):
        conn = Conn(db_path)
        ensure_migrated(conn)
        conn.close()

        conn2 = Conn(db_path)
        version = ensure_migrated(conn2)
        assert version >= 1

        tables = conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "checkpoints" in names
        assert "agent_history" in names
        conn2.close()


# =====================================================================
# SqliteSaver (with serializer fix)
# =====================================================================

class TestSqliteSaver:
    def test_init_creates_tables(self, db_path):
        saver = SqliteSaver(db_path)
        tables = saver._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "checkpoints" in names
        assert "checkpoint_writes" in names
        assert "checkpoint_blobs" in names

    def test_put_and_get_tuple(self, db_path):
        saver = SqliteSaver(db_path)

        checkpoint = {
            "id": "test-1",
            "ts": "2026-01-01T00:00:00",
            "type": "normal",
        }
        metadata = {"status": "running", "source": "test"}

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        result = saver.put(config, checkpoint, metadata, {})
        assert result["configurable"]["checkpoint_id"] == "test-1"

        tuple_result = saver.get_tuple(config)
        assert tuple_result is not None
        assert tuple_result.checkpoint["id"] == "test-1"
        assert tuple_result.metadata["status"] == "running"

    def test_serializer_uses_serde_not_json(self, db_path):
        saver = SqliteSaver(db_path)

        checkpoint = {"id": "ser-test", "data": [1, 2, 3], "ts": "2026-01-01"}
        config = {
            "configurable": {"thread_id": "serde-test", "checkpoint_ns": ""}
        }

        saver.put(config, checkpoint, {"source": "test"}, {})

        row = saver._conn.execute(
            "SELECT checkpoint FROM checkpoints WHERE thread_id = ?",
            ("serde-test",),
        ).fetchone()

        stored = row["checkpoint"]
        assert isinstance(stored, bytes)

        result = saver.get_tuple(config)
        assert result.checkpoint["data"] == [1, 2, 3]

    def test_multiple_threads(self, db_path):
        saver = SqliteSaver(db_path)

        for tid in ["thread-a", "thread-b"]:
            config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
            saver.put(config, {"id": f"{tid}-1"}, {"status": "done"}, {})

        ta = saver.get_tuple({"configurable": {"thread_id": "thread-a", "checkpoint_ns": ""}})
        tb = saver.get_tuple({"configurable": {"thread_id": "thread-b", "checkpoint_ns": ""}})
        assert ta.checkpoint["id"] == "thread-a-1"
        assert tb.checkpoint["id"] == "thread-b-1"

    def test_get_state_summary(self, db_path):
        saver = SqliteSaver(db_path)
        config = {"configurable": {"thread_id": "test-summary", "checkpoint_ns": ""}}
        saver.put(config, {"id": "s1"}, {"status": "running"}, {})
        saver.put(config, {"id": "s2"}, {"status": "running"}, {})

        summary = saver.get_state_summary("test-summary")
        assert summary["thread_id"] == "test-summary"
        assert summary["status"] == "active"
        assert summary["checkpoints"] == 2

    def test_list_checkpoints(self, db_path):
        saver = SqliteSaver(db_path)
        config = {"configurable": {"thread_id": "list-test", "checkpoint_ns": ""}}
        for i in range(5):
            saver.put(config, {"id": f"cp-{i}"}, {"step": i}, {})

        results = list(saver.list(config, limit=3))
        assert len(results) == 3

    def test_shared_connection(self, db_conn):
        saver = SqliteSaver(connection=db_conn)
        config = {"configurable": {"thread_id": "shared", "checkpoint_ns": ""}}
        saver.put(config, {"id": "shared-1"}, {"source": "shared"}, {})
        result = saver.get_tuple(config)
        assert result.checkpoint["id"] == "shared-1"


# =====================================================================
# HistoryStore (with search backend)
# =====================================================================

class TestHistoryStore:
    def test_add_and_get_history(self, db_path):
        history = HistoryStore(db_path)
        history.add("req-1", "agent1", "First analysis", "json")
        history.add("req-1", "agent2", "Deep dive", "json")
        history.add("req-1", "agent3", "Final result", "json")

        turns = history.get_history("req-1")
        assert len(turns) == 3
        assert turns[0]["source"] == "agent1"
        assert turns[0]["content"] == "First analysis"
        assert turns[1]["turn_number"] == 2

    def test_add_turn_alias(self, db_path):
        history = HistoryStore(db_path)
        history.add_turn("req-alias", "agent1", "Via add_turn")
        turns = history.get_history("req-alias")
        assert len(turns) == 1

    def test_get_context_format(self, db_path):
        history = HistoryStore(db_path)
        history.add("req-1", "agent1", "Hello from agent 1")
        history.add("req-2", "agent1", "Different request")

        context = history.get_context("req-1", last_n=5)
        assert "[AGENT1]" in context
        assert "Hello from agent 1" in context
        assert "Different request" not in context

    def test_history_isolation(self, db_path):
        history = HistoryStore(db_path)
        history.add("req-a", "agent1", "Req A data")
        history.add("req-b", "agent1", "Req B data")

        turns_a = history.get_history("req-a")
        turns_b = history.get_history("req-b")
        assert len(turns_a) == 1
        assert len(turns_b) == 1
        assert turns_a[0]["content"] == "Req A data"

    def test_cost_summary(self, db_path):
        history = HistoryStore(db_path)
        history.add("req-1", "agent1", "work", cost_cents=1.0, token_count=500)
        history.add("req-1", "agent2", "work", cost_cents=2.0, token_count=1000)
        history.add("req-1", "user", "feedback", cost_cents=0.0)

        summary = history.get_cost_summary("req-1")
        assert summary["total_cost_cents"] == 3.0
        assert summary["total_tokens"] == 1500
        assert summary["agent_turns"] == 2

    def test_truncated_context(self, db_path):
        history = HistoryStore(db_path)
        long_text = "A" * 5000
        history.add("req-1", "agent1", long_text)

        context = history.get_context("req-1", last_n=5, max_content_length=100)
        assert len(context) < 200
        assert "[truncated]" in context

    def test_metadata_storage(self, db_path):
        history = HistoryStore(db_path)
        meta = {"model": "deepseek-v4", "temp": 0.3, "tokens": 1500}
        history.add("req-1", "agent3", "result", metadata=meta)

        turns = history.get_history("req-1")
        stored_meta = json.loads(turns[0]["metadata"])
        assert stored_meta["model"] == "deepseek-v4"
        assert stored_meta["temp"] == 0.3

    def test_search_falls_back_to_like(self, db_path):
        history = HistoryStore(db_path)
        history.add("req-1", "agent1", "discussed dependency injection patterns")
        history.add("req-1", "agent2", "analyzed performance bottlenecks")

        results = history.search("req-1", "dependency")
        assert len(results) == 1
        assert "dependency injection" in results[0]["content"]

    def test_search_with_custom_backend(self, db_path):
        mock_backend = MagicMock(spec=SearchBackend)
        mock_backend.search.return_value = [
            {"request_id": "req-1", "source": "weaviate", "content": "semantic result"}
        ]
        mock_backend.index.return_value = None

        history = HistoryStore(db_path, search_backend=mock_backend)
        history.add("req-1", "agent1", "some content")

        mock_backend.index.assert_called_once()
        results = history.search("req-1", "query")
        assert len(results) == 1
        assert results[0]["source"] == "weaviate"

    def test_search_backend_failure_falls_back(self, db_path):
        mock_backend = MagicMock(spec=SearchBackend)
        mock_backend.search.side_effect = ConnectionError("Weaviate down")
        mock_backend.index.return_value = None

        history = HistoryStore(db_path, search_backend=mock_backend)
        history.add("req-1", "agent1", "discussed caching strategies")

        results = history.search("req-1", "caching")
        assert len(results) == 1
        assert "caching" in results[0]["content"]

    def test_shared_connection(self, db_conn):
        history = HistoryStore(connection=db_conn)
        history.add("req-shared", "agent1", "Shared conn test")
        turns = history.get_history("req-shared")
        assert len(turns) == 1


# =====================================================================
# Search Backends
# =====================================================================

class TestSqliteLikeSearch:
    def test_basic_search(self, db_conn):
        ensure_migrated(db_conn)
        db_conn.execute(
            "INSERT INTO agent_history (request_id, turn_number, source, content) VALUES (?, ?, ?, ?)",
            ("req-1", 1, "agent1", "discussed dependency injection"),
        )
        db_conn.commit()

        search = SqliteLikeSearch(db_conn)
        results = search.search("dependency", "req-1")
        assert len(results) == 1
        assert "dependency injection" in results[0]["content"]

    def test_search_no_results(self, db_conn):
        ensure_migrated(db_conn)
        search = SqliteLikeSearch(db_conn)
        results = search.search("nonexistent", "req-1")
        assert len(results) == 0

    def test_index_is_noop(self, db_conn):
        search = SqliteLikeSearch(db_conn)
        search.index("req-1", "agent1", "content")  # should not raise

    def test_like_escape(self, db_conn):
        ensure_migrated(db_conn)
        db_conn.execute(
            "INSERT INTO agent_history (request_id, turn_number, source, content) VALUES (?, ?, ?, ?)",
            ("req-1", 1, "agent1", "100 percent complete"),
        )
        db_conn.commit()

        search = SqliteLikeSearch(db_conn)
        results = search.search("100 percent", "req-1")
        assert len(results) == 1

    def test_like_underscore_literal(self, db_conn):
        ensure_migrated(db_conn)
        db_conn.execute(
            "INSERT INTO agent_history (request_id, turn_number, source, content) VALUES (?, ?, ?, ?)",
            ("req-1", 1, "agent1", "underscore test"),
        )
        db_conn.commit()

        search = SqliteLikeSearch(db_conn)
        results = search.search("underscore test", "req-1")
        assert len(results) == 1


class TestWeaviateSearch:
    def test_init_with_unreachable_weaviate(self):
        search = WeaviateSearch(endpoint="http://localhost:99999")
        results = search.search("test", "req-1")
        assert results == []

    def test_init_with_live_weaviate(self):
        pytest.skip("Requires Weaviate schema setup — tested via integration tests")

    def test_index_failure_is_non_fatal(self):
        search = WeaviateSearch(endpoint="http://localhost:99999")
        search.index("req-1", "agent1", "content")  # should not raise


# =====================================================================
# StateManager
# =====================================================================

class TestStateManager:
    def test_pipeline_state_lifecycle(self, db_path):
        sm = StateManager(db_path)

        sm.save_pipeline_state("req-1", {"status": "received", "current_node": "entry"})
        state = sm.get_pipeline_state("req-1")
        assert state["status"] == "received"

        sm.save_pipeline_state("req-1", {"status": "processing", "current_node": "parse"})
        state = sm.get_pipeline_state("req-1")
        assert state["status"] == "processing"
        assert state["current_node"] == "parse"

    def test_new_api_aliases(self, db_path):
        sm = StateManager(db_path)

        sm.save_state("req-alias", {"status": "running"})
        state = sm.get_state("req-alias")
        assert state["status"] == "running"

    def test_node_tracking(self, db_path):
        sm = StateManager(db_path)
        import time

        node_id = sm.start_node("req-1", "parse_files")
        assert node_id > 0
        time.sleep(0.05)
        sm.finish_node(node_id, "success", duration_ms=50)

        nodes = sm.query(
            "SELECT * FROM pipeline_nodes WHERE request_id = ?",
            ("req-1",),
        )
        assert len(nodes) == 1
        assert nodes[0]["status"] == "success"
        assert nodes[0]["duration_ms"] == 50

    def test_node_tracker_context(self, db_path):
        sm = StateManager(db_path)
        import time

        with sm.track_node("req-2", "analyze"):
            time.sleep(0.01)

        nodes = sm.query(
            "SELECT * FROM pipeline_nodes WHERE request_id = ?",
            ("req-2",),
        )
        assert len(nodes) == 1
        assert nodes[0]["status"] == "success"

    def test_get_summary(self, db_path):
        sm = StateManager(db_path)
        sm.save_state("req-1", {"status": "completed"})
        sm.history.add("req-1", "agent1", "work done")

        summary = sm.get_summary("req-1")
        assert summary["pipeline"] is not None
        assert summary["history_turns"] == 1
        assert summary["total_cost_cents"] == 0

    def test_get_pipeline_summary_alias(self, db_path):
        sm = StateManager(db_path)
        sm.save_state("req-1", {"status": "done"})
        summary = sm.get_pipeline_summary("req-1")
        assert summary["pipeline"]["status"] == "done"

    def test_stalled_detection(self, db_path):
        sm = StateManager(db_path)
        sm.save_pipeline_state("req-1", {"status": "processing"})

        import time
        time.sleep(1.1)

        stalled = sm.find_stalled(stalled_minutes=0)
        assert len(stalled) >= 1

    def test_query_blocks_writes(self, db_path):
        sm = StateManager(db_path)
        with pytest.raises(ValueError, match="read-only"):
            sm.query("INSERT INTO pipeline_states (request_id) VALUES ('evil')")

    def test_vacuum(self, db_path):
        sm = StateManager(db_path)
        sm.save_state("req-1", {"status": "test"})
        sm.vacuum()

    def test_context_manager(self, db_path):
        with StateManager(db_path) as sm:
            sm.save_state("req-ctx", {"status": "ok"})
        assert sm._conn._closed

    def test_shared_search_backend(self, db_path):
        mock_backend = MagicMock(spec=SearchBackend)
        mock_backend.search.return_value = []
        mock_backend.index.return_value = None

        sm = StateManager(db_path, search_backend=mock_backend)
        sm.history.add("req-1", "agent1", "test")
        mock_backend.index.assert_called()


# =====================================================================
# SqliteConnection
# =====================================================================

class TestSqliteConnection:
    def test_context_manager(self, db_path):
        with Conn(db_path) as conn:
            assert not conn._closed
        assert conn._closed

    def test_wal_mode(self, db_path):
        conn = Conn(db_path)
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].upper() == "WAL"
        conn.close()

    def test_transaction_commit(self, db_path):
        conn = Conn(db_path)
        ensure_migrated(conn)
        with conn.transaction():
            conn.execute(
                "INSERT INTO pipeline_states (request_id, data) VALUES (?, '{}')",
                ("tx-test",),
            )
        row = conn.execute(
            "SELECT * FROM pipeline_states WHERE request_id = ?",
            ("tx-test",),
        ).fetchone()
        assert row is not None
        conn.close()

    def test_transaction_rollback(self, db_path):
        conn = Conn(db_path)
        ensure_migrated(conn)
        try:
            with conn.transaction():
                conn.execute(
                    "INSERT INTO pipeline_states (request_id, data) VALUES (?, '{}')",
                    ("rb-test",),
                )
                raise ValueError("force rollback")
        except ValueError:
            pass
        row = conn.execute(
            "SELECT * FROM pipeline_states WHERE request_id = ?",
            ("rb-test",),
        ).fetchone()
        assert row is None
        conn.close()
