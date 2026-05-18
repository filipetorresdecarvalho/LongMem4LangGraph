"""Tests for longmem4langgraph core components."""

import os
import json
import tempfile
import pytest
from pathlib import Path

from longmem4langgraph import SqliteSaver, HistoryStore, StateManager


@pytest.fixture
def db_path():
    """Create a temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestSqliteSaver:
    """Tests for the LangGraph checkpointer."""

    def test_init_creates_tables(self, db_path):
        saver = SqliteSaver(db_path)
        # Verify tables exist
        tables = saver._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "checkpoints" in names
        assert "checkpoint_writes" in names
        assert "checkpoint_blobs" in names

    def test_put_and_get_tuple(self, db_path):
        saver = SqliteSaver(db_path)

        # Create a checkpoint
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

        # Save
        result = saver.put(config, checkpoint, metadata)
        assert result["configurable"]["checkpoint_id"] == "test-1"

        # Read back
        tuple_result = saver.get_tuple(config)
        assert tuple_result is not None
        assert tuple_result.checkpoint["id"] == "test-1"
        assert tuple_result.metadata["status"] == "running"

    def test_multiple_threads(self, db_path):
        saver = SqliteSaver(db_path)

        # Save for two different threads
        for tid in ["thread-a", "thread-b"]:
            config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
            saver.put(config, {"id": f"{tid}-1"}, {"status": "done"})

        # Verify isolation
        ta = saver.get_tuple({"configurable": {"thread_id": "thread-a", "checkpoint_ns": ""}})
        tb = saver.get_tuple({"configurable": {"thread_id": "thread-b", "checkpoint_ns": ""}})
        assert ta.checkpoint["id"] == "thread-a-1"
        assert tb.checkpoint["id"] == "thread-b-1"

    def test_get_state_summary(self, db_path):
        saver = SqliteSaver(db_path)
        config = {"configurable": {"thread_id": "test-summary", "checkpoint_ns": ""}}
        saver.put(config, {"id": "s1"}, {"status": "running"})
        saver.put(config, {"id": "s2"}, {"status": "running"})

        summary = saver.get_state_summary("test-summary")
        assert summary["thread_id"] == "test-summary"
        assert summary["status"] == "active"
        assert summary["checkpoints"] == 2


class TestHistoryStore:
    """Tests for AGNO-style agent memory."""

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
        assert len(context) < 200  # 100 chars + tag + [truncated]
        assert "[truncated]" in context

    def test_metadata_storage(self, db_path):
        history = HistoryStore(db_path)
        meta = {"model": "deepseek-v4", "temp": 0.3, "tokens": 1500}
        history.add("req-1", "agent3", "result", metadata=meta)

        turns = history.get_history("req-1")
        stored_meta = json.loads(turns[0]["metadata"])
        assert stored_meta["model"] == "deepseek-v4"
        assert stored_meta["temp"] == 0.3


class TestStateManager:
    """Tests for the combined StateManager."""

    def test_pipeline_state_lifecycle(self, db_path):
        sm = StateManager(db_path)

        # Save
        sm.save_pipeline_state("req-1", {"status": "received", "current_node": "entry"})
        state = sm.get_pipeline_state("req-1")
        assert state["status"] == "received"

        # Update
        sm.save_pipeline_state("req-1", {"status": "processing", "current_node": "parse"})
        state = sm.get_pipeline_state("req-1")
        assert state["status"] == "processing"
        assert state["current_node"] == "parse"

    def test_node_tracking(self, db_path):
        sm = StateManager(db_path)
        import time

        node_id = sm.start_node("req-1", "parse_files")
        assert node_id > 0
        time.sleep(0.01)
        sm.finish_node(node_id, "success")

        nodes = sm.query(
            "SELECT * FROM pipeline_nodes WHERE request_id = ?",
            ("req-1",)
        )
        assert len(nodes) == 1
        assert nodes[0]["status"] == "success"
        assert nodes[0]["duration_ms"] > 0

    def test_node_tracker_context(self, db_path):
        sm = StateManager(db_path)
        import time

        with sm.track_node("req-2", "analyze"):
            time.sleep(0.01)

        nodes = sm.query(
            "SELECT * FROM pipeline_nodes WHERE request_id = ?",
            ("req-2",)
        )
        assert len(nodes) == 1
        assert nodes[0]["status"] == "success"

    def test_skill_registration(self, db_path):
        sm = StateManager(db_path)
        sm.register_skill("req-1", "n-plus-1-detection",
                         "/skills/n-plus-1.md", "deepseek",
                         ["performance", "abap"])

        skills = sm.query(
            "SELECT * FROM skills_generated WHERE request_id = ?",
            ("req-1",)
        )
        assert len(skills) == 1
        assert skills[0]["skill_name"] == "n-plus-1-detection"
        assert skills[0]["use_count"] == 0

        sm.increment_skill_use("n-plus-1-detection")
        skills = sm.query(
            "SELECT use_count FROM skills_generated WHERE skill_name = ?",
            ("n-plus-1-detection",)
        )
        assert skills[0]["use_count"] == 1

    def test_stalled_detection(self, db_path):
        sm = StateManager(db_path)
        # Mark a pipeline as processing
        sm.save_pipeline_state("req-1", {"status": "processing"})

        # With 0 minutes staleness, it should find it
        stalled = sm.find_stalled(stalled_minutes=0)
        assert len(stalled) >= 1
