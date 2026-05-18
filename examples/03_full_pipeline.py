"""Example 3: Full StateManager — persistence + history + recovery.

Shows the complete StateManager API with pipeline tracking,
cost logging, skills registration, and crash recovery.
"""

import asyncio
import tempfile
import time
from langgraph.graph import StateGraph, END
from typing import TypedDict

from longmem4langgraph import StateManager


class PipelineState(TypedDict):
    request_id: str
    status: str


def build_pipeline(db_path: str):
    """Build a pipeline with full StateManager integration."""
    sm = StateManager(db_path)

    def parse_files(state: PipelineState):
        rid = state["request_id"]
        sm.save_pipeline_state(rid, {"status": "parsing", "current_node": "parse"})

        with sm.track_node(rid, "parse_files"):
            time.sleep(0.1)  # Simulate work
            sm.history.add(rid, "agent1", "Parsed 15 ABAP files", "summary")

        sm.save_pipeline_state(rid, {"status": "analyzing", "current_node": "analyze"})
        return {"status": "analyzing"}

    def analyze_code(state: PipelineState):
        rid = state["request_id"]

        with sm.track_node(rid, "analyze_code"):
            time.sleep(0.1)
            sm.history.add(rid, "agent2", "Analyzed code patterns", "json",
                          metadata={"patterns": 7, "issues": 2})

        sm.save_pipeline_state(rid, {"status": "generating", "current_node": "generate"})
        return {"status": "generating"}

    def generate_docs(state: PipelineState):
        rid = state["request_id"]

        with sm.track_node(rid, "generate_docs"):
            time.sleep(0.1)
            sm.history.add(rid, "agent3", "Generated final documentation", "json",
                          cost_cents=1.5, token_count=4500)

        sm.save_pipeline_state(rid, {"status": "completed", "current_node": END})
        return {"status": "completed"}

    def validate_docs(state: PipelineState):
        rid = state["request_id"]

        with sm.track_node(rid, "validate_docs"):
            time.sleep(0.05)
            sm.history.add(rid, "agent4", "Validation passed", "decision")

        return {"status": "validated"}

    # Build graph
    builder = StateGraph(PipelineState)
    builder.add_node("parse", parse_files)
    builder.add_node("analyze", analyze_code)
    builder.add_node("generate", generate_docs)
    builder.add_node("validate", validate_docs)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "analyze")
    builder.add_edge("analyze", "generate")
    builder.add_edge("generate", "validate")
    builder.add_edge("validate", END)

    graph = builder.compile(checkpointer=sm.checkpointer)

    return graph, sm


async def main():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"State database: {db_path}\n")

    graph, sm = build_pipeline(db_path)

    # Run pipeline
    rid = "req-full-001"
    config = {"configurable": {"thread_id": rid}}
    result = await graph.arun(
        {"request_id": rid, "status": "received"},
        config,
    )

    print("=== Pipeline Complete ===")
    print(f"Final status: {result['status']}")
    print()

    # Get full summary
    print("=== Pipeline Summary ===")
    summary = sm.get_pipeline_summary(rid)
    print(f"Nodes executed: {len(summary['nodes'])}")
    print(f"History turns:  {summary['history_turns']}")
    print(f"Total cost:     ${summary['total_cost_cents']/100:.2f}")
    print()

    # Show nodes
    print("=== Node Timeline ===")
    for node in summary["nodes"]:
        dur = node.get("duration_ms", 0)
        status_icon = "✅" if node["status"] == "success" else "❌"
        print(f"  {status_icon} {node['node_name']}: {dur}ms ({node['status']})")

    # Register a skill
    sm.register_skill(rid, "detect-n-plus-1",
                     "/skills/generated/detect-n-plus-1.md",
                     "deepseek", ["performance", "abap"])

    # Query skills
    print("\n=== Skills ===")
    skills = sm.query(
        "SELECT skill_name, source, tags FROM skills_generated WHERE request_id = ?",
        (rid,)
    )
    for s in skills:
        print(f"  📚 {s['skill_name']} (from {s['source']})")

    # Check stalled pipelines (should be none)
    stalled = sm.find_stalled(stalled_minutes=0)
    print(f"\n=== Stalled pipelines: {len(stalled)} ===")

    # Mark complete
    sm.mark_completed(rid)
    print("Pipeline marked as completed.")

    # Cleanup
    import os
    os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(main())
