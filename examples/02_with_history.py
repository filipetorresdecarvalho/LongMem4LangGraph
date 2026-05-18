"""Example 2: AGNO-style agent memory with HistoryStore.

Shows how to give each agent access to the full conversation
history of what previous agents said — like AGNO's session.memory,
but persistent and queryable.
"""

import asyncio
import tempfile
from langgraph.graph import StateGraph, END
from typing import TypedDict

from longmem4langgraph import SqliteSaver, HistoryStore


class AgentState(TypedDict):
    request_id: str
    user_query: str
    agent1_result: str
    agent2_result: str
    final_result: str


def build_analysis_pipeline(db_path: str):
    """Pipeline where Agent 2 can read what Agent 1 wrote."""

    history = HistoryStore(db_path)

    def agent1_explorer(state: AgentState):
        """Agent 1: Basic exploration."""
        rid = state["request_id"]
        query = state["user_query"]

        result = f"Agent 1 analyzed the request: '{query}'\n"
        result += f"- Found 3 key concepts\n"
        result += f"- Identified 2 related topics\n"
        result += f"- Preliminary classification: technical"

        # Write to history — Agent 2 will read this
        history.add(rid, "agent1", result, "json")

        return {"agent1_result": result}

    def agent2_deep_dive(state: AgentState):
        """Agent 2: Deep analysis — reads Agent 1's output from memory."""
        rid = state["request_id"]

        # Read what Agent 1 wrote (last 5 turns)
        context = history.get_context(rid, last_n=5)

        # Agent 2 uses the history to build on Agent 1's work
        result = f"Agent 2 deep dive:\n\n"
        result += f"Based on previous analysis:\n{context[:300]}...\n\n"
        result += f"Deep analysis results:\n"
        result += f"- Confirmed the classification\n"
        result += f"- Found 2 additional patterns\n"
        result += f"- Generated detailed report"

        history.add(rid, "agent2", result, "json")

        return {"agent2_result": result}

    def agent3_summarizer(state: AgentState):
        """Agent 3: Reads ALL history and produces final result."""
        rid = state["request_id"]

        # Read everything
        all_context = history.get_context(rid, last_n=20)

        result = f"Final summary:\n\n"
        result += f"Full pipeline history:\n{all_context[:500]}...\n\n"
        result += f"CONCLUSION: Comprehensive analysis complete.\n"
        result += f"Documentation ready for delivery."

        history.add(rid, "agent3", result, "json")

        return {"final_result": result}

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("explore", agent1_explorer)
    builder.add_node("deep_dive", agent2_deep_dive)
    builder.add_node("summarize", agent3_summarizer)

    builder.set_entry_point("explore")
    builder.add_edge("explore", "deep_dive")
    builder.add_edge("deep_dive", "summarize")
    builder.add_edge("summarize", END)

    graph = builder.compile(checkpointer=SqliteSaver(db_path))

    return graph, history


async def main():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"State database: {db_path}\n")

    graph, history = build_analysis_pipeline(db_path)

    # Run
    config = {"configurable": {"thread_id": "analysis-1"}}
    result = await graph.arun(
        {
            "request_id": "req-001",
            "user_query": "Analyze this ABAP Z-FI document program",
        },
        config,
    )

    print("=== Pipeline Results ===")
    print(f"Agent 1: {result['agent1_result'][:80]}...")
    print(f"Agent 2: {result['agent2_result'][:80]}...")
    print(f"Final:   {result['final_result'][:80]}...")
    print()

    # Query the history
    print("=== History Summary ===")
    turns = history.count_turns("req-001")
    cost = history.get_cost_summary("req-001")
    print(f"Total turns: {turns}")
    print(f"Agent turns: {cost['agent_turns']}")
    print()

    # Show full history
    print("=== Full History ===")
    all_history = history.get_history("req-001")
    for h in all_history:
        print(f"  [{h['source'].upper()}] Turn {h['turn_number']}: {h['content'][:60]}...")

    # Cleanup
    import os
    os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(main())
