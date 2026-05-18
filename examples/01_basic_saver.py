"""Example 1: Basic checkpoint persistence.

Shows how to use SqliteSaver as a drop-in replacement for
LangGraph's in-memory checkpointer. The graph state persists
across runs — even if you restart the process.
"""

import asyncio
import tempfile
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from longmem4langgraph import SqliteSaver


class CalculatorState(TypedDict):
    """Simple calculator graph state."""
    numbers: list
    sum: Optional[int]
    product: Optional[int]
    average: Optional[float]


def add_node(state: CalculatorState) -> dict:
    numbers = state.get("numbers", [])
    return {"sum": sum(numbers)}


def multiply_node(state: CalculatorState) -> dict:
    numbers = state.get("numbers", [])
    prod = 1
    for n in numbers:
        prod *= n
    return {"product": prod}


def average_node(state: CalculatorState) -> dict:
    numbers = state.get("numbers", [])
    if numbers:
        return {"average": sum(numbers) / len(numbers)}
    return {"average": 0.0}


def build_calculator(db_path: str):
    """Build a simple 3-node graph with persistent state."""

    builder = StateGraph(CalculatorState)

    builder.add_node("add", add_node)
    builder.add_node("multiply", multiply_node)
    builder.add_node("average", average_node)

    builder.set_entry_point("add")
    builder.add_edge("add", "multiply")
    builder.add_edge("multiply", "average")
    builder.add_edge("average", END)

    # Here's the magic — pass SqliteSaver as checkpointer
    graph = builder.compile(checkpointer=SqliteSaver(db_path))

    return graph


async def main():
    # Use a temp file so this example is self-contained
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"State database: {db_path}")
    print()

    graph = build_calculator(db_path)

    # Run 1 — with numbers [1, 2, 3]
    print("=== Run 1: numbers = [1, 2, 3] ===")
    config = {"configurable": {"thread_id": "calc-1"}}
    result = await graph.arun(
        {"numbers": [1, 2, 3]},
        config,
    )
    print(f"  Sum:     {result.get('sum')}")
    print(f"  Product: {result.get('product')}")
    print(f"  Average: {result.get('average')}")
    print()

    # Demonstrate persistence by creating a new graph instance
    # that reads from the same database
    print("=== Creating NEW graph instance — state survives! ===")
    graph2 = build_calculator(db_path)

    # Get state without running — it reads from SQLite
    state = graph2.get_state({"configurable": {"thread_id": "calc-1"}})
    print(f"  Recovered sum:     {state.values.get('sum')}")
    print(f"  Recovered product: {state.values.get('product')}")
    print(f"  Recovered average: {state.values.get('average')}")
    print()

    # Run 2 — a new pipeline
    print("=== Run 2: numbers = [10, 20, 30] (new thread) ===")
    config2 = {"configurable": {"thread_id": "calc-2"}}
    result2 = await graph2.arun(
        {"numbers": [10, 20, 30]},
        config2,
    )
    print(f"  Sum:     {result2.get('sum')}")
    print(f"  Product: {result2.get('product')}")
    print(f"  Average: {result2.get('average')}")

    # Cleanup
    import os
    os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(main())
