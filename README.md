# LongMem4LangGraph — Long-term Memory for LangGraph

> SQLite-backed persistent memory for LangGraph that gives your agents AGNO-style full conversation history, crash recovery, and queryable state — without Redis, PostgreSQL, or external dependencies.

## Why?

LangGraph is great for building agent state machines, but its default memory is **in-memory only** — lost on server restart, inaccessible outside the graph, and agents only see the current state, not the full history of what came before.

**LongMem4LangGraph** fixes this with a single SQLite file:

| Feature | LangGraph Default | LongMem4LangGraph |
|---------|:-----------------:|:-----------------:|
| Persistence | ❌ Lost on restart | ✅ SQLite file |
| AGNO-style history | ❌ State only | ✅ Full conversation memory |
| Crash recovery | ❌ Manual | ✅ Auto-detect + resume |
| Queryable | ❌ Not possible | ✅ SQL queries |
| Cross-module access | ❌ Graph only | ✅ Any code can read |
| Concurrent reads | ❌ Single thread | ✅ WAL mode |
| Setup | ❌ Needs Redis/PG | ✅ pip install + ready |

## Install

```bash
pip install longmem4langgraph
```

## Quick Start

### 1. Persistent Checkpoint Saver

Replace LangGraph's default in-memory checkpointer:

```python
from langgraph.graph import StateGraph
from longmem4langgraph import SqliteSaver

# Create a graph with persistent SQLite state
builder = StateGraph(AgentState)
# ... add nodes, edges ...
graph = builder.compile(checkpointer=SqliteSaver("state.db"))

# State survives restarts — run this tomorrow and it picks up where it left off
```

### 2. AGNO-Style Agent Memory

Give each agent access to the full history of what previous agents did:

```python
from longmem4langgraph import HistoryStore

history = HistoryStore("state.db")

def agent2_node(state):
    # Read last 20 turns of agent conversation (like AGNO memory)
    context = history.get_context(state["request_id"], last_n=20)
    
    # Inject into prompt
    prompt = f"Previous analysis:\n{context}\n\nNow generate..."
    ...
```

### 3. Crash Recovery

Recover pipelines that were interrupted by a crash:

```python
from longmem4langgraph import recover_pipelines

# On system startup:
failed = await recover_pipelines("state.db", graph)
print(f"Recovered {len(failed)} interrupted pipelines")
```

### 4. Query Everything

```python
from longmem4langgraph import StateManager

sm = StateManager("state.db")

# Full pipeline audit
summary = sm.get_pipeline_summary("req-123")
print(f"Node count: {summary['total_nodes']}")
print(f"LLM calls: {summary['total_agent_calls']}")
print(f"Total cost: ${summary['total_cost']:.2f}")

# SQL query for advanced analytics
expensive = sm.query("""
    SELECT agent_name, SUM(cost_cents) as total
    FROM agent_calls GROUP BY agent_name
    ORDER BY total DESC
""")
```

## Module Reference

| Module | Class/Function | Purpose |
|--------|---------------|---------|
| `longmem4langgraph` | `StateManager` | SQLite-backed Singleton — all-in-one state, history, audit |
| `longmem4langgraph.saver` | `SqliteSaver` | LangGraph `BaseCheckpointSaver` implementation |
| `longmem4langgraph.history` | `HistoryStore` | AGNO-style full conversation memory |
| `longmem4langgraph.recovery` | `recover_pipelines()` | Auto-detect and resume crashed pipelines |

## Example: Full Agent Pipeline

```python
import asyncio
from langgraph.graph import StateGraph, END
from typing import TypedDict
from longmem4langgraph import SqliteSaver, HistoryStore

class AgentState(TypedDict):
    request_id: str
    input_text: str
    agent1_result: str
    agent2_result: str

saver = SqliteSaver("pipeline.db")
history = HistoryStore("pipeline.db")

def agent1(state: AgentState):
    result = f"Agent 1 analyzed: {state['input_text'][:50]}..."
    history.add(state["request_id"], "agent1", result)
    return {"agent1_result": result}

def agent2(state: AgentState):
    context = history.get_context(state["request_id"], last_n=5)
    result = f"Agent 2 used context: {context[:100]}..."
    history.add(state["request_id"], "agent2", result)
    return {"agent2_result": result}

builder = StateGraph(AgentState)
builder.add_node("agent1", agent1)
builder.add_node("agent2", agent2)
builder.set_entry_point("agent1")
builder.add_edge("agent1", "agent2")
builder.add_edge("agent2", END)

graph = builder.compile(checkpointer=saver)

# Run
config = {"configurable": {"thread_id": "test-1"}}
result = graph.invoke(
    {"request_id": "test-1", "input_text": "Hello agents!"},
    config
)
print(result)
```

## License

MIT — free for everyone. Use it, improve it, share it.
