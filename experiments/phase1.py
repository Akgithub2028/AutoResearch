# %% [markdown]
# # Phase 1: Core Graph
# This notebook implements Phase 1 of the AutoResearch project.
# Goal: Make a minimal LangGraph workflow.
# Nodes: Planner, Writer, END
# No retrieval. No tools.
# Verifying: State, Node transitions, Conditional routing, Streaming, Checkpointing

# %% [markdown]
# ## 1. Imports and Setup

# %%
import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# %% [markdown]
# ## 2. State Definition
# Define the minimal state for Phase 1. It must support downstream needs.

# %%
class ResearchState(TypedDict):
    """
    Minimal state representation for Phase 1.
    """
    query: str
    plan: str
    draft: str
    messages: Annotated[List[str], operator.add]
    next_node: str # Used for explicit conditional routing

# %% [markdown]
# ## 3. Nodes (Agents)
# Each node has exactly one responsibility. They do not manipulate graph topology.
# In Phase 1, we simulate LLM calls with deterministic functions to verify graph structure.

# %%
def planner_node(state: ResearchState) -> dict:
    """
    Planner Node: Analyzes the query and creates a plan.
    Simulates structured decision making.
    """
    query = state.get("query", "")
    print(f"[Planner] Analyzing query: '{query}'")
    
    plan = "1. Write introduction\n2. Write main body\n3. Write conclusion"
    messages = ["[Planner] Plan created."]
    
    # We conditionally route to the writer
    return {"plan": plan, "messages": messages, "next_node": "writer"}


def writer_node(state: ResearchState) -> dict:
    """
    Writer Node: Consumes the plan and produces a draft.
    """
    plan = state.get("plan", "")
    print(f"[Writer] Following plan:\n{plan}")
    
    draft = "This is a simulated research draft based on the plan."
    messages = ["[Writer] Draft generated."]
    
    # After writer, we end the process in Phase 1
    return {"draft": draft, "messages": messages, "next_node": "end"}

# %% [markdown]
# ## 4. Conditional Routing
# We use a router to determine where to go next based on the state.

# %%
def route_next(state: ResearchState) -> str:
    """
    Routing logic based on state.
    """
    next_node = state.get("next_node", "end")
    if next_node == "writer":
        return "writer"
    return END

# %% [markdown]
# ## 5. Graph Construction
# Construct the directed graph with checkpointing for memory.

# %%
def build_graph():
    """
    Constructs and compiles the StateGraph.
    """
    # 1. Initialize StateGraph
    builder = StateGraph(ResearchState)
    
    # 2. Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("writer", writer_node)
    
    # 3. Add edges and conditional routing
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_next, {"writer": "writer", END: END})
    builder.add_edge("writer", END)
    
    # 4. Compile with checkpointing
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

# %% [markdown]
# ## 6. Validation and Execution
# Run the graph to verify state transitions and checkpointing.

# %%
def run_phase1_validation():
    print("--- Starting Phase 1 Validation ---")
    graph = build_graph()
    
    config = {"configurable": {"thread_id": "phase1_test_thread"}}
    initial_state = {"query": "Explain LangGraph basics", "messages": []}
    
    print("\n[Execution Stream]")
    for event in graph.stream(initial_state, config=config):
        for node_name, node_state in event.items():
            print(f"--- Node: {node_name} completed ---")
            # print state updates (excluding the ever-growing messages list if it gets too long)
            print(f"Updates: {node_state}")
            
    print("\n[Final Checkpoint State]")
    final_state = graph.get_state(config).values
    print(f"Final Draft: {final_state.get('draft')}")
    print(f"Total Messages: {final_state.get('messages')}")
    print("--- Phase 1 Validation Complete ---")

if __name__ == "__main__":
    run_phase1_validation()

# %%
