# %% [markdown]
# # Phase 2: State Management
# This notebook implements Phase 2 of the AutoResearch project.
# Goal: Define the complete `ResearchState` that will act as the backbone of the project.
# We will upgrade the minimal graph from Phase 1 to utilize and validate this new state structure.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# %% [markdown]
# ## 2. State Definition
# Defines the comprehensive state structure required for all downstream nodes.
# `total=False` is used so that we don't have to initialize every field at the START node.
# We use `Annotated[List[str], operator.add]` for `execution_history` to ensure it appends across node executions.

# %%
class ResearchState(TypedDict, total=False):
    """
    Comprehensive state representation for the AutoResearch system.
    Follows SKILL.md rules: Typed, minimal necessary fields, avoids raw LLM blobs if possible.
    """
    query: str
    intent: str
    plan: str
    retrieval_required: bool
    search_queries: List[str]
    retrieved_documents: List[Dict[str, Any]]
    graded_documents: List[Dict[str, Any]]
    draft: str
    reflection: Dict[str, Any]
    revision: str
    confidence: float
    memory: Dict[str, Any]
    citations: List[str]
    execution_history: Annotated[List[str], operator.add]
    metrics: Dict[str, Any]
    
    # Internal routing control
    next_node: str 

# %% [markdown]
# ## 3. Nodes (Agents)
# We update the mock nodes from Phase 1 to populate the new fields in `ResearchState`.
# This validates that our graph can handle complex typing and field updates.

# %%
def planner_node(state: ResearchState) -> dict:
    """
    Planner Node: Simulates query analysis, intent classification, and planning.
    Now populates more comprehensive state fields.
    """
    query = state.get("query", "")
    print(f"[Planner] Analyzing query: '{query}'")
    
    # Simulating Planner structured output
    return {
        "intent": "technical_explanation",
        "plan": "1. Define LangGraph\n2. Explain State\n3. Summarize routing",
        "retrieval_required": False,  # For Phase 2, we still mock execution without tools
        "confidence": 0.95,
        "execution_history": ["[Planner] Intent classified and plan generated."],
        "metrics": {"planner_tokens": 150},
        "next_node": "writer"
    }


def writer_node(state: ResearchState) -> dict:
    """
    Writer Node: Consumes the plan and intent to produce a draft.
    Updates the execution history and draft state.
    """
    plan = state.get("plan", "")
    intent = state.get("intent", "general")
    print(f"[Writer] Following plan for intent '{intent}':\n{plan}")
    
    draft = "LangGraph is a library for building stateful, multi-actor applications with LLMs..."
    
    return {
        "draft": draft,
        "execution_history": ["[Writer] Draft generated based on plan."],
        "metrics": {"writer_tokens": 300}, # Note: in a real app, you might want to merge dictionaries or use a reducer for metrics
        "next_node": "end"
    }

# %% [markdown]
# ## 4. Conditional Routing
# Identical to Phase 1: simple conditional routing to verify state transitions.

# %%
def route_next(state: ResearchState) -> str:
    """
    Routing logic based on state's next_node field.
    """
    next_node = state.get("next_node", "end")
    if next_node == "writer":
        return "writer"
    return END

# %% [markdown]
# ## 5. Graph Construction

# %%
def build_graph():
    """
    Constructs and compiles the StateGraph using the new ResearchState.
    """
    builder = StateGraph(ResearchState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("writer", writer_node)
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_next, {"writer": "writer", END: END})
    builder.add_edge("writer", END)
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

# %% [markdown]
# ## 6. Validation and Execution
# Run the graph to verify that the comprehensive state object behaves correctly.

# %%
def run_phase2_validation():
    print("--- Starting Phase 2 Validation ---")
    graph = build_graph()
    
    config = {"configurable": {"thread_id": "phase2_test_thread"}}
    
    # Initialize with minimal state
    initial_state = {
        "query": "Explain LangGraph state management", 
        "execution_history": []
    }
    
    print("\n[Execution Stream]")
    for event in graph.stream(initial_state, config=config):
        for node_name, node_state in event.items():
            print(f"--- Node: {node_name} completed ---")
            
            # Print specific fields to verify State modifications
            print(f"Draft Length: {len(node_state.get('draft', ''))}")
            print(f"Intent: {node_state.get('intent')}")
            print(f"Execution History: {node_state.get('execution_history')}")
            
    print("\n[Final Checkpoint State]")
    final_state = graph.get_state(config).values
    
    # Verify the state holds everything we expect
    print(f"Query: {final_state.get('query')}")
    print(f"Intent: {final_state.get('intent')}")
    print(f"Retrieval Required: {final_state.get('retrieval_required')}")
    print(f"Confidence: {final_state.get('confidence')}")
    print(f"Final Execution History: {final_state.get('execution_history')}")
    print("--- Phase 2 Validation Complete ---")

if __name__ == "__main__":
    run_phase2_validation()

# %%
