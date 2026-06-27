# %% [markdown]
# # Phase 3: Planner Agent
# This notebook implements Phase 3 of the AutoResearch project.
# Goal: Integrate a real LangChain Planner Agent into our state graph.
# The planner reads its prompt from `src/prompts/planner.md` and uses Gemini via structured outputs.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our new Planner agent logic
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.agents.planner import planner_agent

# %% [markdown]
# ## 2. State Definition (From Phase 2)

# %%
class ResearchState(TypedDict, total=False):
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
    next_node: str 

# %% [markdown]
# ## 3. Nodes
# Update the graph `planner_node` to invoke the real LangChain agent.

# %%
def planner_node(state: ResearchState) -> dict:
    """
    Planner Node: Invokes the LangChain Planner Agent.
    """
    query = state.get("query", "")
    print(f"[Planner Node] Invoking LLM Planner for query: '{query}'")
    
    # Execute the agent
    planner_output = planner_agent(query)
    
    # Construct state updates based on structured output
    # Note: 'plan' in Phase 1 was a string describing steps. The planner output doesn't have 'plan',
    # but it outputs the structured orchestration parameters.
    # We will serialize it to the execution history.
    history_msg = f"[Planner] Output: intent={planner_output['intent']}, retrieval_required={planner_output['needs_retrieval']}"
    
    return {
        "intent": planner_output["intent"],
        "retrieval_required": planner_output["needs_retrieval"],
        "execution_history": [history_msg],
        "next_node": "writer"
    }

def writer_node(state: ResearchState) -> dict:
    """
    Writer Node: Mock writer for validation.
    """
    intent = state.get("intent", "general")
    print(f"[Writer Node] Writing based on intent '{intent}'.")
    
    return {
        "draft": f"Simulated draft for intent: {intent}",
        "execution_history": ["[Writer] Generated draft."],
        "next_node": "end"
    }

# %% [markdown]
# ## 4. Conditional Routing

# %%
def route_next(state: ResearchState) -> str:
    next_node = state.get("next_node", "end")
    if next_node == "writer":
        return "writer"
    return END

# %% [markdown]
# ## 5. Graph Construction

# %%
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("writer", writer_node)
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_next, {"writer": "writer", END: END})
    builder.add_edge("writer", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# %% [markdown]
# ## 6. Validation and Execution

# %%
def run_phase3_validation():
    print("--- Starting Phase 3 Validation ---")
    graph = build_graph()
    
    config = {"configurable": {"thread_id": "phase3_test_thread"}}
    
    initial_state = {
        "query": "How does Quantum Entanglement work?", 
        "execution_history": []
    }
    
    print("\n[Execution Stream]")
    for event in graph.stream(initial_state, config=config):
        for node_name, node_state in event.items():
            print(f"--- Node: {node_name} completed ---")
            
    print("\n[Final Checkpoint State]")
    final_state = graph.get_state(config).values
    
    print(f"Query: {final_state.get('query')}")
    print(f"Intent: {final_state.get('intent')}")
    print(f"Retrieval Required: {final_state.get('retrieval_required')}")
    print(f"Execution History: {final_state.get('execution_history')}")
    print("--- Phase 3 Validation Complete ---")

if __name__ == "__main__":
    run_phase3_validation()

# %%
