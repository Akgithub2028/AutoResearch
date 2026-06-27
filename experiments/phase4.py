# %% [markdown]
# # Phase 4: Adaptive Retrieval
# This notebook implements Phase 4 of the AutoResearch project.
# Goal: Adaptive RAG. Implement conditional routing based on the Planner's `needs_retrieval` flag.
# If true, the graph routes to the Retrieval Agent, which fetches data from Wikipedia or ArXiv.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.agents.planner import planner_agent
from src.agents.retrieval import retrieval_agent

# %% [markdown]
# ## 2. State Definition

# %%
class ResearchState(TypedDict, total=False):
    query: str
    intent: str
    plan: str
    retrieval_required: bool
    needs_web: bool
    search_queries: List[str]
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add] # Reducer added for documents
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

# %%
def planner_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    print(f"[Planner Node] Invoking LLM Planner for query: '{query}'")
    
    planner_output = planner_agent(query)
    
    # We let Planner dictate if we need web search
    needs_web = planner_output.get("needs_web", False)
    
    history_msg = f"[Planner] intent={planner_output['intent']}, retrieval={planner_output['needs_retrieval']}, web={needs_web}"
    
    return {
        "intent": planner_output["intent"],
        "retrieval_required": planner_output["needs_retrieval"],
        "needs_web": needs_web,
        "execution_history": [history_msg]
    }

def retrieve_node(state: ResearchState) -> dict:
    """
    Retrieval Node: Executes adaptive search tools.
    """
    result = retrieval_agent(state)
    return result

def writer_node(state: ResearchState) -> dict:
    intent = state.get("intent", "general")
    docs = state.get("retrieved_documents", [])
    doc_count = len(docs)
    
    print(f"[Writer Node] Writing based on intent '{intent}' using {doc_count} source documents.")
    
    return {
        "draft": f"Simulated draft utilizing {doc_count} external sources.",
        "execution_history": [f"[Writer] Generated draft using {doc_count} documents."]
    }

# %% [markdown]
# ## 4. Conditional Routing
# This is the core of Adaptive RAG. The router inspects `retrieval_required`.

# %%
def route_after_planner(state: ResearchState) -> str:
    """
    Determines whether to route to Retrieval or straight to Writer.
    """
    if state.get("retrieval_required"):
        print("[Router] -> Retrieval Required. Routing to 'retrieve'")
        return "retrieve"
    else:
        print("[Router] -> No Retrieval Needed. Routing to 'writer'")
        return "writer"

# %% [markdown]
# ## 5. Graph Construction

# %%
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("writer", writer_node)
    
    builder.add_edge(START, "planner")
    
    # Adaptive conditional routing
    builder.add_conditional_edges("planner", route_after_planner, {"retrieve": "retrieve", "writer": "writer"})
    
    builder.add_edge("retrieve", "writer")
    builder.add_edge("writer", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# %% [markdown]
# ## 6. Validation and Execution

# %%
def run_phase4_validation():
    print("--- Starting Phase 4 Validation ---")
    graph = build_graph()
    
    config = {"configurable": {"thread_id": "phase4_test_thread"}}
    
    # We'll test with a query that naturally requires academic retrieval
    initial_state = {
        "query": "Quantum Entanglement", 
        "execution_history": []
    }
    
    print("\n[Execution Stream]")
    for event in graph.stream(initial_state, config=config):
        for node_name, node_state in event.items():
            print(f"--- Node: {node_name} completed ---")
            
    print("\n[Final Checkpoint State]")
    final_state = graph.get_state(config).values
    
    docs = final_state.get('retrieved_documents', [])
    print(f"Retrieval Required: {final_state.get('retrieval_required')}")
    print(f"Needs Web: {final_state.get('needs_web')}")
    print(f"Documents Retrieved: {len(docs)}")
    if len(docs) > 0:
        print(f"Sample Source: {docs[0]['source']}")
        print(f"Content Preview: {docs[0]['content'][:100]}...")
    
    print(f"Execution History: {final_state.get('execution_history')}")
    print("--- Phase 4 Validation Complete ---")

if __name__ == "__main__":
    run_phase4_validation()

# %%
