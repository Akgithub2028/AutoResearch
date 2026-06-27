# %% [markdown]
# # Phase 5: CRAG Layer (Corrective Retrieval Augmented Generation)
# This notebook implements Phase 5 of the AutoResearch project.
# Goal: Evaluate retrieved documents, assign confidence scores, and conditionally route to a Query Rewrite agent if confidence is too low.

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
from src.agents.grading import grading_agent
from src.agents.rewrite import rewrite_agent

# %% [markdown]
# ## 2. State Definition

# %%
class ResearchState(TypedDict, total=False):
    query: str
    intent: str
    plan: str
    retrieval_required: bool
    needs_web: bool
    search_queries: List[str] # Now used by rewrite agent
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]
    graded_documents: List[Dict[str, Any]]
    draft: str
    reflection: Dict[str, Any]
    revision: str
    confidence: float
    grader_rationale: str # Internal state to pass rationale to rewrite
    memory: Dict[str, Any]
    citations: List[str]
    execution_history: Annotated[List[str], operator.add]
    metrics: Dict[str, Any]
    
# %% [markdown]
# ## 3. Nodes

# %%
def planner_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    print(f"[Planner Node] Invoking LLM Planner for query: '{query}'")
    
    planner_output = planner_agent(query)
    needs_web = planner_output.get("needs_web", False)
    
    # Initialize search_queries with the original query if empty
    search_queries = state.get("search_queries", [])
    if not search_queries:
        search_queries = [query]
        
    history_msg = f"[Planner] intent={planner_output['intent']}, retrieval={planner_output['needs_retrieval']}"
    
    return {
        "intent": planner_output["intent"],
        "retrieval_required": planner_output["needs_retrieval"],
        "needs_web": needs_web,
        "search_queries": search_queries,
        "execution_history": [history_msg]
    }

def retrieve_node(state: ResearchState) -> dict:
    # Use the first search query from the list
    search_queries = state.get("search_queries", [])
    current_query = search_queries[0] if search_queries else state.get("query", "")
    
    # Temporarily override state query for the retrieval agent
    temp_state = state.copy()
    temp_state["query"] = current_query
    
    result = retrieval_agent(temp_state)
    return result

def grade_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_documents", [])
    print(f"[Grade Node] Evaluating {len(docs)} documents against query.")
    
    grade_output = grading_agent(query, docs)
    confidence = grade_output.get("overall_confidence", 0.0)
    rationale = grade_output.get("rationale", "")
    
    print(f"[Grade Node] Confidence: {confidence} | Rationale: {rationale}")
    
    return {
        "confidence": confidence,
        "grader_rationale": rationale,
        "graded_documents": grade_output.get("graded_documents", []),
        "execution_history": [f"[Grader] Scored {confidence}: {rationale}"]
    }

def rewrite_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    rationale = state.get("grader_rationale", "")
    print(f"[Rewrite Node] Rewriting query due to low confidence...")
    
    rewrite_output = rewrite_agent(query, rationale)
    new_queries = rewrite_output.get("search_queries", [])
    print(f"[Rewrite Node] New queries generated: {new_queries}")
    
    return {
        "search_queries": new_queries,
        # We optionally clear retrieved documents to start fresh, 
        # but since we use operator.add, returning empty list prevents clearing it.
        # For simplicity, we just keep accumulating docs.
        "execution_history": [f"[Rewrite] Generated {len(new_queries)} new queries."]
    }

def writer_node(state: ResearchState) -> dict:
    intent = state.get("intent", "general")
    docs = state.get("graded_documents", [])
    doc_count = len(docs)
    print(f"[Writer Node] Writing draft using {doc_count} graded documents.")
    return {
        "draft": f"Simulated draft based on {doc_count} validated documents.",
        "execution_history": [f"[Writer] Draft complete."]
    }

# %% [markdown]
# ## 4. Conditional Routing

# %%
def route_after_planner(state: ResearchState) -> str:
    if state.get("retrieval_required"):
        return "retrieve"
    return "writer"

def route_after_grade(state: ResearchState) -> str:
    confidence = state.get("confidence", 0.0)
    if confidence >= 0.75:
        print("[Router] -> Confidence High (>=0.75). Routing to 'writer'.")
        return "writer"
    else:
        print("[Router] -> Confidence Low (<0.75). Routing to 'rewrite'.")
        return "rewrite"

# %% [markdown]
# ## 5. Graph Construction

# %%
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("writer", writer_node)
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner, {"retrieve": "retrieve", "writer": "writer"})
    
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade, {"writer": "writer", "rewrite": "rewrite"})
    
    builder.add_edge("rewrite", "retrieve") # Loop back
    builder.add_edge("writer", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# %% [markdown]
# ## 6. Validation and Execution

# %%
def run_phase5_validation():
    print("--- Starting Phase 5 Validation ---")
    graph = build_graph()
    config = {"configurable": {"thread_id": "phase5_test_thread"}}
    
    # We use a query that will intentionally trigger 0 documents in our mock retrieval 
    # (since arxiv network call fails or wiki fails) which will trigger the grader mock's low confidence.
    initial_state = {
        "query": "Quantum Entanglement anomalies", 
        "execution_history": []
    }
    
    print("\n[Execution Stream]")
    
    # Note: We need recursion limit because loops could be infinite if mocked poorly. 
    # LangGraph defaults to 25.
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            print(f"--- Node: {node_name} completed ---")
            
    print("\n[Final Checkpoint State]")
    final_state = graph.get_state(config).values
    
    print(f"Final Confidence: {final_state.get('confidence')}")
    print(f"Search Queries generated: {final_state.get('search_queries')}")
    print("Execution History:")
    for step in final_state.get('execution_history', []):
        print(f"  - {step}")
        
    print("--- Phase 5 Validation Complete ---")

if __name__ == "__main__":
    run_phase5_validation()

# %%
