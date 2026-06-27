# %% [markdown]
# # Phase 8: Reflection
# This notebook implements Phase 8 of the AutoResearch project.
# Goal: Introduce the Reflection Agent to critique the Draft output against the initial Query and Fused Evidence.

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
from src.agents.fusion import fusion_agent
from src.agents.writer import writer_agent
from src.agents.reflection import reflection_agent

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
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]
    graded_documents: List[Dict[str, Any]]
    fused_evidence: List[Dict[str, Any]]
    draft: str
    reflection: Dict[str, Any] # Populated in this Phase
    revision: str
    confidence: float
    grader_rationale: str
    memory: Dict[str, Any]
    citations: List[str]
    execution_history: Annotated[List[str], operator.add]
    metrics: Dict[str, Any]
    
# %% [markdown]
# ## 3. Nodes

# %%
def planner_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    planner_output = planner_agent(query)
    search_queries = state.get("search_queries", [])
    if not search_queries:
        search_queries = [query]
    return {
        "intent": planner_output["intent"],
        "retrieval_required": planner_output["needs_retrieval"],
        "needs_web": planner_output.get("needs_web", False),
        "search_queries": search_queries,
        "execution_history": [f"[Planner] intent={planner_output['intent']}, retrieval={planner_output['needs_retrieval']}"]
    }

def retrieve_node(state: ResearchState) -> dict:
    search_queries = state.get("search_queries", [])
    current_query = search_queries[0] if search_queries else state.get("query", "")
    temp_state = state.copy()
    temp_state["query"] = current_query
    return retrieval_agent(temp_state)

def grade_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_documents", [])
    grade_output = grading_agent(query, docs)
    confidence = grade_output.get("overall_confidence", 0.0)
    return {
        "confidence": confidence,
        "grader_rationale": grade_output.get("rationale", ""),
        "graded_documents": grade_output.get("graded_documents", []),
        "execution_history": [f"[Grader] Scored {confidence}"]
    }

def rewrite_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    rationale = state.get("grader_rationale", "")
    rewrite_output = rewrite_agent(query, rationale)
    new_queries = rewrite_output.get("search_queries", [])
    return {
        "search_queries": new_queries,
        "execution_history": [f"[Rewrite] Generated {len(new_queries)} new queries."]
    }

def fusion_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    graded_docs = state.get("graded_documents", [])
    fusion_output = fusion_agent(query, graded_docs)
    evidence = fusion_output.get("fused_evidence", [])
    return {
        "fused_evidence": evidence,
        "execution_history": [f"[Fusion] Extracted {len(evidence)} claims."]
    }

def writer_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    intent = state.get("intent", "general")
    evidence = state.get("fused_evidence", [])
    writer_output = writer_agent(query, intent, evidence)
    return {
        "draft": writer_output.get("draft", ""),
        "execution_history": [f"[Writer] Draft complete."]
    }

def reflection_node(state: ResearchState) -> dict:
    """
    NEW: Reflection Node. Consumes Draft and Evidence to critique the output.
    """
    query = state.get("query", "")
    draft = state.get("draft", "")
    evidence = state.get("fused_evidence", [])
    
    print(f"[Reflection Node] Critiquing draft...")
    reflection_output = reflection_agent(query, draft, evidence)
    reflection_data = reflection_output.get("reflection", {})
    
    satisfactory = reflection_data.get("is_satisfactory", False)
    print(f"[Reflection Node] Satisfactory: {satisfactory}")
    
    return {
        "reflection": reflection_data,
        "execution_history": [f"[Reflection] Critique complete. Satisfactory: {satisfactory}"]
    }

# %% [markdown]
# ## 4. Conditional Routing

# %%
def route_after_planner(state: ResearchState) -> str:
    if state.get("retrieval_required"):
        return "retrieve"
    return "writer"

def route_after_grade(state: ResearchState) -> str:
    if state.get("confidence", 0.0) >= 0.75:
        return "fusion"
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
    builder.add_node("fusion", fusion_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reflection", reflection_node) # NEW
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner, {"retrieve": "retrieve", "writer": "writer"})
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade, {"fusion": "fusion", "rewrite": "rewrite"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("fusion", "writer")
    
    # In Phase 8, writer outputs to reflection, which ends the graph. (Phase 9 will loop back).
    builder.add_edge("writer", "reflection")
    builder.add_edge("reflection", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# %% [markdown]
# ## 6. Validation and Execution

# %%
def run_phase8_validation():
    print("--- Starting Phase 8 Validation ---")
    graph = build_graph()
    config = {"configurable": {"thread_id": "phase8_test_thread"}}
    
    initial_state = {
        "query": "Quantum Entanglement anomalies", 
        "execution_history": []
    }
    
    print("\n[Execution Stream]")
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            print(f"--- Node: {node_name} completed ---")
            
    print("\n[Final Checkpoint State]")
    final_state = graph.get_state(config).values
    
    print("Execution History:")
    for step in final_state.get('execution_history', []):
        print(f"  - {step}")
        
    print("\n[Reflection Critique Preview]")
    reflection = final_state.get('reflection', {})
    print(f"Critique: {reflection.get('critique')}")
    print(f"Unsupported Claims: {reflection.get('unsupported_claims')}")
    print(f"Missing Citations: {reflection.get('missing_citations')}")
    print(f"Hallucination Risk: {reflection.get('hallucination_risk')}")
    print(f"Is Satisfactory: {reflection.get('is_satisfactory')}")
    
    print("\n--- Phase 8 Validation Complete ---")

if __name__ == "__main__":
    run_phase8_validation()

# %%
