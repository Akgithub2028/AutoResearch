# %% [markdown]
# # Phase 11: Evaluation Framework & Adaptive Compute Controller
# This notebook implements Phase 11 of the AutoResearch project.
# Goal: Introduce a Meta-Agent (Adaptive Compute Controller) to optimize budget and an Evaluation Framework to track metrics.

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
from src.agents.controller import controller_agent
from src.agents.planner import planner_agent
from src.agents.retrieval import retrieval_agent
from src.agents.grading import grading_agent
from src.agents.rewrite import rewrite_agent
from src.agents.fusion import fusion_agent
from src.agents.writer import writer_agent
from src.agents.reflection import reflection_agent
from src.agents.revision import revision_agent
from src.agents.memory import memory_agent
from src.evaluation.metrics import EvaluationMetrics

eval_framework = EvaluationMetrics()

# %% [markdown]
# ## 2. State Definition

# %%
class ResearchState(TypedDict, total=False):
    query: str
    start_time: float # NEW
    compute_budget: Dict[str, Any] # NEW
    intent: str
    plan: str
    retrieval_required: bool
    needs_web: bool
    search_queries: List[str]
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]
    retrieval_count: int # NEW
    graded_documents: List[Dict[str, Any]]
    fused_evidence: List[Dict[str, Any]]
    draft: str
    reflection: Dict[str, Any]
    revision_count: int
    confidence: float
    grader_rationale: str
    memory: Dict[str, Any]
    citations: List[str]
    execution_history: Annotated[List[str], operator.add]
    metrics: Dict[str, Any] # NEW
    
# %% [markdown]
# ## 3. Nodes

# %%
import time
def controller_node(state: ResearchState) -> dict:
    """
    NEW: Adaptive Compute Controller. Assigns budget based on query complexity.
    """
    query = state.get("query", "")
    start_time = state.get("start_time", time.time()) # Capture start time if not already provided
    print(f"[Controller Node] Analyzing query to allocate compute budget.")
    controller_output = controller_agent(query)
    budget = controller_output.get("compute_budget", {})
    print(f"[Controller Node] Budget: Reflection={budget.get('allow_reflection')}, Max Retrieval={budget.get('max_retrieval_steps')}")
    
    return {
        "start_time": start_time,
        "compute_budget": budget,
        "execution_history": [f"[Controller] Allocated Budget: {budget}"]
    }

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
    result = retrieval_agent(temp_state)
    
    count = state.get("retrieval_count", 0) + 1
    result["retrieval_count"] = count
    return result

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
    query = state.get("query", "")
    draft = state.get("draft", "")
    evidence = state.get("fused_evidence", [])
    reflection_output = reflection_agent(query, draft, evidence)
    reflection_data = reflection_output.get("reflection", {})
    satisfactory = reflection_data.get("is_satisfactory", False)
    rev_count = state.get("revision_count", 0)
    return {
        "reflection": reflection_data,
        "revision_count": rev_count,
        "execution_history": [f"[Reflection] Critique complete. Satisfactory: {satisfactory}"]
    }

def revision_node(state: ResearchState) -> dict:
    draft = state.get("draft", "")
    reflection = state.get("reflection", {})
    rev_count = state.get("revision_count", 0) + 1
    revision_output = revision_agent(draft, reflection)
    return {
        "draft": revision_output.get("draft", ""),
        "revision_count": rev_count,
        "execution_history": [f"[Revision] Applied feedback to create revision #{rev_count}."]
    }

def memory_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    draft = state.get("draft", "")
    history = state.get("execution_history", [])
    memory_output = memory_agent(query, draft, history)
    return {
        "memory": memory_output.get("memory", {}),
        "execution_history": ["[Memory] Distilled long-term memory."]
    }

def evaluation_node(state: ResearchState) -> dict:
    """
    NEW: Evaluation Framework Node. Finalizes performance metrics.
    """
    print("[Evaluation Node] Calculating final execution metrics...")
    metrics = eval_framework.finalize_metrics(state)
    return {
        "metrics": metrics,
        "execution_history": ["[Evaluation] Computed system metrics."]
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
    ret_count = state.get("retrieval_count", 0)
    budget = state.get("compute_budget", {})
    max_retries = budget.get("max_retrieval_steps", 1)
    
    if confidence >= 0.75:
        return "fusion"
    elif ret_count >= max_retries:
        print(f"[Router] -> Max retrieval loops ({max_retries}) reached based on compute budget. Forcing 'fusion'.")
        return "fusion"
    else:
        return "rewrite"

def route_after_writer(state: ResearchState) -> str:
    budget = state.get("compute_budget", {})
    allow_reflection = budget.get("allow_reflection", True)
    
    if allow_reflection:
        return "reflection"
    else:
        print("[Router] -> Compute budget disabled reflection. Routing to 'memory'.")
        return "memory"

def route_after_reflection(state: ResearchState) -> str:
    reflection = state.get("reflection", {})
    is_satisfactory = reflection.get("is_satisfactory", False)
    rev_count = state.get("revision_count", 0)
    
    if is_satisfactory:
        return "memory"
    elif rev_count >= 2:
        return "memory"
    else:
        return "revision"

# %% [markdown]
# ## 5. Graph Construction

# %%
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("controller", controller_node) # NEW
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("fusion", fusion_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("revision", revision_node)
    builder.add_node("memory", memory_node)
    builder.add_node("evaluation", evaluation_node) # NEW
    
    builder.add_edge(START, "controller")
    builder.add_edge("controller", "planner")
    builder.add_conditional_edges("planner", route_after_planner, {"retrieve": "retrieve", "writer": "writer"})
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade, {"fusion": "fusion", "rewrite": "rewrite"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("fusion", "writer")
    
    # Conditional route after writer based on budget
    builder.add_conditional_edges("writer", route_after_writer, {"reflection": "reflection", "memory": "memory"})
    
    builder.add_conditional_edges("reflection", route_after_reflection, {"revision": "revision", "memory": "memory"})
    builder.add_edge("revision", "reflection")
    
    builder.add_edge("memory", "evaluation")
    builder.add_edge("evaluation", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# %% [markdown]
# ## 6. Validation and Execution

# %%
def run_phase11_validation():
    print("--- Starting Phase 11 Validation ---")
    graph = build_graph()
    config = {"configurable": {"thread_id": "phase11_test_thread"}}
    
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
        
    print(f"\n[Generated Evaluation Metrics Preview]")
    import json
    print(json.dumps(final_state.get('metrics', {}), indent=2))
    
    print("\n--- Phase 11 Validation Complete ---")

if __name__ == "__main__":
    run_phase11_validation()

# %%
