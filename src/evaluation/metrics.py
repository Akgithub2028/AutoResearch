import time
from typing import Dict, Any

class EvaluationMetrics:
    def __init__(self):
        pass
        
    def finalize_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        start_time = state.get("start_time", time.time())
        wall_clock = time.time() - start_time
        history = state.get("execution_history", [])
        
        # Count LLM node executions from history
        llm_calls = sum(1 for step in history if step.startswith("[Planner]") or 
                                               step.startswith("[Grader]") or 
                                               step.startswith("[Rewrite]") or 
                                               step.startswith("[Writer]") or 
                                               step.startswith("[Reflection]") or 
                                               step.startswith("[Revision]") or 
                                               step.startswith("[Memory]"))
                                               
        retrieval_steps = sum(1 for step in history if step.startswith("[Retrieval]"))
        reflection_iterations = state.get("revision_count", 0)
        
        # Generation/Retrieval Mock Metrics (In a real system, these would be computed via LangSmith or LLM Evaluators)
        retrieval_precision = 0.85 if retrieval_steps > 0 else 1.0
        hallucination_estimate = 0.1 if reflection_iterations > 0 else 0.3
        
        return {
            "execution_metrics": {
                "llm_calls": llm_calls,
                "wall_clock_latency_seconds": round(wall_clock, 2),
                "reflection_iterations": reflection_iterations,
                "retrieval_steps": retrieval_steps
            },
            "retrieval_metrics": {
                "retrieval_precision": retrieval_precision,
            },
            "generation_metrics": {
                "hallucination_estimate": hallucination_estimate,
            }
        }
