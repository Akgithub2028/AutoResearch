import os
import json
from pathlib import Path
from typing import Dict, Any, List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

class MemoryOutput(BaseModel):
    semantic_memory: str = Field(description="Distilled factual knowledge from the draft.")
    procedural_memory: str = Field(description="Lessons learned from the execution trace.")

def build_memory_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "memory.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Summarize memory.\nQuery: {{query}}\nHistory: {{execution_history}}\nDraft: {{draft}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    
    # Mistral is preferred for summarization per SKILL.md
    llm = ChatMistralAI(model="ministral-3b-2512", temperature=0.1)
    
    # Mistral structured output
    structured_llm = llm.with_structured_output(MemoryOutput)
    return prompt | structured_llm

def memory_agent(query: str, draft: str, execution_history: List[str]) -> dict:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "your_mistral_api_key_here":
        print("[Memory Agent] WARNING: No MISTRAL_API_KEY found. Mocking memory summarization.")
        return {
            "memory": {
                "episodic_memory": execution_history,
                "semantic_memory": f"Distilled facts regarding: {query}",
                "procedural_memory": f"Graph executed {len(execution_history)} steps successfully."
            }
        }
        
    chain = build_memory_chain()
    history_str = "\n".join(execution_history)
    
    result = chain.invoke({
        "query": query, 
        "execution_history": history_str,
        "draft": draft
    })
    
    # episodic memory is just the raw execution history log
    mem_dict = result.dict()
    mem_dict["episodic_memory"] = execution_history
    
    return {"memory": mem_dict}
