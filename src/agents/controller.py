import os
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class ControllerOutput(BaseModel):
    allow_reflection: bool = Field(description="Whether to permit the expensive reflection phase.")
    max_retrieval_steps: int = Field(description="Maximum allowed retrieval loops (1-3).")
    model_preference: str = Field(description="Either 'gemini' or 'mistral'.")
    rationale: str = Field(description="Brief explanation of the allocation.")

def build_controller_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "controller.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Allocate budget.\nQuery: {{query}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(ControllerOutput)
    return prompt | structured_llm

def controller_agent(query: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Controller Agent] WARNING: No GEMINI_API_KEY found. Mocking compute budget.")
        return {
            "compute_budget": {
                "allow_reflection": True,
                "max_retrieval_steps": 2,
                "model_preference": "mistral",
                "rationale": "Mock controller defaulting to full research pipeline."
            }
        }
        
    chain = build_controller_chain()
    result = chain.invoke({"query": query})
    
    return {"compute_budget": result.dict()}
