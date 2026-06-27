import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class ReflectionOutput(BaseModel):
    critique: str = Field(description="High-level summary of the review.")
    unsupported_claims: List[str] = Field(description="List of claims in the draft lacking evidence.")
    hallucination_risk: str = Field(description="Risk level: 'low', 'medium', 'high'.")
    missing_citations: List[str] = Field(description="List of places where citations are missing.")
    is_satisfactory: bool = Field(description="True if no major revisions are needed, False otherwise.")

def build_reflection_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "reflection.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Review this draft.\nQuery: {{query}}\nEvidence: {{evidence}}\nDraft: {{draft}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    # Using Gemini 1.5 Flash for Reflection as dictated by SKILL.md
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(ReflectionOutput)
    return prompt | structured_llm

def reflection_agent(query: str, draft: str, evidence: List[Dict[str, Any]]) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Reflection Agent] WARNING: No GEMINI_API_KEY found. Mocking critique.")
        return {
            "reflection": {
                "critique": "Mock critique: The draft requires more specific citations based on evidence.",
                "unsupported_claims": ["Mock comparison"],
                "hallucination_risk": "medium",
                "missing_citations": ["Section 3"],
                "is_satisfactory": False
            }
        }
        
    chain = build_reflection_chain()
    evidence_str = "\n".join([str(e) for e in evidence])
    
    result = chain.invoke({
        "query": query, 
        "evidence": evidence_str,
        "draft": draft
    })
    
    return {"reflection": result.dict()}
