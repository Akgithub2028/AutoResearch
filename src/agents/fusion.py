import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class Evidence(BaseModel):
    claim: str = Field(description="A specific, verifiable fact or assertion.")
    source: str = Field(description="The source identifier (e.g., URL or ArXiv ID).")
    confidence: float = Field(description="Confidence score for this claim (0.0 to 1.0).")
    citation: str = Field(description="A brief inline citation reference.")
    type: str = Field(description="Type of source (e.g., 'paper', 'web', 'book').")

class FusionOutput(BaseModel):
    evidence: List[Evidence] = Field(description="List of normalized evidence claims.")

def build_fusion_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "fusion.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Synthesize these into claims.\nQuery: {{query}}\nDocs: {{graded_documents}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(FusionOutput)
    return prompt | structured_llm

def fusion_agent(query: str, graded_documents: List[Dict[str, Any]]) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Fusion Agent] WARNING: No GEMINI_API_KEY found. Mocking fusion.")
        if len(graded_documents) == 0:
            return {"fused_evidence": []}
            
        mock_evidence = []
        for i, doc in enumerate(graded_documents):
            mock_evidence.append({
                "claim": f"Mocked extracted claim from doc {i}",
                "source": doc.get("source", "unknown"),
                "confidence": 0.85,
                "citation": f"[Doc {i}]",
                "type": doc.get("type", "unknown")
            })
        return {"fused_evidence": mock_evidence}
        
    chain = build_fusion_chain()
    docs_str = "\n".join([str(d) for d in graded_documents])
    result = chain.invoke({"query": query, "graded_documents": docs_str})
    
    # Convert Pydantic objects to dicts for state
    fused_evidence = [ev.dict() for ev in result.evidence]
    return {"fused_evidence": fused_evidence}
