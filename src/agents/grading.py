import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class GraderOutput(BaseModel):
    relevance_score: float = Field(description="Score between 0 and 1 indicating relevance.")
    reliability_score: float = Field(description="Score between 0 and 1 indicating source reliability.")
    coverage_score: float = Field(description="Score between 0 and 1 indicating coverage completeness.")
    overall_confidence: float = Field(description="Combined confidence score between 0 and 1.")
    rationale: str = Field(description="Brief explanation of the grading.")

def build_grader_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "grading.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Evaluate this.\nQuery: {{query}}\nDocs: {{documents}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(GraderOutput)
    return prompt | structured_llm

mock_call_count = 0

def grading_agent(query: str, documents: List[Dict[str, Any]]) -> dict:
    global mock_call_count
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Grading Agent] WARNING: No GEMINI_API_KEY found. Mocking evaluation.")
        mock_call_count += 1
        
        # To prevent infinite reflection loops during validation:
        if mock_call_count > 1:
            return {
                "overall_confidence": 0.8,
                "rationale": "Mock override: Second pass, assuming rewrite fixed it.",
                "graded_documents": documents
            }
        
        if len(documents) == 0:
            return {
                "overall_confidence": 0.4, # Should trigger rewrite
                "rationale": "No documents were retrieved.",
                "graded_documents": documents
            }
        return {
            "overall_confidence": 0.9, # Will pass
            "rationale": "Documents seem sufficient.",
            "graded_documents": documents
        }
        
    chain = build_grader_chain()
    docs_str = "\n".join([str(d) for d in documents])
    result = chain.invoke({"query": query, "documents": docs_str})
    
    output = result.dict()
    output["graded_documents"] = documents
    return output
