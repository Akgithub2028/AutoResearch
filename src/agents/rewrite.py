import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class RewriteOutput(BaseModel):
    rewritten_queries: List[str] = Field(description="List of optimized search queries.")

def build_rewrite_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "rewrite.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Rewrite query based on rationale.\nQuery: {{original_query}}\nRationale: {{grader_rationale}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    structured_llm = llm.with_structured_output(RewriteOutput)
    return prompt | structured_llm

def rewrite_agent(original_query: str, grader_rationale: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Rewrite Agent] WARNING: No GEMINI_API_KEY found. Mocking rewrite.")
        return {
            "search_queries": [original_query + " fundamental mechanisms", "overview of " + original_query]
        }
        
    chain = build_rewrite_chain()
    result = chain.invoke({"original_query": original_query, "grader_rationale": grader_rationale})
    return {
        "search_queries": result.rewritten_queries
    }
