import os
from pathlib import Path
from typing import Dict, Any, List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_writer_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "writer.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Write a report.\nQuery: {{query}}\nIntent: {{intent}}\nEvidence: {{evidence}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    
    # Mistral is preferred for longer generations per SKILL.md
    llm = ChatMistralAI(model="ministral-3b-2512", temperature=0.3)
    
    # We use StrOutputParser because we want raw Markdown prose, not JSON.
    return prompt | llm | StrOutputParser()

def writer_agent(query: str, intent: str, evidence: List[Dict[str, Any]]) -> dict:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "your_mistral_api_key_here":
        print("[Writer Agent] WARNING: No MISTRAL_API_KEY found. Mocking draft generation.")
        mock_draft = f"""# Executive Summary
Mocked executive summary for query: {query}.

# Background
This is a synthesized background based on the {len(evidence)} verified claims.

# Key Concepts
- Concept A
- Concept B

# Comparative Analysis
Mock comparison.

# Limitations
Due to the absence of API keys, this is a deterministic mockup.

# Future Work
Integration of the actual Mistral LLM endpoints.
"""
        return {"draft": mock_draft}
        
    chain = build_writer_chain()
    evidence_str = "\n".join([str(e) for e in evidence])
    
    result_markdown = chain.invoke({
        "query": query, 
        "intent": intent, 
        "evidence": evidence_str
    })
    
    return {"draft": result_markdown}
