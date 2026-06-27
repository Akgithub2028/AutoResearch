import os
import json
from pathlib import Path
from typing import Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_revision_chain():
    prompt_path = Path(__file__).parent.parent / "prompts" / "revision.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        prompt_text = "Revise this draft.\nDraft: {{draft}}\nReflection: {{reflection}}"
        
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")
    
    # Mistral is preferred for longer generations (revisions) per SKILL.md
    llm = ChatMistralAI(model="ministral-3b-2512", temperature=0.3)
    
    # We use StrOutputParser because we want raw Markdown prose, not JSON.
    return prompt | llm | StrOutputParser()

def revision_agent(draft: str, reflection: Dict[str, Any]) -> dict:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "your_mistral_api_key_here":
        print("[Revision Agent] WARNING: No MISTRAL_API_KEY found. Mocking revision.")
        mock_draft = f"""# Executive Summary
This draft has been revised! Mocked executive summary.

# Background
Synthesized background. Addressed critique: {reflection.get('critique', '')}

# Key Concepts
- Concept A
- Concept B

# Comparative Analysis
Mock comparison, now with more citations.

# Limitations
Due to the absence of API keys, this is a deterministic mockup.

# Future Work
Integration of the actual Mistral LLM endpoints.
"""
        return {"draft": mock_draft}
        
    chain = build_revision_chain()
    reflection_str = json.dumps(reflection, indent=2)
    
    result_markdown = chain.invoke({
        "draft": draft, 
        "reflection": reflection_str
    })
    
    return {"draft": result_markdown}
