import os
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv(Path(__file__).parent.parent.parent / "configs" / ".env")
load_dotenv() # Load from current directory as fallback

class PlannerOutput(BaseModel):
    intent: str = Field(description="The overarching intent of the user's query.")
    needs_retrieval: bool = Field(description="Whether external information retrieval is required.")
    needs_web: bool = Field(description="Whether web search specifically is required.")
    needs_reflection: bool = Field(description="Whether the output will need rigorous reflection/review.")
    difficulty: str = Field(description="Estimated difficulty: 'easy', 'medium', 'hard'.")
    expected_output: str = Field(description="The format of the expected output.")

def build_planner_chain():
    """
    Builds the LangChain runnable for the Planner agent.
    Reads the prompt from src/prompts/planner.md.
    """
    # 1. Read the prompt file
    prompt_path = Path(__file__).parent.parent / "prompts" / "planner.md"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        # Fallback if running from a different working directory structure
        prompt_text = "Analyze the query and output JSON.\nQuery: {{query}}"

    # 2. Extract the actual prompt template (ignoring the markdown metadata headers if needed)
    # For simplicity, we just pass the whole text to the LLM, but LangChain's PromptTemplate works better.
    # We'll just extract the part after the "---" separator.
    parts = prompt_text.split("---")
    template_content = parts[-1].strip() if len(parts) > 1 else prompt_text.strip()
    
    # 3. Create the prompt template
    prompt = PromptTemplate.from_template(template_content, template_format="jinja2")

    # 4. Initialize the LLM
    # We use Gemini 2.5 Flash (or gemini-1.5-flash as the currently available fallback in API)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )

    # 5. Attach the structured output schema
    structured_llm = llm.with_structured_output(PlannerOutput)

    # 6. Create the chain
    chain = prompt | structured_llm
    
    return chain

def planner_agent(query: str) -> dict:
    """
    Executes the Planner agent logic.
    Returns a dictionary representing the structured output.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Planner Agent] WARNING: No GEMINI_API_KEY found. Using deterministic mock fallback for validation.")
        return {
            "intent": "research",
            "needs_retrieval": True,
            "needs_web": True,
            "needs_reflection": True,
            "difficulty": "medium",
            "expected_output": "technical report"
        }

    chain = build_planner_chain()
    # Note: Using invoke requires setting up the correct variable. 
    # Our prompt template uses {{query}} which Jinja2 expects as a kwargs parameter.
    result = chain.invoke({"query": query})
    return result.dict()
