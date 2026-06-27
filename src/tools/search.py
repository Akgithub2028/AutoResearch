from typing import Dict, Any, List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun

def execute_web_search(query: str) -> List[Dict[str, Any]]:
    """
    Executes a real web search using DuckDuckGo.
    Returns a list of retrieved document dictionaries.
    """
    try:
        tool = DuckDuckGoSearchRun()
        result = tool.invoke({"query": query})
        
        return [{
            "source": f"wikipedia:{query}",
            "content": result,
            "type": "web"
        }]
    except Exception as e:
        print(f"[Web Search Tool] Error: {e}")
        return []

def execute_arxiv_search(query: str) -> List[Dict[str, Any]]:
    """
    Executes an academic search using ArXiv.
    """
    try:
        tool = ArxivQueryRun()
        result = tool.invoke({"query": query})
        
        return [{
            "source": f"arxiv:{query}",
            "content": result,
            "type": "paper"
        }]
    except Exception as e:
        print(f"[ArXiv Search Tool] Error: {e}")
        return []
