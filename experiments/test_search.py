from src.tools.search import execute_web_search

print("Testing DuckDuckGo Search...")
results = execute_web_search("Quantum entanglement")
print(f"Results retrieved: {len(results)}")
if len(results) > 0:
    print(results[0]["content"][:100] + "...")
