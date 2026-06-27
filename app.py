import streamlit as st
import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from md2pdf.core import md2pdf

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "configs", ".env"))

from src.graph import build_graph
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AutoResearch Assistant", page_icon="🔬", layout="wide")

st.title("AutoResearch Assistant 🔬")
st.markdown("A Self-Reflective Hierarchical Multi-Agent Research Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("API Keys")
    gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    mistral_key = st.text_input("Mistral API Key", type="password", value=os.getenv("MISTRAL_API_KEY", ""))
    
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
        
    st.subheader("Budget Overrides (Optional)")
    st.info("The Adaptive Controller normally sets these dynamically. You can enforce overrides here.")
    force_max_retrieval = st.slider("Max Retrieval Loops", 1, 5, 3)
    force_reflection = st.checkbox("Allow Reflection", value=True)
    
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_draft" not in st.session_state:
    st.session_state.research_draft = None
if "research_metrics" not in st.session_state:
    st.session_state.research_metrics = None

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_pdf(markdown_content: str):
    css_file_path = "/tmp/style.css"
    with open(css_file_path, "w") as f:
        f.write("body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; } h1, h2, h3 { color: #333; } img { max-width: 100%; }")
    pdf_path = "/tmp/research_report.pdf"
    md2pdf(pdf_path, md_content=markdown_content, css_file_path=css_file_path)
    return pdf_path

# Q&A Follow-up Chain
def get_qa_chain():
    template = """
You are an advanced AI research assistant. The user has previously requested a research report.
Here is the generated research report for context:

<research_report>
{draft}
</research_report>

Please answer the user's follow-up question based on the provided report. If the answer is not in the report, use your general knowledge but mention that it wasn't covered in the initial research.

User Question: {question}
"""
    prompt = PromptTemplate.from_template(template)
    llm = ChatMistralAI(model="ministral-3b-2512", temperature=0.3, streaming=True)
    return prompt | llm | StrOutputParser()

# Chat Input
if prompt := st.chat_input("Enter a research topic (e.g. 'Quantum Entanglement anomalies') or ask a follow-up..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Mode 1: Research Generation (if no draft exists yet)
        if st.session_state.research_draft is None:
            st.info("Initiating Autonomous Research Pipeline...")
            
            # Setup graph
            graph = build_graph()
            config = {"configurable": {"thread_id": f"streamlit_session_{int(time.time())}"}}
            initial_state = {
                "query": prompt,
                "execution_history": []
            }
            
            with st.status("Executing Multi-Agent Graph...", expanded=True) as status:
                for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                    for node_name, node_state in event.items():
                        st.write(f"✅ Node **{node_name}** completed.")
                        
                status.update(label="Research Complete!", state="complete", expanded=False)
            
            # Retrieve final draft from the final state output using get_state
            full_state = graph.get_state(config).values
            draft = full_state.get("draft", "Error: No draft generated.")
            metrics = full_state.get("metrics", {})
            
            st.session_state.research_draft = draft
            st.session_state.research_metrics = metrics
            
            response_text = f"### Research Complete\n\n{draft}"
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Visualizations
            st.subheader("📊 Execution Analytics")
            col1, col2, col3, col4 = st.columns(4)
            ex_metrics = metrics.get("execution_metrics", {})
            col1.metric("Latency (s)", ex_metrics.get("wall_clock_latency_seconds", 0))
            col2.metric("LLM Calls", ex_metrics.get("llm_calls", 0))
            col3.metric("Reflection Loops", ex_metrics.get("reflection_iterations", 0))
            col4.metric("Retrieval Steps", ex_metrics.get("retrieval_steps", 0))
            
            # PDF Generation
            try:
                pdf_path = generate_pdf(draft)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_file,
                        file_name="research_report.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")

        # Mode 2: Interactive Q&A (if draft exists)
        else:
            chain = get_qa_chain()
            # We use chain.stream directly and pass to st.write_stream
            stream_generator = chain.stream({
                "draft": st.session_state.research_draft,
                "question": prompt
            })
            
            response = st.write_stream(stream_generator)
            st.session_state.messages.append({"role": "assistant", "content": response})
