import streamlit as st

st.set_page_config(page_title="PyZoBot: Zotero Library QA System", layout="wide")

# Header and Logo
header_html = """
<div style="text-align: center;">
    <img src='https://i.postimg.cc/4xPdhkB2/PYZo-Bot-new-logo-small.png' alt='PyZoBot Logo' style='width:auto; height:40%;'>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center;'>PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# Welcome message
st.markdown("""
## Welcome to PyZoBot System

Choose from our different implementations in the sidebar:

<ol>
    <li><strong>PyZoBot OpenAI:</strong>
        <ul>
            <li>Implementing OpenAI's LLMs: GPT-4, GPT-3.5-Turbo, GPT-4o</li>
            <li>Using OpenAI's Embedding Model: text-embedding-3-large</li>
        </ul>
            <br> 
    </li>
    <li><strong>PyZoBot Open Source:</strong>
        <ul>
            <li>Integrating Ollama with the open-source large language model:  Llama 3.1 (8B) and Mistral</li>
            <li>Using Open Source Embedding: "nomic-embed-text", "jina/jina-embeddings-v2-base-en"</li>
        </ul>
            <br> 
    </li>
    <li><strong>PyZoBot GraphRAG OpenAI:</strong>
        <ul>
            <li>Implementing Knowledge Graph with OpenAI's LLMs: GPT-4, GPT-3.5-Turbo, GPT-4o</li>
            <li>Using OpenAI's Embedding Model: text-embedding-3-large</li>
        </ul>
            <br> 
    </li>
    <li><strong>PyZoBot GraphRAG Open Source:</strong> 
        <ul>
            <li>Implementing a Knowledge Graph using Ollama and the open-source LLMs: Llama 3.1 (8B) and Mistral</li>
            <li>Using Open Source Embedding: "nomic-embed-text", "jina/jina-embeddings-v2-base-en"</li>
        </ul>
    </li>
            <br> 
</ol>


### Getting Started
1. Select your preferred version from the sidebar
2. Configure your API keys and settings
3. Start exploring your Zotero library!
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Select a version from the sidebar to begin!</p>",
    unsafe_allow_html=True
)