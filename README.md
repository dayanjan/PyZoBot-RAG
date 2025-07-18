
# PyZoBot: Zotero + LLM-Powered Conversational Retrieval Platform

PyZoBot is a Streamlit-based application that integrates your Zotero reference library with cutting-edge LLMs (GPT, LLaMA 3.1, Mistral) via Retrieval-Augmented Generation (RAG). It enables transparent question-answering and synthesis directly from curated scientific literature.

---

## 📁 Repository Structure

pyzobot_app/

├── lib/

├── pages/

│ ├── 1_PyZoBot_OpenAI.py

│ ├── 2_PyZoBot_OpenSource.py

│ ├── 3_PyZoBot_GraphRAG_OpenAI.py

│ └── 4_PyZoBot_GraphRAG_OpenSource.py

├── utils/

├── venv/

├── main.py

├── requirements.txt


---

## 🧠 Features

- 🔎 Supports **OpenAI models** (GPT-3.5, GPT-4, GPT-4o) and **open-source models** (Mistral, LLaMA 3.1 via Ollama)
- 📚 RAG & Graph-RAG architectures
- 📁 Zotero integration (PDFs, metadata, collection filtering)
- 🧠 Chunking options: `Recursive`, `Layout-aware`, `Semantic`
- 📈 Knowledge Graph visualization
- 🔐 Works **offline** with Ollama for full privacy

---

## 🚀 Quick Start Guide

### 1. Install Python 3.11.2
Download: [https://www.python.org/downloads/release/python-3112](https://www.python.org/downloads/release/python-3112)

### 2. Clone the Repo

```bash
git clone https://github.com/SuadAshammari/pyzobot_app.git
cd pyzobot_app
```


### 3. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```


### 5. Install and Set Up Ollama (For Open Source Use)
Install Ollama from: https://ollama.com

Then run:

```bash

ollama pull mistral
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull jina/jina-embeddings-v2-base-en
```

### 6. Run the App
```bash
streamlit run main.py
```

---

## 🔧 Sidebar Configuration Guide
### Zotero API Setup:
- Add your Zotero API key
- Choose library type: user or group
- Enter your library ID or group ID
- (Optional) Add collection ID

### Model Configuration:
- OpenAI: GPT models, API key
- Ollama: Select local LLM and embedding model

### Document Chunking:
- Choose chunking method
- Adjust chunk size & overlap

### Run Query:
- Select model
- Adjust retrieval k and max tokens
- Ask your research question!
