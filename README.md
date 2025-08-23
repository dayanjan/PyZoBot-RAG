
# PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation

## Authors

**Lead Developer:**
- **Suad Alshammari, Pharm.D.** <sup>1,2</sup>
  - Department of Clinical Practice, College of Pharmacy, Northern Border University, Rafha, Saudi Arabia
  - Department of Pharmacotherapy and Outcomes Science, School of Pharmacy, Virginia Commonwealth University, Richmond, VA, USA
  - *This work was developed as part of Dr. Alshammari's PhD thesis*

**Project Supervisor:**
- **Dayanjan S. Wijesinghe, Ph.D.** <sup>2</sup>
  - Department of Pharmacotherapy and Outcomes Science, School of Pharmacy, Virginia Commonwealth University, Richmond, VA, USA

**Contributors:**
- **Walaa Abu Rukbah, Pharm.D.** <sup>2,3</sup> - Department of Pharmacy Practice, Faculty of Pharmacy, University of Tabuk, Saudi Arabia
- **Lama Basalelah, Pharm.D.** <sup>2,4</sup> - Faculty of Pharmacy, Imam Abdulrahman Bin Faisal University, Saudi Arabia
- **Ali Alsuhibani, Pharm.D.** <sup>2,5</sup> - Department of Pharmacy Practice, Unaizah College of Pharmacy, Qassim University, Saudi Arabia
- **Ali Alghubayshi, Pharm.D.** <sup>2,6</sup> - Department of Clinical Pharmacy, School of Pharmacy, University of Hail, Saudi Arabia
- **Bridget T. McInnes, Ph.D.** <sup>7</sup> - Department of Computer Science, College of Engineering, Virginia Commonwealth University, Richmond, VA, USA

---

## About PyZoBot

PyZoBot is an AI-driven platform developed using Python that addresses the exponential growth of scientific literature and information overload challenges. It integrates Zotero's reference management capabilities with advanced Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) technology.

### Key Innovations:
- **Merges traditional reference management with AI**: Combines Zotero's curated libraries with cutting-edge LLMs
- **Multiple implementation options**: Offers both cloud-based (OpenAI) and privacy-conscious local (open-source) deployments
- **Advanced retrieval techniques**: Includes standard RAG and knowledge graph-enhanced RAG implementations
- **Transparent information synthesis**: Displays specific document chunks and citations used for response generation
- **Research integrity**: Maintains scholarly standards through traceable references and citations

PyZoBot is a Streamlit-based application that enables transparent question-answering and synthesis directly from curated scientific literature, effectively managing information overload while maintaining research integrity.

<img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/73f2c68c-197e-4227-9a57-f83821317078" />

---

## 📁 Repository Structure
```
pyzobot_app/
├── pages/
│ ├── 1_PyZoBot_OpenAI.py
│ ├── 2_PyZoBot_OpenSource.py
│ ├── 3_PyZoBot_GraphRAG_OpenAI.py
│ └── 4_PyZoBot_GraphRAG_OpenSource.py
├── main.py
├── requirements.txt
```

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
