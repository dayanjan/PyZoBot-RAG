import streamlit as st  # type: ignore

import os
import shutil
import uuid
import re
import tempfile
from pyzotero import zotero
import pandas as pd
import requests
from pathlib import Path
import io
from llama_index.core import download_loader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from IPython.display import display, JSON

import json

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

# from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements, elements_to_json
import unstructured.partition
import nltk
from unstructured.partition.pdf import partition_pdf
from langchain_experimental.text_splitter import SemanticChunker
from llama_index.readers.file import PyMuPDFReader

import streamlit as st
import os
import shutil
import uuid
import re
import tempfile
from pyzotero import zotero
import pandas as pd
import requests
from pathlib import Path

from llama_index.core import download_loader
from llama_index.core import Settings
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
# from llama_index.core import Document
from llama_index.core import Document as LlamaDocument
from langchain.docstore.document import Document as LangchainDocument
import pyvis
import nest_asyncio
nest_asyncio.apply()
if "k" not in st.session_state:
    st.session_state["k"] = 5
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 1000
if "chunking_method" not in st.session_state:
    st.session_state["chunking_method"] = "recursive"
if "pdf_paths" not in st.session_state:
    st.session_state["pdf_paths"] = None
if "graph_index" not in st.session_state:
    st.session_state["graph_index"] = None


# Clear session state only when switching pages
if 'current_page' not in st.session_state or st.session_state['current_page'] != __file__:
    # Keep only essential initializations
    temp_k = st.session_state.get("k", 5)
    temp_max_tokens = st.session_state.get("max_tokens", 1000)
    temp_chunking_method = st.session_state.get("chunking_method", "recursive")
    
    # Clear all states
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    
    # Delete knowledge graph visualization file if it exists
    viz_file = "kg_viz.html"
    if os.path.exists(viz_file):
        try:
            os.remove(viz_file)
        except Exception as e:
            print(f"Error removing visualization file: {e}")
        
    # Restore essential variables
    st.session_state["k"] = temp_k
    st.session_state["max_tokens"] = temp_max_tokens
    st.session_state["chunking_method"] = temp_chunking_method
    st.session_state['current_page'] = __file__
    st.session_state["graph_index"] = None  # Always reset graph index on page switch

    # Version-specific initialization
    if __file__.endswith('GraphRAG_OpenAI.py'):
        st.session_state["model_name"] = "gpt-4"
    else:
        st.session_state["model_name"] = "mistral"
        st.session_state["embedding_model"] = "nomic-embed-text"

# Initialize variables at the top
split_docs = None
embeddings = None
ids = None

st.set_page_config(page_title="PyZoBot GraphRAG: Zotero Library QA System", layout="wide" )



header_html = """
<div style="text-align: center;">
    <img src='https://i.postimg.cc/4xPdhkB2/PYZo-Bot-new-logo-small.png' alt='PyZoBot Logo' style='width:auto; height:40%;'>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Display the title within the app
# st.title("PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation")
st.markdown(
    "<h1 style='text-align: center;'>PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation</h1>",
    unsafe_allow_html=True
)
st.markdown("---") 

# User inputs for Zotero
with st.sidebar:
    st.markdown(""" 
                <h2 style='color: red;'><strong>Step 1: Setup and Configure Zotero</strong></h3>""", unsafe_allow_html=True)
    zotero_api_key = st.text_input(
        "**Enter your Zotero API key:**", type="password", key="zotero_api_key"
    )
    library_type = st.selectbox(
        "**Select your Zotero library type:**", ["group", "user"], key="library_type"
    )
    library_id = st.text_input("**Enter your Zotero library ID Or User ID:**", key="library_id")
    st.markdown(""" <h4 style='color: red;'><strong>Collection Filter (Optional)""", unsafe_allow_html=True)
    collection_id = st.text_input(
        "**Enter Zotero collection ID (leave empty for all items):**",
        key="collection_id")
    
    # Fetch PDFs button moved here
    fetch_button = st.button("Fetch PDFs from Zotero", key="fetch_pdfs_button")
    

    st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
    st.markdown(""" 
                <h2 style='color: red;'><strong>Step 2: OpenAI Configuration</strong></h3>""", unsafe_allow_html=True)

    openai_api_key = st.text_input(
        "**Enter your OpenAI API key:**", type="password", key="openai_api_key"
    )
    st.session_state["model_name"] = st.selectbox(
        "**Select OpenAI model:**", 
        ["gpt-4", "gpt-3.5-turbo", "gpt-4o"], 
        index=["gpt-4", "gpt-3.5-turbo", "gpt-4o"].index(st.session_state["model_name"]), 
        key="model_name_select"
    )
    st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color: red;'><strong>Step 3: Document Processing Setup</strong></h3>
    """, unsafe_allow_html=True)
    
    chunking_method = st.selectbox(
    "**Select Chunking Method:**",
    ["Recursive", "Layout-Aware", "Semantic"],  # Added "Semantic" to the options
    index=0 if st.session_state["chunking_method"] == "recursive" else 
           1 if st.session_state["chunking_method"] == "layout-aware" else 2,
    key="chunking_method_radio"
)
    st.session_state["chunking_method"] = chunking_method.lower()
    
    # Show relevant parameters based on chunking method
    if st.session_state["chunking_method"] == "recursive":
        chunk_size = st.number_input(
            "**Enter chunk size:**",
            min_value=100,
            max_value=4000,
            value=500,
            key="chunk_size"
        )
        chunk_overlap = st.number_input(
            "**Enter chunk overlap:**",
            min_value=0,
            max_value=2000,
            value=200,
            key="chunk_overlap"
        )
    elif st.session_state["chunking_method"] == "layout-aware":
        st.info("Layout-aware chunking will preserve document structure and layout.")
    else:  # semantic
        st.info("Semantic chunking will create chunks based on meaningful content breaks using AI.")

    # Add Process Documents button
    st.markdown("""
    <h3 style='color: red;'><strong>Process Documents</strong></h3>
    """, unsafe_allow_html=True)
    
    process_button = st.button("Process Documents with Selected Parameters")

main_container = st.container()

def handle_library_id_change():
    """Handle changes in the Zotero library ID and cleanup resources."""
    if "previous_library_id" in st.session_state:
        if library_id != st.session_state["previous_library_id"]:
            st.session_state["library_id_changed"] = True
            
            # Clean up graph index
            if "graph_index" in st.session_state:
                try:
                    # Delete the graph index
                    st.session_state["graph_index"] = None
                    st.warning("Knowledge graph index cleared due to library change.")
                except Exception as e:
                    st.error(f"Error cleaning up graph index: {str(e)}")
            
            # Clean up temporary directories
            if "temp_dir" in st.session_state:
                try:
                    shutil.rmtree(st.session_state["temp_dir"], ignore_errors=True)
                    del st.session_state["temp_dir"]
                except Exception as e:
                    st.error(f"Error cleaning up temporary directory: {str(e)}")
            
            # Clean up chat history
            if "chat_history" in st.session_state:
                st.session_state["chat_history"] = []
                st.info("Chat history cleared due to library change.")
            
            st.session_state["previous_library_id"] = library_id
            return True  # Indicates a library change occurred
    else:
        st.session_state["library_id_changed"] = True
        st.session_state["previous_library_id"] = library_id
        return False  # First initialization

handle_library_id_change()


def handle_index_change(new_index):
    """Handle graph index changes and cleanup visualization."""
    try:
        # Clear existing visualization if exists
        if os.path.exists("kg_viz.html"):
            os.remove("kg_viz.html")
            
        # Update the index in session state
        st.session_state["graph_index"] = new_index
        
    except Exception as e:
        st.error(f"Error handling index change: {e}")
# def handle_index_change(new_index):
#     """Handle graph index changes and cleanup visualization."""
#     try:
#         # Clear existing visualization files if they exist
#         for filename in ["kg_viz.html", "kg_viz_download.html"]:
#             if os.path.exists(filename):
#                 os.remove(filename)
            
#         # Update the index in session state
#         st.session_state["graph_index"] = new_index
        
#     except Exception as e:
#         st.error(f"Error handling index change: {e}")


def create_unique_temp_dir():
    # Remove the existing directory if it exists
    if "temp_dir" in st.session_state:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    # Create a new temporary directory
    st.session_state.temp_dir = tempfile.mkdtemp()
    return st.session_state.temp_dir

def process_zotero_library(library_id, zotero_api_key, library_type, collection_id=None):
    zot = zotero.Zotero(
        library_id=library_id, library_type=library_type, api_key=zotero_api_key
    )
    items = zot.everything(zot.top())

    # Process Zotero items
    df = pd.json_normalize(items)
    
    # Add collection filtering if collection_id is provided
    if collection_id:
        # Filter rows based on the presence of the collection_id
        df = df[df['data.collections'].apply(lambda x: collection_id in x if isinstance(x, list) else False)]
        
        if df.empty:
            st.error(f"No items found in collection with ID: {collection_id}")
            return None
    
    df1 = df[df["meta.numChildren"] == 0]
    df2 = df[df["meta.numChildren"] != 0]
    df2["links.self.href"] = df2["links.self.href"].astype(str) + "/children"
    frames = [df1, df2]
    df3 = pd.concat(frames)
    df4 = df3

    def fetch_url_content_as_json(url):
        try:
            headers = {"Zotero-API-Key": f"{zotero_api_key}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

    df4["JSONContent"] = df4["links.self.href"].apply(fetch_url_content_as_json)

    def flatten_json(nested_json: dict, exclude: list = [""]) -> dict:
        """
        Flatten a list of nested dicts.
        """
        out = dict()

        def flatten(x: (list, dict, str), name: str = "", exclude=exclude):
            if type(x) is dict:
                for a in x:
                    if a not in exclude:
                        flatten(x[a], f"{name}{a}.")
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, f"{name}{i}.")
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(nested_json)
        return out

    df_source2 = pd.DataFrame([flatten_json(x) for x in df4["JSONContent"]])

    df9 = df_source2
    cols_to_join = [col for col in df9.columns if col.endswith(".enclosure.href")]
    df9["enclosure.href"] = df9[cols_to_join].apply(
        lambda x: "##".join(x.values.astype(str)), axis=1
    )

    df10 = df9
    cols_to_join = [col for col in df10.columns if col.endswith(".enclosure.title")]
    df10["enclosure.title"] = df10[cols_to_join].apply(
        lambda x: "##".join(x.values.astype(str)), axis=1
    )
    df11 = df10[["enclosure.title", "enclosure.href"]]
    df12 = df11
    new_df = (
        df12["enclosure.title"]
        .str.split("##", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("enclosure.title")
    )
    df12 = df12.drop("enclosure.title", axis=1).join(new_df)
    df13 = df12
    new_df2 = (
        df13["enclosure.href"]
        .str.split("##", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("enclosure.href")
    )
    df13 = df13.drop("enclosure.href", axis=1).join(new_df2)
    df13.dropna(inplace=True)
    df15 = df13
    df15 = df15.replace("nan", pd.NA)
    df15 = df15.dropna()
    df15["PDF_Names"] = df15["enclosure.title"]
    df15 = df15[["PDF_Names", "enclosure.href"]]
    df16 = df15.drop_duplicates(keep="first")
    df17 = df16[df16["PDF_Names"].str.endswith(".pdf")]
    df20 = df17
    # Define your output folder
    output_folder = create_unique_temp_dir()
    headers = {"Zotero-API-Key": f"{zotero_api_key}"}
    # Iterate through the dataframe
    for index, row in df20.iterrows():
        api_url = row["enclosure.href"]
        pdf_filename = row["PDF_Names"]
        # Make an HTTP GET request for each URL
        response = requests.get(api_url, headers=headers)
        # Check if the request was successful
        if response.status_code == 200:
            binary_content = response.content
            content_type = response.headers.get("Content-Type")
            # Check if the content type is 'application/pdf'
            if content_type == "application/pdf":
                pdf_filename = row["PDF_Names"]
                pdf_filepath = os.path.join(output_folder, pdf_filename)
                # Save the PDF to the specified folder
                with open(pdf_filepath, "wb") as pdf_file:
                    pdf_file.write(binary_content)
                print(f"Saved PDF: {pdf_filename}")
            else:
                print(f"Skipped non-PDF content for URL: {api_url}")
        else:
            print(f"Failed to fetch data from the API for URL: {api_url}")
    return output_folder



if fetch_button:
    if zotero_api_key and library_id and library_type:
        try:
            with st.spinner("Fetching PDFs..."):
                temp_dir = process_zotero_library(
                    library_id, 
                    zotero_api_key, 
                    library_type,
                    collection_id if collection_id else None
                )
                
                if temp_dir:
                    pdf_files = os.listdir(temp_dir)
                    if pdf_files:
                        # Store pdf paths
                        pdf_paths = [
                            os.path.join(temp_dir, file).replace("\\", "//")
                            for file in pdf_files
                        ]
                        st.session_state["pdf_paths"] = pdf_paths
                        
                        # Display success message and files
                        st.success(f"Successfully fetched {len(pdf_files)} PDFs")
                        st.write("Retrieved PDFs:")
                        for file in pdf_files:
                            st.write(f"- {file}")
                    else:
                        st.warning("No PDF files found in the specified collection.")
                else:
                    st.error("Failed to create temporary directory for PDFs.")
                
                # Set library_id_changed to True to trigger cleanup and reinitialization
                st.session_state["library_id_changed"] = True
                
        except Exception as e:
            st.error(f"Error fetching PDFs: {str(e)}")
    else:
        st.error("Please enter all required Zotero details (API key, library ID, and type).")

###################################################################################################11/16/2024
def process_with_recursive_chunking(pdf_paths, chunk_size, chunk_overlap):
    """Process PDFs using recursive chunking method."""
    try:
        # Import and initialize the PyMuPDF loader
        PyMuPDFReader = download_loader("PyMuPDFReader")
        loader = PyMuPDFReader()
        all_documents = []
        split_docs = []

        # Load documents
        with st.spinner("Document loading..."):
            for pdf_file in pdf_paths:
                documents = loader.load_data(file_path=pdf_file, metadata=True)
                all_documents.extend(documents)

        if not all_documents:
            return None, None, "No documents were loaded."

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Process each document
        with st.spinner("Chunking documents..."):
            for document in all_documents:
                text = document.text  # Use the correct attribute to get text
                source = document.metadata["file_path"].split("/")[-1]
                chunks = text_splitter.split_text(text)
                
                for chunk in chunks:
                    metadata = {"source": source}
                    chunk_instance = LangchainDocument(page_content=chunk, metadata=metadata)
                    split_docs.append(chunk_instance)

        # Generate IDs for chunks
        if split_docs:
            ids = [str(i) for i in range(1, len(split_docs) + 1)]
            return split_docs, ids, f"{len(split_docs)} chunks were successfully created using recursive chunking."
        else:
            return None, None, "No chunks were created."
        
    except Exception as e:
        return None, None, f"Error in recursive chunking: {str(e)}"


def process_with_layout_chunking(pdf_paths):
    """Process PDFs using layout-aware chunking method."""
    try:
        split_docs = []

        with st.spinner("Processing documents with layout-aware chunking..."):
            for pdf_path in pdf_paths:
                # Normalize the path for Windows
                normalized_path = os.path.normpath(pdf_path)
                
                try:
                    # Read the PDF file content
                    with open(normalized_path, 'rb') as file:
                        pdf_content = file.read()
                    
                    # Create a temporary file with a simple name
                    temp_dir = tempfile.mkdtemp()
                    temp_pdf = os.path.join(temp_dir, "temp.pdf")
                    
                    with open(temp_pdf, 'wb') as file:
                        file.write(pdf_content)
                    
                    # Use the temporary file for processing
                    elements = partition_pdf(
                        temp_pdf,
                        strategy="hi_res",
                        extract_images_in_pdf=False
                    )

                    source = os.path.basename(pdf_path)
                    
                    for element in elements:
                        if hasattr(element, 'text') and element.text.strip():
                            metadata = {"source": source}
                            chunk_instance = LangchainDocument(
                                page_content=element.text.strip(),
                                metadata=metadata
                            )
                            split_docs.append(chunk_instance)
                            
                    # Clean up temporary files
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                except Exception as e:
                    st.warning(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
                    continue

        if split_docs:
            ids = [str(i) for i in range(1, len(split_docs) + 1)]
            return split_docs, ids, f"{len(split_docs)} chunks were successfully created using layout-aware chunking."
        else:
            return None, None, "No chunks were created."
        
    except Exception as e:
        return None, None, f"Error in layout-aware chunking: {str(e)}"
    

def process_with_semantic_chunking(pdf_paths, openai_api_key):
    """Process PDFs using semantic chunking method."""
    try:
        split_docs = []
        
        with st.spinner("Processing documents with semantic chunking..."):
            # Initialize the semantic chunker
            semantic_chunker = SemanticChunker(
                OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=openai_api_key
                ),
                breakpoint_threshold_type="gradient"
            )
            
            # Initialize PyMuPDF loader
            PyMuPDFReader = download_loader("PyMuPDFReader")
            loader = PyMuPDFReader()
            
            # Process each PDF
            for pdf_path in pdf_paths:
                try:
                    # Load and process document
                    documents = loader.load(file_path=pdf_path)
                    text_content = ""
                    for doc in documents:
                        text_content += doc.text + "\n"
                    
                    # Create semantic chunks
                    semantic_chunks = semantic_chunker.create_documents([text_content])
                    
                    # Get source filename
                    source = os.path.basename(pdf_path)
                    
                    # Convert to Document instances
                    for chunk in semantic_chunks:
                        doc_instance = LangchainDocument(
                            page_content=chunk.page_content,
                            metadata={"source": source}
                        )
                        split_docs.append(doc_instance)
                    
                    
                except Exception as e:
                    st.warning(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
                    continue
            
            if split_docs:
                ids = [str(i) for i in range(1, len(split_docs) + 1)]
                return split_docs, ids, f"{len(split_docs)} chunks were successfully created using semantic chunking."
            else:
                return None, None, "No chunks were created."
                
    except Exception as e:
        return None, None, f"Error in semantic chunking: {str(e)}"





def create_graph_index(split_docs, openai_api_key, model_name):
    """Create and return a knowledge graph index from the split documents."""
    try:
        # Convert LangChain documents to LlamaIndex documents
        llama_docs = []
        for doc in split_docs:
            llama_doc = LlamaDocument(
                text=doc.page_content,
                metadata={"source": doc.metadata.get("source", "unknown")}
            )
            llama_docs.append(llama_doc)
        
        # Set up LLM with user-selected model
        llm = OpenAI(temperature=0, model=model_name, api_key=openai_api_key)
        Settings.llm = llm
        
        # Configure embedding model with retries
        embed_model = OpenAIEmbedding(
            api_key=openai_api_key,
            embed_model="text-embedding-3-large",
            retry_on_exceptions=True,
            max_retries=5,
            retry_interval_seconds=2
        )
        Settings.embed_model = embed_model
        
        # Create the property graph index using converted documents
        
        index = PropertyGraphIndex.from_documents(
            documents=llama_docs,  # Use converted documents
            embed_model=embed_model,
            kg_extractors=[
                ImplicitPathExtractor(),
                SimpleLLMPathExtractor(
                    llm=llm,
                    num_workers=4,
                    max_paths_per_chunk=25,
                ),
            ],
            show_progress=True,
            use_async=False
        )
        
        return index, "Knowledge graph index created successfully ✨"
    except Exception as e:
        error_msg = str(e)
        if "Connection error" in error_msg:
            return None, "Error: Unable to connect to OpenAI API. Please check your internet connection and API key."
        return None, f"Error creating knowledge graph index: {error_msg}"

def answer_question(question, graph_index):
    """Answer questions using the knowledge graph index."""
    try:
        # Create query engine with keyword retrieval
        query_engine = graph_index.as_query_engine(
            retriever_mode="keyword",
            verbose=True,
            response_mode="tree_summarize",
        )
        
        # Get response
        response = query_engine.query(question)
        
        return {
            "answer": str(response),
            "sources": [node.text for node in response.source_nodes] if hasattr(response, 'source_nodes') else []
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "sources": []
        }



# Handle document processing and create tabs
# def create_graph_visualization(graph_index):
#     """Create an interactive visualization of the knowledge graph."""
#     try:
#         with st.spinner("Generating knowledge graph visualization..."):
#             # Convert the graph to NetworkX format and save as HTML
#             g = graph_index.property_graph_store.save_networkx_graph(name="kg_viz.html")
            
#             # Read the generated HTML file
#             with open("kg_viz.html", "r", encoding="utf-8") as f:
#                 graph_html = f.read()
            
#             return graph_html
#     except Exception as e:
#         st.error(f"Error creating visualization: {str(e)}")
#         return None
# def create_graph_visualization(graph_index):
#     """Create an interactive visualization of the knowledge graph."""
#     try:
#         with st.spinner("Generating knowledge graph visualization..."):
#             # Convert the graph to NetworkX format and save as HTML
#             g = graph_index.property_graph_store.save_networkx_graph(name="kg_viz.html")
            
#             # Read the generated HTML file
#             with open("kg_viz.html", "r", encoding="utf-8") as f:
#                 graph_html = f.read()
            
#             return graph_html
#     except Exception as e:
#         st.error(f"Error creating visualization: {str(e)}")
#         return None
def create_graph_visualization(graph_index):
    """Create an interactive visualization of the knowledge graph."""
    try:
        with st.spinner("Generating knowledge graph visualization..."):
            # Convert the graph to NetworkX format and save as HTML
            g = graph_index.property_graph_store.save_networkx_graph(name="kg_viz.html")
            
            # Ensure file is completely written by reading after creation
            with open("kg_viz.html", "r", encoding="utf-8") as f:
                graph_html = f.read()
            
            # Store in session state to prevent file access during download
            st.session_state['current_graph_html'] = graph_html
            
            return graph_html
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None




# Handle document processing
if process_button and openai_api_key and "pdf_paths" in st.session_state:
    # Check for library changes and handle cleanup
    library_changed = handle_library_id_change()
    
    # Clear existing graph index if it exists or if library changed
    if "graph_index" in st.session_state:
        st.session_state["graph_index"] = None
        if not library_changed:  # Only show this message if not already shown by library change
            st.info("Previous knowledge graph index cleared. Creating new index...")
    
    # Process documents based on selected chunking method
    if st.session_state["chunking_method"] == "recursive":
        split_docs, ids, chunking_message = process_with_recursive_chunking(
            st.session_state["pdf_paths"],
            chunk_size,
            chunk_overlap
        )
    elif st.session_state["chunking_method"] == "layout-aware":
        split_docs, ids, chunking_message = process_with_layout_chunking(
            st.session_state["pdf_paths"]
        )
    else:  # semantic
        split_docs, ids, chunking_message = process_with_semantic_chunking(
            st.session_state["pdf_paths"],
            openai_api_key
        )

    st.write(chunking_message)
    
    if split_docs:
        # Create knowledge graph index
        with st.spinner("Creating knowledge graph index... This may take a few minutes."):
            graph_index, graph_message = create_graph_index(
                split_docs, 
                openai_api_key,
                st.session_state["model_name"]
            )
        
        if graph_index:
            # st.session_state["graph_index"] = graph_index
            handle_index_change(graph_index)
            st.success(graph_message)
        else:
            st.error(graph_message)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Create main container and tabs right after sidebar
# main_container = st.container()
with main_container:

    # Add tab state tracking
    if "current_tab" not in st.session_state:
        st.session_state["current_tab"] = "Chat Interface"
    
    # Create tabs that are always visible
    chat_tab, viz_tab = st.tabs(["💬 Chat Interface", "❉ Knowledge Graph"])

    # Store current tab selection
    if viz_tab:
        st.session_state["current_tab"] = "Knowledge Graph"
    if chat_tab:
        st.session_state["current_tab"] = "Chat Interface"
    
    with chat_tab:
        # Chat history management
        st.subheader("Chat History")
        
                
        # Display chat history with styling
        if st.session_state.get("chat_history", []):
            st.markdown("""
                <style>
                .chat-message {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                }
                .question {
                    background-color: #f0f2f6;
                }
                .answer {
                    background-color: #e8f0fe;
                }
                .sources {
                    margin-left: 20px;
                    font-size: 0.9em;
                    color: #666;
                    background-color:#fef6e8;
                }
                </style>
            """, unsafe_allow_html=True)
            
            for i, (q, a, sources) in enumerate(st.session_state["chat_history"], 1):
                st.markdown(f'<div class="chat-message question">'
                          f'<strong>Question {i}:</strong><br>{q}</div>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message answer">'
                          f'<strong>Answer:</strong><br>{a}</div>', 
                          unsafe_allow_html=True)
                if sources:
                    st.markdown('<div class="sources">'
                              '<strong>Sources:</strong>', 
                              unsafe_allow_html=True)
                    for source in sources:
                        st.markdown(f'<div class="sources">- {source}</div>', 
                                  unsafe_allow_html=True)
        else:
            st.info("No chat history yet. Start asking questions to build your conversation history.")
        
        st.markdown("<hr style='border: 2px solid #333;'>", unsafe_allow_html=True)
        
        # Question input
        st.markdown("""
        <style>
        .question-input-label {
            font-size: 20px;
            color: #ff3633;
            margin-bottom: -30px;
        }
        </style>
        <p class="question-input-label"><strong>Enter your question:</strong></p>
        """, unsafe_allow_html=True)
        
        question = st.text_input("", key="question_input")
        
        if st.button("Get Answer"):
            if "graph_index" not in st.session_state or st.session_state["graph_index"] is None:
                st.error("Please process the documents first to create the knowledge graph.")
            else:
                with st.spinner("Generating answer..."):
                    result = answer_question(question, st.session_state["graph_index"])
                
                answer = result["answer"]
                sources = result["sources"]
                
                # Save to chat history
                st.session_state["chat_history"].append((question, answer, sources))
                
                # Display results
                st.subheader("Answer:")
                st.write(answer)
                st.markdown("---")
                
                if sources:
                    st.subheader("Sources:")
                    for source in sources:
                        st.write(f"- {source}")

        # Add chat management buttons in a row
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("🗑️ Clear History"):
                st.session_state["chat_history"] = []
                st.rerun()
        with col2:
            if st.session_state.get("chat_history", []):
                # Prepare chat history for download
                chat_history_str = ""
                for i, (q, a, sources) in enumerate(st.session_state["chat_history"], 1):
                    chat_history_str += f"\nQuestion {i}: {q}\n"
                    chat_history_str += f"Answer: {a}\n"
                    if sources:
                        chat_history_str += "Sources:\n"
                        for source in sources:
                            chat_history_str += f"- {source}\n"
                    chat_history_str += "-" * 50 + "\n"
                chat_history_bytes = io.BytesIO(chat_history_str.encode('utf-8'))
                st.download_button(
                    "📥 Download Chat History",
                    data=chat_history_bytes,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )   
    with viz_tab:
        st.subheader("Knowledge Graph Visualization")
        
        if "graph_index" not in st.session_state or st.session_state["graph_index"] is None:
            st.info("Process your documents to generate the knowledge graph visualization. Once processed, you'll be able to see and interact with the graph here.")
        else:
            st.info("This visualization shows the relationships between concepts in your documents. You can interact with it by dragging nodes and zooming.")
            

            
            # Generate and display visualization
            graph_html = create_graph_visualization(st.session_state["graph_index"])
            if graph_html:
                # Custom CSS for the visualization
                st.markdown("""
                    <style>
                        iframe {
                            width: 100%;
                            height: 600px;
                            border: none;
                            border-radius: 10px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            background-color: white;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                # Display the graph
                st.components.v1.html(graph_html)
                
                # # Add download button for the visualization
                # st.download_button(
                #     label="📥 Download Graph Visualization",
                #     data=graph_html,
                #     file_name="knowledge_graph.html",
                #     mime="text/html",
            if 'current_graph_html' in st.session_state:
                    st.download_button(
                        label="📥 Download Graph Visualization",
                        data=st.session_state['current_graph_html'],  # Use stored HTML
                        file_name="knowledge_graph.html",
                        mime="text/html"
                    )
                
     

