
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

from llama_index.core import download_loader
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

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


if "k" not in st.session_state:
    st.session_state["k"] = 5
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 1000
if "chunking_method" not in st.session_state:
    st.session_state["chunking_method"] = "recursive"
if "pdf_paths" not in st.session_state:
    st.session_state["pdf_paths"] = None


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
    if 'graph_html' in st.session_state:
        del st.session_state['graph_html']      

    # Restore essential variables
    st.session_state["k"] = temp_k
    st.session_state["max_tokens"] = temp_max_tokens
    st.session_state["chunking_method"] = temp_chunking_method
    st.session_state['current_page'] = __file__

    # Version-specific initialization
    if __file__.endswith('OpenAI.py'):
        st.session_state["model_name"] = "gpt-4"
    else:
        st.session_state["model_name"] = "mistral"
        st.session_state["embedding_model"] = "nomic-embed-text"

# Initialize variables at the top
split_docs = None
embeddings = None
ids = None

st.set_page_config(page_title="PyZoBot: Zotero Library QA System", layout="wide" )



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

    st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color: red;'><strong>Step 4: Query Processing and Response Generation</strong>
    """, unsafe_allow_html=True)



    st.session_state["model_name"] = st.selectbox(
        "**Select OpenAI model:**", 
        ["gpt-4", "gpt-3.5-turbo", "gpt-4o"], 
        index=["gpt-4", "gpt-3.5-turbo", "gpt-4o"].index(st.session_state["model_name"]), 
        key="model_name_select"
    )
    st.session_state["max_tokens"] = st.number_input(
        "**Enter max tokens:**", 
        min_value=100, 
        max_value=4000, 
        value=st.session_state["max_tokens"], 
        key="max_tokens_input"
    )
    st.session_state["k"] = st.slider(
        "**Number of documents to retrieve:**", 
        min_value=1, 
        max_value=30, 
        value=st.session_state["k"], 
        key="k_slider"
    )
     


# Function to handle library ID change
def handle_library_id_change():
    if "previous_library_id" in st.session_state:
        if library_id != st.session_state["previous_library_id"]:
            # Library ID has changed
            st.session_state["library_id_changed"] = True
            # Remove 'db' and related data from session state if they exist
            for key in ["db", "all_documents", "pdf_paths", "chroma_persist_dir"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Remove temporary directories
            if "temp_dir" in st.session_state:
                shutil.rmtree(st.session_state["temp_dir"], ignore_errors=True)
                del st.session_state["temp_dir"]
            if "chroma_persist_dir" in st.session_state:
                shutil.rmtree(st.session_state["chroma_persist_dir"], ignore_errors=True)
                del st.session_state["chroma_persist_dir"]
    else:
        st.session_state["library_id_changed"] = True  # First time
    # Update previous_library_id
    st.session_state["previous_library_id"] = library_id

# Call the function to handle library ID change
handle_library_id_change()



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
                text = document.text
                source = document.metadata["file_path"].split("/")[-1]
                chunks = text_splitter.split_text(text)
                
                for chunk in chunks:
                    metadata = {"source": source}
                    chunk_instance = Document(page_content=chunk, metadata=metadata)
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
                            chunk_instance = Document(
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
                        doc_instance = Document(
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



def create_vector_store(split_docs, ids, openai_api_key, library_id):
    """Create and save vector store from chunked documents."""
    try:
        if not split_docs or not ids:
            return None, None, "No documents to process."

        with st.spinner("Creating embeddings and building vector store..."):
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-ada-002"
            )

            persist_directory = tempfile.mkdtemp()
            library_id_sanitized = re.sub(r"\W+", "_", library_id)
            collection_id = str(uuid.uuid4())[:8]
            collection_name = f"user_vectors_{library_id_sanitized}_{collection_id}"

            if len(collection_name) > 63:
                collection_name = collection_name[:63]
            if not collection_name[0].isalnum():
                collection_name = "a" + collection_name[1:]
            if not collection_name[-1].isalnum():
                collection_name = collection_name[:-1] + "z"

            db = Chroma.from_documents(
                split_docs,
                embeddings,
                collection_name=collection_name,
                ids=ids,
                persist_directory=persist_directory,
            )
            db.persist()

            return db, persist_directory, "Vector store successfully created and saved."

    except Exception as e:
        return None, None, f"Error creating vector store: {str(e)}"

if openai_api_key and "pdf_paths" in st.session_state:
    if process_button:
        if "db" in st.session_state:
            if "chroma_persist_dir" in st.session_state:
                shutil.rmtree(st.session_state["chroma_persist_dir"], ignore_errors=True)
            del st.session_state["db"]

        # Choose chunking method based on user selection
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

        if split_docs and ids:
            # Create vector store
            db, persist_dir, vector_store_message = create_vector_store(
                split_docs,
                ids,
                openai_api_key,
                library_id
            )
            
            if db:
                st.session_state["db"] = db
                st.session_state["chroma_persist_dir"] = persist_dir
                st.success(vector_store_message)
            else:
                st.error(vector_store_message)
    

##################################################################################
import io

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def answer_question(question, db):
    system_template = """
        Answer the user query using ONLY the relevant content provided in this prompt.
        Do not use any external knowledge or information not present in the given context.
        If the provided content does not contain information to answer the query, respond with "I don't have enough information to answer this question based on the given context."
        Take your time and provide as much information as you can in the answer.\n

        For each statement in your answer, provide in-text citations after the sentence, e.g., [1].
        Start with number [1] every time you generate an answer and make the number matching the source document.
        If a statement has multiple citations, provide them all, e.g., [1], [2], [3].

        By the end of the answer, provide a References section as Markdown (### References) including the number and the file name, e.g.:
        [1] Author et al. - YEAR - file name.pdf

        Write each reference on a new line.

        {summaries}
        """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(
        model_name=st.session_state["model_name"],
        temperature=0,
        max_tokens=st.session_state["max_tokens"],
        openai_api_key=openai_api_key,
    )

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(
            search_type="mmr", search_kwargs={"k": st.session_state["k"], "lambda_mult": 0.25}
        ),
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )

    result = chain(question)
    return result


with st.container():
    st.subheader("Chat History")
    
    # Add styling for chat messages
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
        .question-input-label {
            font-size: 20px;
            color: #ff3633;
            margin-bottom: -30px;
        }
        div[data-testid="stTextInput"][aria-describedby="question_input"] input {
            font-size: 20px;
            color: #ff3633;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display chat history with new styling
    if st.session_state.get("chat_history", []):
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

    # Question input section
    st.markdown("""
        <p class="question-input-label"><strong>Enter your question:</strong></p>
    """, unsafe_allow_html=True)
    
    question = st.text_input("", key="question_input")
    
    if st.button("Get Answer"):
        if "db" not in st.session_state:
            st.error("Please process the Zotero library first.")
        else:
            with st.spinner("Generating answer..."):
                result = answer_question(question, st.session_state["db"])
            
            answer = result["answer"]
            sources = result["source_documents"]
            
            # Save to chat history
            st.session_state["chat_history"].append((question, answer, sources))
            st.subheader("Answer:")
            st.write(answer)
            st.markdown("---")
            
            if sources:
                st.subheader("Source Documents:")
                for index, doc in enumerate(sources, start=1):
                    st.write(f"{index}: {doc}")
                
                st.subheader("All relevant sources:")
                for source in set([doc.metadata["source"] for doc in sources]):
                    st.write(source)
    
    # Chat management buttons after Get Answer
    if st.session_state.get("chat_history", []):
        st.markdown("---")
        col1, col2 = st.columns([1, 6])
        
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state["chat_history"] = []
                st.rerun()
        
        with col2:
            # Prepare chat history for download with UTF-8 encoding
            chat_history_str = ""
            for i, (q, a, sources) in enumerate(st.session_state["chat_history"], 1):
                chat_history_str += f"\nQuestion {i}: {q}\n"
                chat_history_str += f"Answer: {a}\n"
                if sources:
                    chat_history_str += "Sources:\n"
                    for source in sources:
                        chat_history_str += f"- {source}\n"
                chat_history_str += "-" * 50 + "\n"
            
            # Convert to UTF-8 encoded bytes
            chat_history_bytes = io.BytesIO(chat_history_str.encode('utf-8'))
            
            st.download_button(
                "📥 Download Chat History",
                data=chat_history_bytes,
                file_name="chat_history.txt",
                mime="text/plain"
            )

