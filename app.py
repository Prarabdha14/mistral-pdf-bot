import os

# --- 1. MAC CRASH FIX ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. IMPORTS ---
import streamlit as st
import tempfile
from dotenv import load_dotenv

# NEW IMPORTS FOR MIGRATION
from langchain_huggingface import HuggingFaceEmbeddings # <--- Replaces MistralAIEmbeddings
from langchain_milvus import Milvus # <--- Replaces FAISS

from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

load_dotenv()

st.set_page_config(page_title="Milvus RAG Bot", page_icon="💾")
st.title("🤖 Chat with PDFs (Milvus + HuggingFace)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    # We still need Mistral Key for the CHAT part, but not for embeddings
    env_key = os.getenv("MISTRAL_API_KEY")
    if env_key:
        st.success("✅ Mistral Key Loaded")
        api_key = env_key
    else:
        api_key = st.text_input("Mistral API Key", type="password")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # Add a reset button that clears Milvus collection if needed
    if st.button("Clear Database"):
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore.delete(ids=[]) # Logic to clear would go here
            st.success("Memory Cleared!")
            st.session_state.clear()
            st.rerun()

# --- PDF READING LOGIC (Unstructured) ---
def get_pdf_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        elements = partition_pdf(filename=tmp_path, strategy="hi_res", infer_table_structure=True)
        full_text = "\n\n".join([el.text for el in elements])
        os.remove(tmp_path)
        return [Document(page_content=full_text, metadata={"source": uploaded_file.name})]
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise e

# --- MAIN LOGIC ---
if api_key and uploaded_file:
    os.environ["MISTRAL_API_KEY"] = api_key

    # Initialize Vector Store if not done
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF & Connecting to Milvus..."):
            try:
                # 1. Load Text
                docs = get_pdf_text(uploaded_file)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)
                
                # 2. Embeddings (Changed to HuggingFace)
                # This runs LOCALLY on your Mac, free of charge.
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # 3. Vector Store (Changed to Milvus)
                # Connects to the Docker container we started
                vectorstore = Milvus.from_documents(
                    splits,
                    embeddings,
                    connection_args={"host": "127.0.0.1", "port": "19530"},
                    collection_name="pdf_chat_collection",
                    drop_old=True # Resets the DB for every new PDF (optional)
                )
                
                st.session_state.vectorstore = vectorstore
                st.success(f"Stored {len(splits)} chunks in Milvus!")
                
            except Exception as e:
                st.error(f"Error connecting to Milvus: {e}")
                st.stop()

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "vectorstore" in st.session_state:
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatMistralAI(model="mistral-tiny")
        
        template = """Answer based on context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        if user_input := st.chat_input("Ask about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(user_input)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})