import streamlit as st
import os
import tempfile

# --- IMPORTS ---
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Mistral LCEL Bot", page_icon="⚡")
st.title("Chat with PDF")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Mistral API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

# --- HELPER FUNCTION: FORMAT DOCS ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- MAIN LOGIC ---
if api_key and uploaded_file:
    # Force Env Var
    os.environ["MISTRAL_API_KEY"] = api_key

    # 1. LOAD & CHUNK (Cached in Session)
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Load
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                os.remove(tmp_path)
                
                # Split
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)
                
                # Embed
                embeddings = MistralAIEmbeddings(model="mistral-embed")
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                st.session_state.vectorstore = vectorstore
                st.success(f"Ready! Processed {len(splits)} chunks.")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

    # 2. BUILD THE CHAIN (LCEL STYLE)
    # This is the modern way that fixes the "NoneType" error
    try:
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatMistralAI(model="mistral-tiny")
        
        # Define Prompt
        template = """You are a helpful assistant. Use the context below to answer the question.
        If you don't know, say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # --- THE FIX: MANUAL CHAIN CONSTRUCTION ---
        # We use the pipe '|' syntax instead of create_retrieval_chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 3. CHAT INTERFACE
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Invoke the LCEL chain
                    response = rag_chain.invoke(user_input)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Chain Error: {e}")

else:
    st.info("👈 Enter API Key and Upload PDF to start")