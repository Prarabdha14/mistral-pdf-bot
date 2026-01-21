import streamlit as st
import os
import tempfile

# --- STANDARD IMPORTS ---
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- NEW: UNSTRUCTURED IMPORT ---
from unstructured.partition.pdf import partition_pdf

st.set_page_config(page_title="Mistral RAG Bot", page_icon="🤖")
st.title("🤖 Chat with PDFs (Powered by Unstructured)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Mistral API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

# --- HELPER: FORMAT DOCS ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- NEW LOGIC: READ PDF WITH UNSTRUCTURED ---
def get_pdf_text(uploaded_file):
    # 1. Save uploaded file to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # 2. Use Unstructured to read it (The Manager's Choice)
        # strategy="hi_res" is slower but handles tables/layouts much better
        elements = partition_pdf(
            filename=tmp_path,
            strategy="hi_res", 
            infer_table_structure=True
        )
        
        # 3. Combine all the elements into one big text
        # (We separate chunks with double newlines to keep paragraphs distinct)
        full_text = "\n\n".join([el.text for el in elements])
        
        # Clean up the temp file
        os.remove(tmp_path)
        
        return [Document(page_content=full_text, metadata={"source": uploaded_file.name})]

    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise e

# --- MAIN APP LOGIC ---
if api_key and uploaded_file:
    os.environ["MISTRAL_API_KEY"] = api_key

    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF with Unstructured... (This handles tables & columns!)"):
            try:
                # 1. Load Text
                docs = get_pdf_text(uploaded_file)
                
                # 2. Split
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)
                
                # 3. Embed
                embeddings = MistralAIEmbeddings(model="mistral-embed")
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                st.session_state.vectorstore = vectorstore
                st.success(f"Successfully processed {len(splits)} chunks!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Build the chain
    if "vectorstore" in st.session_state:
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatMistralAI(model="mistral-tiny")
        
        template = """Answer based strictly on the context below.
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        if user_input := st.chat_input("Ask a question about the paper..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(user_input)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})