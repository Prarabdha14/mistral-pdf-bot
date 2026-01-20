import os
import getpass

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP API KEY ---
if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter Mistral API Key: ")

# --- 2. AUTOMATIC PDF DETECTION ---
print("--- Step 1: Finding PDF ---")
data_folder = "./data"

# Check if folder exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"Created folder '{data_folder}'. Please put a PDF inside and run again.")
    exit()

# Find the first PDF file in the directory
pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]

if not pdf_files:
    print("Error: No PDF file found in the 'data' folder!")
    exit()

pdf_filename = pdf_files[0]
pdf_path = os.path.join(data_folder, pdf_filename)
print(f" - Found file: {pdf_filename}")

# --- 3. PARSING & CHUNKING  ---
print("--- Step 2: Parsing & Chunking ---")
try:
    # Parsing: Read the file
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f" - Parsed {len(docs)} pages.")
    
    # Chunking: Split into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f" - Created {len(splits)} text chunks.")
except Exception as e:
    print(f"Error processing PDF: {e}")
    exit()

# --- 4. EMBEDDING & INDEXING ---
print("--- Step 3: Creating Search Index (FAISS) ---")
# Using Mistral's embedding model to convert text to numbers
embeddings = MistralAIEmbeddings(model="mistral-embed")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# --- 5. THE BRAIN SETUP ---
print("--- Step 4: Initializing Mistral AI ---")
# Using the fast 'mistral-tiny' model
llm = ChatMistralAI(model="mistral-tiny") 

# Define the "System Prompt" (Rules for the bot)
system_prompt = (
    "You are a helpful assistant. Use the context below to answer the question. "
    "If the answer is not in the context, say 'I do not see that information in the document.' "
    "Keep your answers concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Connect everything into a chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 6. CHAT LOOP ---
print(f"\n--- Chatting with {pdf_filename} (Type 'quit' to exit) ---")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    try:
        response = rag_chain.invoke({"input": user_input})
        print(f"Bot: {response['answer']}")
    except Exception as e:
        print(f"Error: {e}")