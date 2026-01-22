## Phase 1: The Starting Point
Build a basic chatbot to answer questions from standard text PDFs.
Tool:`PyPDFLoader`

## pseudocode
BEGIN
    INPUT pdf_file
    
    # Attempt to read text directly from PDF layer
    extracted_text = PyPDFLoader(pdf_file).load()
    
    IF extracted_text IS NOT EMPTY THEN
        Break text into 1000-character chunks
        Convert chunks to Vector Embeddings (Mistral-Embed)
        Store in FAISS Database
        
        # Retrieval Loop
        WHILE User asks Question:
            Find most similar chunks in FAISS
            Send (Question + Chunks) to LLM
            RETURN Answer
    ELSE
        RETURN Error: "Empty Document"
    END IF
END

## Phase 2: OCR Integration
Enable the bot to read scanned documents and images.
Tool: `Tesseract OCR` + `pdf2image`

## pseudocode
BEGIN
    INPUT pdf_file
    
    # First, try standard reading
    text = StandardLoader(pdf_file)
    
    # Heuristic Check: Did we get enough text?
    IF len(text) < 50 characters THEN
        PRINT "Detected Scanned Document. Switching to OCR..."
        
        # 1. Convert PDF pages to Images
        images = Convert_PDF_To_Images(pdf_file)
        
        # 2. Extract text from each image pixel-by-pixel
        full_text = ""
        FOR image IN images:
             page_text = Tesseract_OCR(image)
             full_text += page_text
    END IF
    
    Process(full_text) -> Vector DB -> Chat
END

## Phase 3: Intelligent Layout Parsing 
Correctly interpret structure (Tables, Columns, Headers)
Tool: `Unstructured`

## pseudocode
BEGIN
    INPUT pdf_file
    
    # 1. Structural Partitioning (Strategy: High Resolution)
    # This detects bounding boxes around tables, images, and headers
    elements = Unstructured.partition_pdf(
        filename=pdf_file,
        strategy="hi_res",
        infer_table_structure=TRUE
    )
    
    full_text = ""
    
    # 2. Intelligent Categorization
    FOR element IN elements:
        IF element.type == "Table":
            # Preserve table structure (keep rows/cols aligned)
            formatted_table = ConvertToMarkdown(element)
            full_text += formatted_table
            
        ELSE IF element.type == "Title":
            # Add header formatting for context
            full_text += "# " + element.text
            
        ELSE:
            # Standard text
            full_text += element.text
            
    # 3. RAG Pipeline
    Chunks = Split(full_text)
    Embeddings = Mistral(Chunks)
    VectorStore = FAISS(Embeddings)
    
    OUTPUT "Ready to Chat"
END

## Phase 4: Hybrid Architecture (Local Embeddings + Cloud Chat)
 Optimize for cost and scalability while maintaining high intelligence for answers.
 Unstructured (Parser) -> LangChain (Chunker) -> Hugging Face (Local Embedder) -> Milvus (DB) -> Mistral (Cloud LLM)

## pseudocode
BEGIN
    INPUT pdf_file
    
    # 1. Parse & Chunk (Standard)
    raw_text = Unstructured.partition_pdf(pdf_file)
    chunks = RecursiveCharacterTextSplitter(raw_text, size=1000)
    
    # 2. Local Embedding (NEW)
    # Uses CPU-optimized model 'all-MiniLM-L6-v2' ~80MB size
    vector_list = []
    FOR chunk IN chunks:
        vector = HuggingFaceModel.encode(chunk) # Returns list of 384 floats
        vector_list.append(vector)
        
    # 3. Persistent Storage (NEW)
    # Connect to Docker Container
    DB_Connection = Milvus.connect(host="localhost", port="19530")
    
    # Insert vectors into collection
    DB_Connection.insert(collection="pdf_chat", vectors=vector_list)
    
    OUTPUT "Stored Successfully in Docker"
END

## Retrieval Algorithm 
BEGIN
    INPUT User_Question
    
    # 1. Embed Question Locally
    query_vector = HuggingFaceModel.encode(User_Question)
    
    # 2. Vector Search (Milvus)
    # Finds top 4 matches based on Cosine Similarity
    matched_chunks = Milvus.search(query_vector, k=4)
    
    # 3. Cloud Generation
    # We only send the small matched text to the cloud, not the whole PDF
    Prompt = "Context: " + matched_chunks + " Question: " + User_Question
    Answer = MistralAPI.generate(Prompt)
    
    RETURN Answer
END