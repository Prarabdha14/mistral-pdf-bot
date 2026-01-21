Phase 1: ChatBot development
Stack: Python, LangChain, FAISS (Vector DB), Mistral AI (LLM).
Architecture: Implemented standard RAG pipeline: Load PDF -> Chunk -> Embed -> Retrieve -> Generate.
Issue: Standard PyPDFLoader failed on scanned documents (images).

Phase 2: OCR Integration (Tesseract)
Created feature branch feat/OCR.
Integrated pytesseract and pdf2image to handle scanned PDFs.
Logic: Added fallback mechanism—if text extraction length is <100 chars, trigger OCR pipeline.
Limitation: Tesseract struggled with multi-column layouts and tables (output was unstructured text).

Phase 3: Advanced Layout Parsing (MinerU vs. Unstructured)
Goal: Improve extraction of tables and engineering math formulas.

Attempt 1 (MinerU):
Tried setting up OpenDataLab's MinerU.
Blocker: Required downloading 10GB+ of local models. Too heavy for local development on MacBook.

Attempt 2 (Unstructured):
Switched to unstructured(pdf) library.
Benefit: Better layout detection (headers vs. text) and table extraction without the massive storage footprint.
Implementation: Updated app.py to use partition_pdf with strategy="hi_res".