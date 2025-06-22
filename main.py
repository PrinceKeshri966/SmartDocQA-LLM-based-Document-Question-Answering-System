import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

from app.models import QuestionRequest, Answer, DocumentInfo
from app.services.document_processor import DocumentProcessor
from app.services.qa_service import QAService
from app.utils.vector_store import VectorStore

# Load environment variables
load_dotenv()

app = FastAPI(title="SmartDocQA API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = VectorStore(embeddings)
document_processor = DocumentProcessor()
qa_service = QAService(vector_store, OPENAI_API_KEY)

# Global variables to track documents
uploaded_documents = []

@app.get("/")
async def root():
    return {"message": "SmartDocQA API is running!"}

@app.post("/upload/", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Create documents directory if it doesn't exist
        os.makedirs("documents", exist_ok=True)
        
        # Save uploaded file
        file_path = f"documents/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        chunks = document_processor.process_file(file_path, file.filename)
        
        # Extract texts and metadata for vector store
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add to vector store
        vector_store.add_documents(texts, metadatas)
        
        # Track uploaded document
        doc_info = DocumentInfo(
            filename=file.filename,
            chunks=len(chunks),
            status="processed"
        )
        uploaded_documents.append(doc_info)
        
        return doc_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/question/", response_model=Answer)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents"""
    try:
        result = qa_service.answer_question(request.question, request.top_k)
        
        return Answer(
            question=request.question,
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/documents/", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    return uploaded_documents

@app.delete("/documents/")
async def clear_documents():
    """Clear all uploaded documents and reset vector store"""
    try:
        # Clear documents directory
        if os.path.exists("documents"):
            shutil.rmtree("documents")
        
        # Clear vector database
        if os.path.exists("vector_db"):
            shutil.rmtree("vector_db")
        
        # Reinitialize vector store
        global vector_store
        vector_store = VectorStore(embeddings)
        
        # Clear document list
        uploaded_documents.clear()
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)