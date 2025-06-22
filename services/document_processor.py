import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
import docx
import pandas as pd

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_file(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process uploaded file and return chunks with metadata"""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path, filename)
        elif file_extension == '.txt':
            return self._process_txt(file_path, filename)
        elif file_extension in ['.docx', '.doc']:
            return self._process_docx(file_path, filename)
        elif file_extension == '.csv':
            return self._process_csv(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        chunks = []
        for i, doc in enumerate(documents):
            text_chunks = self.text_splitter.split_text(doc.page_content)
            for j, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'filename': filename,
                        'page': i + 1,
                        'chunk_id': j,
                        'source_type': 'pdf'
                    }
                })
        return chunks
    
    def _process_txt(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        text_chunks = self.text_splitter.split_text(content)
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                'text': chunk,
                'metadata': {
                    'filename': filename,
                    'chunk_id': i,
                    'source_type': 'txt'
                }
            })
        return chunks
    
    def _process_docx(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        doc = docx.Document(file_path)
        content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        text_chunks = self.text_splitter.split_text(content)
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                'text': chunk,
                'metadata': {
                    'filename': filename,
                    'chunk_id': i,
                    'source_type': 'docx'
                }
            })
        return chunks
    
    def _process_csv(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(file_path)
        content = df.to_string()
        
        text_chunks = self.text_splitter.split_text(content)
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                'text': chunk,
                'metadata': {
                    'filename': filename,
                    'chunk_id': i,
                    'source_type': 'csv'
                }
            })
        return chunks