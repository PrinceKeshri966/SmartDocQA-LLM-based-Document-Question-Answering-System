from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.utils.vector_store import VectorStore

class QAService:
    def __init__(self, vector_store: VectorStore, openai_api_key: str):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )
        
        # Create QA prompt template
        self.qa_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: """
        
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=self.qa_prompt
        )
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Generate answer for a given question"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question, k=top_k)
        
        if not relevant_docs:
            return {
                'answer': "I don't have any relevant documents to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate answer
        answer = self.qa_chain.run(context=context, question=question)
        
        # Extract sources
        sources = []
        for doc in relevant_docs:
            source_info = f"{doc['metadata']['filename']}"
            if 'page' in doc['metadata']:
                source_info += f" (Page {doc['metadata']['page']})"
            sources.append(source_info)
        
        # Calculate confidence based on similarity scores
        avg_distance = sum([doc['distance'] for doc in relevant_docs]) / len(relevant_docs)
        confidence = max(0, 1 - (avg_distance / 2))  # Simple confidence metric
        
        return {
            'answer': answer.strip(),
            'sources': list(set(sources)),  # Remove duplicates
            'confidence': round(confidence, 2)
        }