import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import get_settings
from app.utils.logger import get_logger
from dotenv import load_dotenv
load_dotenv()
logger = get_logger(__name__)
settings = get_settings()
import os

# ... other imports
from app.core.config import get_settings  # <-- Import the settings getter

# Call the function once at the module level to get the loaded settings
settings = get_settings()

class LLMService:
    def __init__(self):
        try:
            # Use the setting directly. No need to load .env here.
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                # Use the model name from the settings
                model=settings.EMBEDDING_MODEL_NAME,
                task_type="retrieval_document"
            )
            self.generative_model = ChatGoogleGenerativeAI(
                # Use the generative model name from the settings
                model=settings.GENERATIVE_MODEL_NAME,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            # ... rest of the code
# ...
            logger.info("Google AI Services configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google AI Services: {e}")
            raise

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts."""
        return self.embedding_model.embed_documents(texts)

    def get_query_embedding(self, text: str) -> list[float]:
        """Generates embedding for a single query string."""
        return self.embedding_model.embed_query(text)

    async def get_answer_from_context(self, question: str, context: str) -> str:
        """
        Generates a concise answer to a question based on the provided context.
        This uses LangChain Expression Language (LCEL) for a streamlined process.
        """
        prompt_template = """
        You are an expert Q&A system for insurance and legal documents. Your answers must be precise, factual, and strictly based on the provided context.

        Answer the user's question using ONLY the context below. Do not use any outside knowledge.
        If the context does not contain the answer, state that clearly: "The provided policy document does not contain information on this topic."
        
        Provide a direct answer to the question. Avoid conversational fluff.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Chain the components together
        chain = prompt | self.generative_model | StrOutputParser()
        
        logger.info(f"Invoking LLM chain for question: '{question[:50]}...'")
        
        response = await chain.ainvoke({"context": context, "question": question})
        
        return response.strip()
    
    # In app/services/llm_service.py

# ... (rest of the file is the same)

    async def get_answer_from_context(self, question: str, context: str) -> str:
        """
        Generates a concise and analytical answer to a question based on the provided context.
        This advanced prompt encourages citation and admitting when the answer is not present.
        """
        prompt_template = """
        You are a highly intelligent and meticulous Q&A system for legal and insurance documents.
        Your task is to provide a clear, accurate, and well-supported answer based *only* on the provided context from various sources.

        Follow these rules strictly:
        1.  **Synthesize Information:** Do not just copy-paste chunks. Synthesize the information from the context to form a coherent answer.
        2.  **Cite Your Sources:** For every piece of information you use, you MUST cite the source document it came from. For example: "The grace period is 30 days (Source: policy_document_1.pdf)."
        3.  **Handle Contradictions:** If you find conflicting information across different sources, point out the contradiction.
        4.  **Admit Ignorance:** If the answer is not found in the provided context, you MUST state: "Based on the provided documents, there is no information available to answer this question." Do not make any assumptions or use outside knowledge.

        CONTEXT:
        {context}
        
        QUESTION:
        "{question}"
        
        Based on the rules above, provide your answer:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain = prompt | self.generative_model | StrOutputParser()
        
        logger.info(f"Invoking LLM chain with advanced prompt for question: '{question[:50]}...'")
        
        response = await chain.ainvoke({"context": context, "question": question})
        
        return response.strip()

# Singleton instance
llm_service = LLMService()

