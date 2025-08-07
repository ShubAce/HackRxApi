# In app/services/llm_service.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.utils.logger import get_logger
from app.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# ... (the top part of your file remains the same) ...

class LLMService:
    # ... (your __init__ and get_embeddings methods remain the same) ...

    async def get_answer_from_context(self, question: str, context: str) -> str:
        """
        Generates a direct, competition-optimized answer.
        This prompt forces speed, accuracy, and honesty.
        """
        prompt_template = """
        You are a high-speed Q&A machine for a competition. Your goal is to answer questions based *ONLY* on the provided text.

        RULES:
        1. Read the 'QUESTION' and the 'CONTEXT' below.
        2. Provide a direct and concise answer to the 'QUESTION' using only the information in the 'CONTEXT'.
        3. If the answer is not present in the 'CONTEXT', you MUST respond with the exact phrase: `Information not available in the provided document.`
        4. Do not use any prior knowledge. Do not add any conversational fluff, explanations, or introductory phrases.

        CONTEXT:
        ---
        {context}
        ---
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # This chain is optimized for speed
        chain = prompt | self.generative_model | StrOutputParser()
        
        logger.info(f"Invoking competition-optimized LLM chain for question: '{question[:50]}...'")
        
        response = await chain.ainvoke({"context": context, "question": question})
        
        return response.strip()

llm_service = LLMService()