from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.utils.logger import get_logger
from app.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from typing import List, Dict

logger = get_logger(__name__)
settings = get_settings()

class LLMService:
    def __init__(self):
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                task_type="retrieval_document"
            )
            self.generative_model = ChatGoogleGenerativeAI(
                model=settings.GENERATIVE_MODEL_NAME,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            logger.info("Google AI Services configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google AI Services: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """A wrapper to generate embeddings for a list of document chunks."""
        return self.embedding_model.embed_documents(texts)

    def get_query_embedding(self, text: str) -> List[float]:
        """A wrapper to generate an embedding for a single query."""
        return self.embedding_model.embed_query(text)

    async def get_all_answers_in_one_shot(self, question_context_pairs: List[Dict]) -> List[str]:
        """Generates all answers in a single LLM call for maximum speed."""
        prompt_context = ""
        for i, pair in enumerate(question_context_pairs):
            prompt_context += f"--- CONTEXT FOR QUESTION {i+1} ---\n{pair['context']}\n\n"

        questions_list = "\n".join([f"{i+1}. {pair['question']}" for i, pair in enumerate(question_context_pairs)])

        one_shot_prompt = f"""
        You are a high-speed Q&A machine. You will be given a list of questions and a block of context corresponding to each question.
        Your task is to answer all questions in order.

        RULES:
        1. For each question, use ONLY its corresponding context block to formulate the answer.
        2. If the answer for a question is not in its context, you MUST use the exact phrase: `Information not available in the provided document.`
        3. Your final output MUST be a single, valid JSON list of strings, where each string is the answer to the corresponding question. The list must have exactly {len(question_context_pairs)} items.
        
        EXAMPLE OUTPUT FORMAT:
        ["Answer for question 1", "Answer for question 2", "Information not available in the provided document."]

        CONTEXTS:
        {prompt_context}

        QUESTIONS:
        {questions_list}

        JSON OUTPUT:
        """
        
        model_with_json = self.generative_model.with_structured_output(List[str])
        chain = ChatPromptTemplate.from_template(one_shot_prompt) | model_with_json
        
        logger.info(f"Invoking ONE-SHOT LLM chain for {len(question_context_pairs)} questions.")
        
        try:
            response_list = await chain.ainvoke({})
            if len(response_list) != len(question_context_pairs):
                logger.warning(f"LLM returned {len(response_list)} answers, expected {len(question_context_pairs)}. Padding with default.")
                while len(response_list) < len(question_context_pairs):
                    response_list.append("Error processing this question.")
            return response_list
        except Exception as e:
            logger.error(f"One-shot LLM call failed: {e}")
            return ["Error processing this question." for _ in question_context_pairs]

llm_service = LLMService()
