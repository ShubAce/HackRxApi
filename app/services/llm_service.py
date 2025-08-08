from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import get_logger
from app.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from typing import List, Dict
from pydantic import BaseModel, Field # <-- Import BaseModel and Field

logger = get_logger(__name__)
settings = get_settings()

# --- DEFINE THE OUTPUT SCHEMA ---
# This class tells the LLM the exact JSON structure we want back.
class AnswerList(BaseModel):
    """A list of answers to the user's questions."""
    answers: List[str] = Field(description="A list of string answers, one for each question asked.")


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

        # The prompt is simplified as the model now knows the exact output format.
        one_shot_prompt = f"""
        You are a high-speed Q&A machine. Answer all questions based ONLY on the provided context for each question.
        If the answer is not in a question's context, respond for that question with the exact phrase: `Information not available in the provided document.`
        
        CONTEXTS:
        {prompt_context}

        QUESTIONS:
        {questions_list}
        """
        
        # --- APPLY THE FIX HERE ---
        # We pass our Pydantic class, not the type hint.
        model_with_json = self.generative_model.with_structured_output(AnswerList)
        
        chain = ChatPromptTemplate.from_template(one_shot_prompt) | model_with_json
        
        logger.info(f"Invoking ONE-SHOT LLM chain for {len(question_context_pairs)} questions.")
        
        try:
            # The chain will now return an instance of our AnswerList class
            response_object = await chain.ainvoke({})
            
            # We extract the list of answers from the object
            response_list = response_object.answers

            if len(response_list) != len(question_context_pairs):
                logger.warning(f"LLM returned {len(response_list)} answers, expected {len(question_context_pairs)}. Padding with default.")
                while len(response_list) < len(question_context_pairs):
                    response_list.append("Error processing this question.")
            return response_list
        except Exception as e:
            logger.error(f"One-shot LLM call failed: {e}")
            return ["Error processing this question." for _ in question_context_pairs]

llm_service = LLMService()
