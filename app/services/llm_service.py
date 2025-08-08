from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import get_logger
from app.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field  # <-- Import BaseModel and Field
import hashlib
import threading
import time

# NOTE: The changes in this file focus on latency reductions WITHOUT changing the public API.
# Key optimizations:
# 1. Reuse structured output model & prompt template (avoid rebuilding per call).
# 2. In‑memory caching for embeddings & one‑shot answers (same inputs => instant return).
# 3. Optional context trimming to cap per‑question context size (reduces tokens sent to LLM).
# 4. More compact system instructions & deterministic temperature.
# 5. Reduced string concatenations by using list joins.
# 6. Light defensive padding and logging only when needed (avoid chatty logs in hot path).

logger = get_logger(__name__)
settings = get_settings()

# --- DEFINE THE OUTPUT SCHEMA ---
# This class tells the LLM the exact JSON structure we want back.
class AnswerList(BaseModel):
    """A list of answers to the user's questions."""
    answers: List[str] = Field(description="A list of string answers, one for each question asked.")


class LLMService:
    _EMBED_CACHE_MAX = 10_000  # simple cap to avoid unbounded growth
    _ANSWER_CACHE_MAX = 2_000

    def __init__(self, max_context_chars: int = 6000):
        """
        max_context_chars: hard cap per context segment (post-trim) to bound tokens.
        """
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)

            # MODELS
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                task_type="retrieval_document"
            )
            self.generative_model = ChatGoogleGenerativeAI(
                model=settings.GENERATIVE_MODEL_NAME,
                temperature=0.1,  # Keep low for determinism & faster decoding
                convert_system_message_to_human=True
            )

            # STRUCTURED OUTPUT PIPE (pre-built for reuse)
            self._structured_model = self.generative_model.with_structured_output(AnswerList)

            # PROMPT TEMPLATE (static skeleton). We'll inject contexts & questions.
            # Variables: {contexts} {questions}
            base_template = (
                "You answer multiple questions strictly from their provided contexts.\n"
                "If an answer is missing in its context respond exactly: Information not available in the provided document.\n\n"
                "CONTEXTS:\n{contexts}\n\nQUESTIONS:\n{questions}\n"
            )
            self._prompt_template = ChatPromptTemplate.from_template(base_template)

            # CACHES (thread-safe)
            self._embed_cache_docs: Dict[str, List[float]] = {}
            self._embed_cache_query: Dict[str, List[float]] = {}
            self._answer_cache: Dict[str, Tuple[float, List[str]]] = {}
            self._lock = threading.RLock()
            self.max_context_chars = max_context_chars
            logger.info("Google AI services initialized (optimized mode).")
        except Exception as e:
            logger.error(f"Failed to configure Google AI Services: {e}")
            raise

    # --------------- INTERNAL UTILITIES ---------------
    def _hash(self, *parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode('utf-8', 'ignore'))
        return h.hexdigest()

    def _trim(self, text: str) -> str:
        if self.max_context_chars and len(text) > self.max_context_chars:
            # Keep head & tail portions if extremely long to preserve potential answer zones
            head = self.max_context_chars // 2
            tail = self.max_context_chars - head - 15
            return text[:head] + "\n...[TRIMMED]...\n" + text[-tail:]
        return text

    def _prune_cache(self, cache: Dict, max_size: int):
        # Simple FIFO pruning based on insertion order (Python 3.7+ dict preserves order)
        if len(cache) > max_size:
            remove_count = len(cache) - max_size
            for k in list(cache.keys())[:remove_count]:
                cache.pop(k, None)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """A wrapper to generate embeddings for a list of document chunks."""
        results: List[List[float]] = []
        to_compute: List[str] = []
        # Collect cached or mark for compute
        with self._lock:
            for t in texts:
                key = self._hash('doc', t)
                if key in self._embed_cache_docs:
                    results.append(self._embed_cache_docs[key])
                else:
                    results.append(None)  # placeholder
                    to_compute.append(t)
        if to_compute:
            computed = self.embedding_model.embed_documents(to_compute)
            iter_comp = iter(computed)
            with self._lock:
                for idx, t in enumerate(texts):
                    if results[idx] is None:
                        emb = next(iter_comp)
                        key = self._hash('doc', t)
                        self._embed_cache_docs[key] = emb
                        results[idx] = emb
                self._prune_cache(self._embed_cache_docs, self._EMBED_CACHE_MAX)
        return results

    def get_query_embedding(self, text: str) -> List[float]:
        """A wrapper to generate an embedding for a single query."""
        key = self._hash('query', text)
        with self._lock:
            if key in self._embed_cache_query:
                return self._embed_cache_query[key]
        emb = self.embedding_model.embed_query(text)
        with self._lock:
            self._embed_cache_query[key] = emb
            self._prune_cache(self._embed_cache_query, self._EMBED_CACHE_MAX)
        return emb

    async def get_all_answers_in_one_shot(self, question_context_pairs: List[Dict]) -> List[str]:
        """Generates all answers in a single LLM call for maximum speed."""
        # Build cache key (order-sensitive)
        cache_key_parts = []
        trimmed_pairs = []
        for i, pair in enumerate(question_context_pairs):
            context_trimmed = self._trim(pair.get('context', ''))
            q = pair.get('question', '')
            cache_key_parts.append(str(i) + '|' + q + '|' + context_trimmed)
            trimmed_pairs.append({'question': q, 'context': context_trimmed})
        cache_key = self._hash('answers', *cache_key_parts)

        with self._lock:
            cached = self._answer_cache.get(cache_key)
            if cached:
                return cached[1]

        # Assemble contexts & questions with minimal overhead
        contexts_fragments = [
            f"--- CONTEXT FOR QUESTION {i+1} ---\n{p['context']}" for i, p in enumerate(trimmed_pairs)
        ]
        questions_fragment = "\n".join([
            f"{i+1}. {p['question']}" for i, p in enumerate(trimmed_pairs)
        ])

        chain = self._prompt_template | self._structured_model

        # Execute
        try:
            response_object = await chain.ainvoke({
                'contexts': "\n\n".join(contexts_fragments),
                'questions': questions_fragment
            })
            answers = response_object.answers
            # Defensive length normalization
            if len(answers) != len(question_context_pairs):
                logger.warning(
                    "Answer count mismatch (%d vs %d). Padding.",
                    len(answers), len(question_context_pairs)
                )
                while len(answers) < len(question_context_pairs):
                    answers.append("Error processing this question.")
            with self._lock:
                self._answer_cache[cache_key] = (time.time(), answers)
                self._prune_cache(self._answer_cache, self._ANSWER_CACHE_MAX)
            return answers
        except Exception as e:
            logger.error(f"One-shot LLM call failed: {e}")
            return ["Error processing this question." for _ in question_context_pairs]

llm_service = LLMService()
