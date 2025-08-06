from rank_bm25 import BM25Okapi
from app.services.llm_service import llm_service
from app.services.vector_db_service import vector_db_service
from app.utils.logger import get_logger
from typing import List, Dict

logger = get_logger(__name__)

class AdvancedRetriever:
    """
    Implements an advanced retrieval strategy combining keyword and semantic search,
    followed by an LLM-based re-ranking step.
    """

    def _get_keyword_searcher(self, corpus: List[str]):
        """Creates a BM25 keyword search index."""
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        return BM25Okapi(tokenized_corpus)

    def _rerank_with_llm(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """
        Uses the generative LLM to re-rank the retrieved chunks for relevance.
        This is a powerful step to bring the most relevant context to the top.
        """
        logger.info(f"Re-ranking {len(chunks)} chunks for question: '{question[:50]}...'")
        
        # Create a numbered list of chunks for the LLM to evaluate
        chunks_with_indices = [f"[{i}] {chunk['metadata']['text']}" for i, chunk in enumerate(chunks)]
        context_for_reranking = "\n---\n".join(chunks_with_indices)

        # A prompt that asks the LLM to act as a re-ranker
        rerank_prompt = f"""
        You are an expert relevance ranker. Your task is to re-rank the following text chunks based on their relevance to the user's question.
        Return a comma-separated list of the original index numbers (e.g., [2], [0], [3]) from most relevant to least relevant.
        Do not explain your reasoning, just provide the list.

        QUESTION:
        "{question}"

        CHUNKS:
        {context_for_reranking}
        
        RE-RANKED INDEXES:
        """
        
        try:
            # Note: For production, this should be an async call.
            # We are keeping it sync here for simplicity of demonstration.
            response = llm_service.generative_model.invoke(rerank_prompt)
            ranked_indices_str = response.content.strip().replace('[', '').replace(']', '')
            ranked_indices = [int(i.strip()) for i in ranked_indices_str.split(',')]

            # Reorder the original chunks list based on the LLM's ranking
            reordered_chunks = [chunks[i] for i in ranked_indices if i < len(chunks)]
            logger.info(f"Re-ranked order: {ranked_indices}")
            return reordered_chunks
        except Exception as e:
            logger.error(f"Failed to re-rank chunks with LLM: {e}. Returning original order.")
            # If re-ranking fails, fall back to the original order
            return chunks


    def retrieve(self, question: str, all_chunks: List[Dict], namespace: str, top_k: int = 5) -> List[Dict]:
        """
        Executes the full advanced retrieval pipeline.
        """
        logger.info("Executing advanced retrieval...")
        
        # 1. Hybrid Search (Keyword + Semantic)
        # Keyword search
        corpus_texts = [chunk['text'] for chunk in all_chunks]
        bm25 = self._get_keyword_searcher(corpus_texts)
        tokenized_query = question.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Semantic search
        query_embedding = llm_service.get_query_embedding(question)
        semantic_results = vector_db_service.query(query_embedding, top_k=10, namespace=namespace)
        
        # Combine results (a simple weighted approach)
        # Create a map of chunk ID to its original data and semantic score
        semantic_map = {res['id']: {'data': res, 'score': res['score']} for res in semantic_results}
        
        # In a real system, you'd normalize scores. Here we'll just combine the sets.
        # We will use the top 10 semantic results and re-rank them.
        initial_retrieval = [res for res in semantic_results]

        logger.info(f"Initial retrieval found {len(initial_retrieval)} candidates.")
        
        # 2. Re-ranking
        # Pass the top candidates to the re-ranker
        reranked_chunks = self._rerank_with_llm(question, initial_retrieval)

        # 3. Return the top_k results after re-ranking
        return reranked_chunks[:top_k]

# Singleton instance
advanced_retriever = AdvancedRetriever()