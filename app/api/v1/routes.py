import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from .models import QueryRequest, QueryResponse
from app.api.deps import verify_token
from app.services.document_service import document_service
from app.services.llm_service import llm_service
from app.services.vector_db_service import vector_db_service
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post(
    "/hackrx/run",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a document with a competition-optimized RAG pipeline",
    dependencies=[Depends(verify_token)]
)
async def run_submission(request: QueryRequest):
    document_url = str(request.documents)
    logger.info(f"Processing request for document: {document_url} with high-speed RAG")

    try:
        # --- Ingestion (This part is already fast) ---
        namespace = document_service.generate_document_namespace(document_url)
        content, file_type = await document_service.get_document_from_url(document_url)
        full_text = document_service.parse_document(content, file_type)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract any text from the document.")

        chunks = document_service.chunk_text(full_text)
        chunk_texts = [chunk['text'] for chunk in chunks]

        embeddings = llm_service.get_embeddings(chunk_texts)
        vectors_to_upsert = [{'id': f'chunk_{i}', 'values': embeddings[i], 'metadata': {'text': chunk_texts[i]}} for i in range(len(chunk_texts))]
        vector_db_service.upsert(vectors=vectors_to_upsert, namespace=namespace)
        logger.info(f"Document upserted into temporary namespace: {namespace}")

        # --- High-Speed Retrieval and Generation ---
        async def process_question(question: str):
            logger.info(f"Processing question: '{question}'")
            query_embedding = llm_service.get_query_embedding(question)

            # 1. Wider Net Retrieval: Get more potential contexts in one shot.
            # We fetch 7 chunks to maximize our chances of finding the right answer.
            retrieved_matches = vector_db_service.query(
                query_embedding=query_embedding,
                top_k=7,
                namespace=namespace
            )

            # 2. No-Delay Context Stuffing: Immediately build the context.
            context = "\n---\n".join([match['metadata']['text'] for match in retrieved_matches])
            
            # 3. Fast Generation: Pass to the competition-tuned LLM service.
            return await llm_service.get_answer_from_context(question, context)

        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"An unexpected error occurred in the RAG pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERRO, detail="An internal server error occurred.")