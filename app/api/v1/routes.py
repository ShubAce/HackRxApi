import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from .models import QueryRequest, QueryResponse
from app.api.deps import verify_token
from app.services.document_service import document_service
from app.services.llm_service import llm_service
from app.services.vector_db_service import vector_db_service
from app.services.retrieval_service import advanced_retriever # <-- IMPORT NEW SERVICE
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post(
    "/hackrx/run",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a document and answer questions with an advanced RAG pipeline",
    dependencies=[Depends(verify_token)]
)
async def run_submission(request: QueryRequest):
    document_url = str(request.documents)
    logger.info(f"Processing request for document: {document_url} with advanced RAG")

    try:
        # --- Ingestion (same as before) ---
        namespace = document_service.generate_document_namespace(document_url)
        content, file_type = await document_service.get_document_from_url(document_url)
        full_text = document_service.parse_document(content, file_type)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract any text from the document.")

        chunks = document_service.chunk_text(full_text)
        
        # Prepare for both vector and keyword search
        all_chunk_data = []
        for i, chunk in enumerate(chunks):
            all_chunk_data.append({
                'id': f'chunk_{i}',
                'text': chunk['text']
            })
        
        # Only embed and upsert if not already done (in a real system, you'd check a database)
        # For this example, we re-process each time as per the original logic.
        chunk_texts = [c['text'] for c in all_chunk_data]
        embeddings = llm_service.get_embeddings(chunk_texts)
        vectors_to_upsert = [{'id': all_chunk_data[i]['id'], 'values': embeddings[i], 'metadata': {'text': chunk_texts[i]}} for i in range(len(chunk_texts))]
        vector_db_service.upsert(vectors=vectors_to_upsert, namespace=namespace)

        # --- Advanced Retrieval and Generation ---
        async def process_question(question: str):
            # 1. Use the AdvancedRetriever
            retrieved_chunks = advanced_retriever.retrieve(
                question=question,
                all_chunks=all_chunk_data,
                namespace=namespace,
                top_k=5
            )

            # 2. Build context with source information (source is the URL itself in this case)
            context = "\n---\n".join([f"Source: {document_url.split('/')[-1].split('?')[0]}\nContent: {chunk['metadata']['text']}" for chunk in retrieved_chunks])

            if not context:
                return "Based on the provided documents, there is no information available to answer this question."

            # 3. Generate answer with the advanced prompt
            return await llm_service.get_answer_from_context(question, context)

        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"An unexpected error occurred in advanced RAG pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")