import asyncio
from fastapi import APIRouter, Depends, HTTPException, status # 'status' is already imported
from .models import QueryRequest, QueryResponse
from app.api.deps import verify_token
from app.services.document_service import document_service
from app.services.llm_service import llm_service
from app.services.vector_db_service import vector_db_service
from app.utils.logger import get_logger
# In app/api/v1/routes.py
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, Request # Import Request
from .models import QueryRequest, QueryResponse
# ... (other imports) ...

router = APIRouter()
logger = get_logger(__name__)

@router.post(
    "/hackrx/run",
    # ...
)
async def run_submission(fastapi_request: Request, request: QueryRequest, dependencies=Depends(verify_token)): # Add fastapi_request
    document_url = str(request.documents)
    logger.info(f"Processing request for document: {document_url} with high-speed RAG")

    try:
        namespace = document_service.generate_document_namespace(document_url)
        
        # --- CACHING LOGIC ---
        # Check if we have already processed this document's namespace
        if namespace not in fastapi_request.app.state.PROCESSED_DOCUMENT_CACHE:
            logger.info(f"CACHE MISS: Namespace '{namespace}' not found. Starting full ingestion...")
            
            # --- Ingestion (only runs on cache miss) ---
            content, file_type = await document_service.get_document_from_url(document_url)
            full_text = document_service.parse_document(content, file_type)
            if not full_text.strip():
                raise HTTPException(status_code=400, detail="Failed to extract any text from the document.")

            chunks = document_service.chunk_text(full_text)
            chunk_texts = [chunk['text'] for chunk in chunks]

            embeddings = llm_service.get_embeddings(chunk_texts)
            vectors_to_upsert = [{'id': f'chunk_{i}', 'values': embeddings[i], 'metadata': {'text': chunk_texts[i]}} for i in range(len(chunk_texts))]
            
            vector_db_service.upsert(vectors=vectors_to_upsert, namespace=namespace)
            
            # Add the processed namespace to the cache
            fastapi_request.app.state.PROCESSED_DOCUMENT_CACHE.add(namespace)
            logger.info(f"Ingestion complete. Namespace '{namespace}' added to cache.")
        else:
            logger.info(f"CACHE HIT: Namespace '{namespace}' found. Skipping ingestion.")

        # --- High-Speed Retrieval and Generation (runs every time) ---
        # In routes.py, this would replace the original 'process_question' and 'gather' logic

        # --- "One-Shot" Retrieval and Generation ---
        
        # 1. First, retrieve context for ALL questions
        question_context_pairs = []
        for question in request.questions:
            query_embedding = llm_service.get_query_embedding(question)
            retrieved_matches = vector_db_service.query(
                query_embedding=query_embedding,
                top_k=5,
                namespace=namespace
            )
            context = "\n---\n".join([match['metadata']['text'] for match in retrieved_matches])
            question_context_pairs.append({"question": question, "context": context})

        # 2. Make the single, powerful LLM call
        answers = await llm_service.get_all_answers_in_one_shot(question_context_pairs)

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"An unexpected error occurred in the RAG pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")