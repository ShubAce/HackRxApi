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
    summary="Process a document from a URL and answer questions"
    # dependencies=[Depends(verify_token)] # <-- Commented out for testing
)
async def run_submission(request: QueryRequest):
    """
    This endpoint implements the complete RAG (Retrieval-Augmented Generation) pipeline
    for a single document provided via URL:
    1.  **Ingestion**: Downloads, parses, chunks, and embeds the document from the URL.
    2.  **Storage**: Upserts the document chunks into a unique, isolated namespace in Pinecone.
    3.  **Retrieval**: For each question, retrieves the most relevant chunks from that document.
    4.  **Generation**: Generates an answer using an LLM with the retrieved context.
    """
    document_url = str(request.documents)
    logger.info(f"Processing request for document: {document_url}")

    try:
        # --- 1. & 2. Ingestion and Storage ---

        # Generate a unique namespace from the URL hash. This is crucial.
        # It isolates the data for this specific document and request.
        namespace = document_service.generate_document_namespace(document_url)

        # Download and parse the document content
        content, file_type = await document_service.get_document_from_url(document_url)
        full_text = document_service.parse_document(content, file_type)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract any text from the document.")

        # Chunk the text into manageable pieces
        chunks = document_service.chunk_text(full_text)
        chunk_texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings for all text chunks
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = llm_service.get_embeddings(chunk_texts)

        # Prepare vectors for Pinecone with unique IDs and metadata
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vectors_to_upsert.append({
                'id': f'chunk_{i}',
                'values': embeddings[i],
                'metadata': {'text': chunk['text']}
            })

        # Upsert the vectors into their dedicated namespace
        vector_db_service.upsert(vectors=vectors_to_upsert, namespace=namespace)
        logger.info(f"Successfully upserted document into namespace: {namespace}")


        # --- 3. & 4. Retrieval and Generation ---

        async def process_question(question: str):
            logger.info(f"Processing question: '{question}'")
            
            # Embed the user's question
            query_embedding = llm_service.get_query_embedding(question)
            
            # Retrieve relevant chunks from the document's specific namespace
            retrieved_matches = vector_db_service.query(
                query_embedding=query_embedding,
                top_k=5,  # Retrieve the top 5 most relevant chunks
                namespace=namespace
            )
            
            # Combine the text from the retrieved chunks to form the context
            context = "\n---\n".join([match['metadata']['text'] for match in retrieved_matches])
            
            if not context:
                return "The provided document does not contain information on this topic."

            # Generate the final answer using the LLM based on the context
            return await llm_service.get_answer_from_context(question, context)

        # Process all questions concurrently for better performance
        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)

        return QueryResponse(answers=answers)

    except ValueError as e:
        logger.error(f"Validation error processing document: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")