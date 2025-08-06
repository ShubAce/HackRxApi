import os
from app.services.document_service import document_service
from app.services.llm_service import llm_service
from app.services.vector_db_service import vector_db_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

async def run_ingestion():
    """
    Processes all PDF files in the 'data' directory and upserts them into Pinecone.
    This function is designed to be run once at application startup.
    """
    data_dir = "app\data"
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory '{data_dir}' not found. Skipping ingestion.")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{data_dir}'. Skipping ingestion.")
        return
        
    # Simple check to prevent re-indexing if the index already has data.
    # For a production system, a more robust check (e.g., using a database flag) is recommended.
    index_stats = vector_db_service.index.describe_index_stats()
    if index_stats.total_vector_count > 0:
        logger.info("Pinecone index already contains vectors. Skipping ingestion process.")
        logger.info("To re-index, delete the existing Pinecone index and restart the server.")
        return

    logger.info(f"Starting ingestion for {len(pdf_files)} PDF files...")

    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        logger.info(f"Processing file: {filename}")
        
        try:
            with open(filepath, "rb") as f:
                content = f.read()

            full_text = document_service.parse_document(content, "pdf")
            if not full_text.strip():
                logger.warning(f"No text extracted from {filename}. Skipping.")
                continue

            chunks = document_service.chunk_text(full_text)
            
            # Add source document metadata to each chunk
            for chunk in chunks:
                chunk['metadata']['source_document'] = filename
            
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks from {filename}...")
            embeddings = llm_service.get_embeddings(chunk_texts)

            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                vectors_to_upsert.append({
                    'id': f'{filename}_chunk_{i}',
                    'values': embeddings[i],
                    # Store the text and the source document name in the metadata
                    'metadata': {'text': chunk['text'], 'source': filename}
                })
            
            # For this multi-file setup, we will upsert into a single, shared namespace.
            # The 'source' in metadata will tell us which document a chunk came from.
            # We can use a default project-wide namespace, or none at all if using a serverless index.
            vector_db_service.upsert(vectors=vectors_to_upsert, namespace="main-knowledge-base")

        except Exception as e:
            logger.error(f"Failed to process and ingest {filename}: {e}", exc_info=True)

    logger.info("Ingestion process completed for all files.")