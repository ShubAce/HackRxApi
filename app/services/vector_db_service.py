import pinecone
from app.core.config import get_settings
from app.utils.logger import get_logger
from typing import List, Dict, Any
from app.core.config import get_settings
logger = get_logger(__name__)
settings = get_settings()

class VectorDBService:
    def __init__(self):
        try:
            self.pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self.dimension = settings.EMBEDDING_DIMENSION
            self._create_index_if_not_exists()
            logger.info("Pinecone service initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def _create_index_if_not_exists(self):
        """Creates the Pinecone index if it doesn't already exist."""
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine', # Cosine similarity is great for semantic search
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logger.info(f"Index '{self.index_name}' created successfully.")
        else:
            logger.info(f"Pinecone index '{self.index_name}' already exists.")
        
        self.index = self.pc.Index(self.index_name)

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str):
        """
        Upserts vectors into the Pinecone index within a specific namespace.
        A namespace is used to isolate documents.
        """
        if not vectors:
            logger.warning("Upsert called with no vectors.")
            return
        
        logger.info(f"Upserting {len(vectors)} vectors to namespace '{namespace}'...")
        # Upsert in batches for efficiency
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
        logger.info("Upsert completed.")

    def query(self, query_embedding: List[float], top_k: int, namespace: str) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index for the most similar vectors.
        """
        logger.info(f"Querying index in namespace '{namespace}' with top_k={top_k}.")
        results = self.index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.get('matches', [])

# Singleton instance
vector_db_service = VectorDBService()