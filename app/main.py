from fastapi import FastAPI
from app.api.v1 import routes as v1_routes
from app.core.config import get_settings
from app.utils.logger import get_logger

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="An LLM-Powered system to process documents from a URL and answer contextual questions.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    logger.info(f"Loaded config: INDEX_NAME='{settings.PINECONE_INDEX_NAME}', GENERATIVE_MODEL='{settings.GENERATIVE_MODEL_NAME}'")

# Include API routers
app.include_router(v1_routes.router, prefix="/api/v1", tags=["Query System"])

@app.get("/", tags=["Health Check"])
async def root():
    """
    A simple health check endpoint.
    """
    return {"status": "ok", "message": "Welcome to the Intelligent Query-Retrieval System API!"} 