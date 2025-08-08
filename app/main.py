from fastapi import FastAPI
from app.api.v1 import routes as v1_routes
from app.core.config import get_settings
from app.utils.logger import get_logger

# Initialize settings and logger at the start.
# If there's an error here (e.g., missing env var), the app will fail fast.
settings = get_settings()
logger = get_logger(__name__)

# Create a simple in-memory cache at the global scope.
# This will persist for the lifetime of the server process on Render.
PROCESSED_DOCUMENT_CACHE = set()

# Create the main FastAPI app instance
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="An LLM-Powered system to process documents and answer contextual questions.",
    version="3.0.0-Optimized"
)

# Use the app's state to make the cache accessible to your API routes.
# This is a clean way to share state without using global variables directly in endpoints.
app.state.PROCESSED_DOCUMENT_CACHE = PROCESSED_DOCUMENT_CACHE

@app.on_event("startup")
async def startup_event():
    """
    This function runs when the application starts. It's a good place
    to log startup information.
    """
    logger.info("Application startup...")
    logger.info(f"Loaded config: INDEX_NAME='{settings.PINECONE_INDEX_NAME}', GENERATIVE_MODEL='{settings.GENERATIVE_MODEL_NAME}'")
    logger.info("In-memory document cache initialized successfully.")

# Include your API router. This is what makes your endpoints accessible.
# The prefix ensures all routes in v1_routes start with /api/v1.
app.include_router(v1_routes.router, prefix="/api/v1", tags=["Query System"])

@app.get("/", tags=["Health Check"])
async def root():
    """
    A simple health check endpoint for the root URL.
    If you can access this endpoint, your application is running correctly.
    """
    return {"status": "ok", "message": "Welcome to the Intelligent Query-Retrieval System API!"}
