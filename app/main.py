# In app/main.py
from fastapi import FastAPI
from app.api.v1 import routes as v1_routes
from app.core.config import get_settings
from app.utils.logger import get_logger

# ... (imports) ...

# Create a simple in-memory cache at the global scope
# This will track the namespaces of documents we've already processed.
PROCESSED_DOCUMENT_CACHE = set()

# ... (your FastAPI app initialization) ...

app = FastAPI(
    title="Intelligent Query–Retrieval System",
    # ...
)

# Pass the cache to the router using state
app.state.PROCESSED_DOCUMENT_CACHE = PROCESSED_DOCUMENT_CACHE

# ... (the rest of your main.py file, including the router) ...