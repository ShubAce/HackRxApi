import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # Security
    API_BEARER_TOKEN: str

    # Google AI
    GOOGLE_API_KEY: str

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "hackrx"
    
    # Model Configuration
    EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"
    GENERATIVE_MODEL_NAME: str = "gemini-1.5-flash" # Optimized for speed and cost
    EMBEDDING_DIMENSION: int = 768 # Dimension for text-embedding-004

    # The change is here:
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra='ignore'  # This tells Pydantic to ignore extra fields
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Returns the settings instance, cached for performance.
    """
    return Settings()