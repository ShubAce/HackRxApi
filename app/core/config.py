import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    pydantic-settings automatically finds and loads the .env file.
    """
    # Security
    API_BEARER_TOKEN: str

    # Google AI
    GOOGLE_API_KEY: str

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "hackrx-rag-index"
    
    # Model Configuration
    EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"
    GENERATIVE_MODEL_NAME: str = "gemini-1.5-flash-latest"
    EMBEDDING_DIMENSION: int = 768

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra='ignore'
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Returns the settings instance, cached for performance.
    """
    return Settings()
