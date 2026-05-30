"""
ProShield Capstone — Configuration Management
Lazy-loaded settings to avoid module-level env var issues
"""
import os
from functools import lru_cache
from pydantic import BaseSettings, Field
from dotenv import load_dotenv; load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: str = Field(default="false", env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="proshield-capstone", env="LANGCHAIN_PROJECT")
    deepeval_api_key: str = Field(default="", env="DEEPEVAL_API_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket: str = Field(default="", env="S3_BUCKET")
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore unknown env vars

@lru_cache()
def get_settings() -> Settings:
    """Lazy-load settings — call this inside functions, never at module level."""
    return Settings()
