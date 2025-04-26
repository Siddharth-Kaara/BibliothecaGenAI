import os
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API settings
    PROJECT_NAME: str = "Bibliotheca Chatbot API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    POSTGRES_SERVERS: List[Dict[str, Any]] = []  # Will be populated from DATABASE_URLS
    DATABASE_URLS: str = ""  # Format: "db1=postgresql://user:pass@host:port/db1,db2=postgresql://user:pass@host:port/db2"
    VALIDATE_SCHEMA_ON_STARTUP: bool = True  # Whether to validate schema definitions against actual DB
    
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2023-07-01-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    
    # LangChain settings
    LLM_MODEL_NAME: str = "gpt-4o"
    VERBOSE_LLM: bool = False
    
    # Security
    SECRET_KEY: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Chart generation
    CHART_DIR: str = "static/charts"
    CHART_URL_BASE: str = "/static/charts"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = 'ignore'
    
    @field_validator("CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, list):
            return v
        raise ValueError(v)
    
    @field_validator("DATABASE_URLS")
    def parse_database_urls(cls, v: str) -> str:
        if not v:
            return v
        
        # Parse the database URLs and populate POSTGRES_SERVERS
        db_configs = []
        for db_config in v.split(","):
            if "=" not in db_config:
                continue
                
            db_name, db_url = db_config.split("=", 1)
            db_configs.append({
                "name": db_name.strip(),
                "url": db_url.strip()
            })
        
        # Instead of trying to access settings, return both values
        return {"url_string": v, "servers": db_configs}

# Create global settings object
settings = Settings()

# Process the database URLs result
if settings.DATABASE_URLS and isinstance(settings.DATABASE_URLS, dict):
    settings.POSTGRES_SERVERS = settings.DATABASE_URLS.get("servers", [])
    settings.DATABASE_URLS = settings.DATABASE_URLS.get("url_string", "")

# Ensure chart directory exists
os.makedirs(settings.CHART_DIR, exist_ok=True)
