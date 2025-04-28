import os
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, PostgresDsn, field_validator, model_validator
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
    DB_POOL_SIZE: int = 10 # Default pool size for database connections
    DB_MAX_OVERFLOW: int = 20 # Default max overflow connections (Added sensible default)
    DB_POOL_TIMEOUT: int = 30 # Default seconds to wait for connection (Kept original implicit default)
    DB_POOL_RECYCLE: int = 1800 # Default seconds after which connections are recycled (e.g., 30 mins)
    
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2023-07-01-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    
    # LangChain settings
    LLM_MODEL_NAME: str = "gpt-4o"
    VERBOSE_LLM: bool = False
    
    # Agent & Graph settings
    MAX_CONCURRENT_TOOLS: int = 10 # Max concurrent tool executions (Semaphore limit)
    MAX_STATE_MESSAGES: int = 50 # Max messages to keep in AgentState history
    MAX_GRAPH_ITERATIONS: int = 10 # Max recursion depth for the LangGraph agent
    
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
    
    @model_validator(mode='after')
    def check_required_settings(cls, values):
        # Check critical settings like API keys after loading
        api_key = values.AZURE_OPENAI_API_KEY
        endpoint = values.AZURE_OPENAI_ENDPOINT
        deployment = values.AZURE_OPENAI_DEPLOYMENT_NAME
        
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is missing or empty.")
        if not endpoint:
             raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is missing or empty.")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is missing or empty.")
            
        # Re-process DATABASE_URLS to populate POSTGRES_SERVERS
        # This needs to happen here because the field validator runs before this model validator
        db_urls = values.DATABASE_URLS
        if db_urls:
            db_configs = []
            for db_config in db_urls.split(","):
                if "=" not in db_config:
                    continue
                db_name, db_url = db_config.split("=", 1)
                db_configs.append({
                    "name": db_name.strip(),
                    "url": db_url.strip()
                })
            # Assign directly to the attribute on the instance
            values.POSTGRES_SERVERS = db_configs
            
        return values

# Create global settings object
settings = Settings()

# Ensure chart directory exists
os.makedirs(settings.CHART_DIR, exist_ok=True)
