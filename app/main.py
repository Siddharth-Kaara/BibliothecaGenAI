import logging
import logging.config
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles

from app.api import chat, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.connection import create_db_engines, validate_schema_definitions

# --- Setup logging FIRST --- #
setup_logging()
logger = logging.getLogger(__name__) # Get logger after setup
logger.info("Logging configured.")
# ------------------------ #

# --- CORS Configuration ---
# Define the origins allowed to access your API
# For local development, allow the frontend server origin
origins = [
    "http://localhost:8080",  # The default port for `python -m http.server`
    "http://localhost:8001",  # Add other ports if you use them
    "http://localhost:9000",  # Add other ports if you use them
    "http://localhost:3000",  # Common React/Vue development port
    "http://localhost:8000",  # Allow the API origin itself (for Swagger UI etc.)
    "http://127.0.0.1:8080", # Explicit IP addresses can also be needed sometimes
    "http://127.0.0.1:8000",
]
# ------------------------

# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use the configured logger
    logger.info("Lifespan: Starting up Bibliotheca Chatbot API")
    logger.info("Lifespan: Initializing async database connections")
    await create_db_engines()
    
    if settings.VALIDATE_SCHEMA_ON_STARTUP:
        logger.info("Lifespan: Validating database schema definitions")
        await validate_schema_definitions()
    
    logger.info("Lifespan: Application startup tasks complete.")
    yield
    
    logger.info("Lifespan: Shutting down Bibliotheca Chatbot API")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Bibliotheca Chatbot API - A LangChain and Azure OpenAI powered API for library data insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

logger.info("FastAPI app created and configured.")

if __name__ == "__main__":
    # Ensure logging is setup before uvicorn potentially takes over
    logger.info(f"Starting Uvicorn server. Host=0.0.0.0, Port=8000, Reload={settings.DEBUG}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(), # Use level from settings
        # Use default Uvicorn log config unless specified otherwise
        # log_config=None 
    ) 