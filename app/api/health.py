import logging
import time
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from typing import Dict, Optional

from app.core.config import settings
from app.db.connection import db_engines
from app.langchain.agent import test_azure_openai_connection

logger = logging.getLogger(__name__)

router = APIRouter()

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    version: str
    dependencies: Dict[str, str]
    uptime: float

# Global variable to track API start time
start_time = time.time()

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(response: Response):
    """Health check endpoint."""
    logger.debug("Health check requested")

    # Check database connections
    db_status = "connected" if db_engines else "disconnected"
    if db_engines:
        # Check if any database is connected
        db_status = "disconnected"
        for db_name, engine in db_engines.items():
            try:
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                db_status = "connected"
                break
            except Exception as e:
                logger.warning(f"Database connection check failed: {str(e)}")

    # Check Azure OpenAI connection
    azure_openai_status = "unknown"
    try:
        if await test_azure_openai_connection():
            azure_openai_status = "available"
        else:
            azure_openai_status = "unavailable"
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
        logger.warning(f"Azure OpenAI connection check failed: {str(e)}")
        azure_openai_status = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    # Determine overall status
    if db_status == "connected" and azure_openai_status == "available":
        overall_status = "healthy"
    elif db_status == "disconnected" and azure_openai_status == "unavailable":
        overall_status = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        overall_status = "degraded"
        response.status_code = status.HTTP_200_OK
    
    uptime = time.time() - start_time
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        dependencies={
            "database": db_status,
            "azure_openai": azure_openai_status
        },
        uptime=uptime
    )
