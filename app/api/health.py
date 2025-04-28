import logging
import time
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from typing import Dict, Optional
from sqlalchemy import text

from app.core.config import settings
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

    # Check database connections using async connections
    db_status = "disconnected"
    
    # Import async db functionality
    from app.db.connection import get_async_db_connection, async_db_engines
    
    if async_db_engines:
        for db_name in async_db_engines:
            try:
                async with get_async_db_connection(db_name) as conn:
                    result = await conn.execute(text("SELECT 1"))

                    row = result.fetchone()
                    if row and row[0] == 1:
                        db_status = "connected"
                        logger.debug(f"Async database connection check successful for {db_name}")
                        break
                    else:
                        logger.warning(f"Database connection check failed for {db_name}: Unexpected result {row}")
            except Exception as e:
                logger.warning(f"Async database connection check failed for {db_name}: {str(e)}")
    else:
        logger.warning("No async database engines configured")

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
