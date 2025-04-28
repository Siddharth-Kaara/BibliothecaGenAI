import logging
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
# from pydantic import BaseModel # No longer directly needed here

# Import the refactored process_chat_message
from app.langchain.agent import process_chat_message
# Remove old memory imports
# from app.langchain.memory import add_messages_to_memory, get_memory_for_session
# Import schemas directly
from app.schemas.chat import ChatRequest, ChatResponse #, ChatMessage, Error # Error handled within process_chat_message

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Processes a chat message using the LangGraph agent and returns a response."""
    # Create a unique ID for this request to track any leaked tasks
    request_id = str(uuid.uuid4())
    logger.info(f"Starting chat request for org {request.organization_id}, session: {request.session_id}")
    
    # Create a new task group to ensure all tasks are properly awaited
    # This helps prevent "leaking" tasks across requests
    try:
        # Set a timeout for the entire processing to prevent hung requests
        api_response_dict = await asyncio.wait_for(
            process_chat_message(
                organization_id=request.organization_id,
                message=request.message,
                session_id=request.session_id,
                request_id=request_id  # Pass request ID for tracing, not logging
            ),
            timeout=60.0  # 60 second timeout to prevent infinite hang
        )
        
        logger.info(f"Successfully completed chat request")
        
        # Construct the response model
        response = ChatResponse(**api_response_dict)
        return response

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after 60 seconds (ReqID: {request_id})")
        return ChatResponse(
            status="error",
            data=None,
            error={
                "code": "REQUEST_TIMEOUT",
                "message": "The request took too long to process. Please try again with a simpler query.",
                "details": {"request_id": request_id}
            },
            timestamp=datetime.now()
        )
    except Exception as e:
        # Generic error handler for unexpected issues
        logger.error(f"Unexpected error in chat API endpoint: {str(e)} (ReqID: {request_id})", exc_info=True)
        
        return ChatResponse(
            status="error",
            data=None,
            error={
                "code": "API_ENDPOINT_ERROR",
                "message": "An unexpected error occurred handling your request.",
                "details": {"error": str(e), "request_id": request_id}
            },
            timestamp=datetime.now()
        )
