import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
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
async def chat(request: ChatRequest):
    """Processes a chat message using the LangGraph agent and returns a response."""
    try:
        # logger.info(f"Received chat request from user {request.user_id} with session_id: {request.session_id}")
        logger.info(f"Received chat request for org {request.organization_id} with session_id: {request.session_id}")

        # The LangGraph agent now handles memory via session_id internally.
        # We pass the necessary info directly to process_chat_message.
        # No need to manually load/add history here.

        # Process the message using the refactored function
        api_response_dict = await process_chat_message(
            # user_id=request.user_id, # Removed
            organization_id=request.organization_id,
            message=request.message,
            session_id=request.session_id, # Pass session_id directly
            # chat_history is no longer passed here
        )

        # process_chat_message now returns the complete dictionary needed for ChatResponse
        # We just need to construct the Pydantic model from it.
        return ChatResponse(**api_response_dict)

    except Exception as e:
        # Generic error handler for unexpected issues in this layer
        logger.error(f"Unexpected error in chat API endpoint: {str(e)}", exc_info=True)
        # Use a standard error structure, consistent with how process_chat_message might return errors
        error_response = ChatResponse(
             status="error",
             data=None,
             error={
                 "code": "API_ENDPOINT_ERROR",
                 "message": "An unexpected error occurred handling your request.",
                 "details": {"error": str(e)}
             },
             timestamp=datetime.now()
        )
        # Return it directly, maybe with a 500 status code, although response_model handles structure
        # Consider raising HTTPException for clearer status codes if needed, but
        # returning the ChatResponse structure ensures consistency.
        # For now, just return the structured error within a 200 OK,
        # as process_chat_message also returns errors this way.
        return error_response
        # Alternative: Raise HTTPException for non-200 status
        # raise HTTPException(
        #     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        #     detail=error_response.dict() # Send the structured error as detail
        # )
