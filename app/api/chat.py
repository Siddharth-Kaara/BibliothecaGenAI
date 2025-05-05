import logging
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Sequence

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Body, Path
# from pydantic import BaseModel # No longer directly needed here

# Import the refactored process_chat_message
from app.langchain.agent import process_chat_message
# Remove old memory imports
# from app.langchain.memory import add_messages_to_memory, get_memory_for_session
# Import schemas directly
from app.schemas.chat import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, HistoryResponse, ChatMessageForHistory # Ensure ChatMessageForHistory is imported if used in responses

# --- Add DB Imports ---
from sqlalchemy.orm import Session
# Import AsyncSession and async_sessionmaker from sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker 
# Import our specific engine getter and dependency function from our connection module
from app.db.connection import get_chat_db_session, get_async_db_engine, chat_management_session_factory
# Import ORM Models needed for type hints
from app.models.chat import ChatMessage, ChatSession # <-- ADDED IMPORTS
# Import repository functions AND custom exceptions
from app.repositories.chat_repo import ( 
    # REMOVED: create_chat_message_request, 
    # REMOVED: update_chat_message_response,
    log_complete_chat_message, 
    add_feedback,
    get_session_history,
    get_or_create_session, # <-- ADDED IMPORT
    # get_recent_messages_for_agent, # <-- REMOVED
    # Import custom exceptions
    RepositoryError, 
    RecordNotFound, 
    SessionMismatchError, 
    FeedbackIntegrityError,
    create_chat_session, # Keep
    log_initial_chat_message, # Keep
)
# --- End DB Imports ---

# --- Langchain imports ---
from langchain_core.messages import HumanMessage, AIMessage # Keep for type checking if needed elsewhere

# Import auth dependency
from app.security import get_current_user, CurrentUser
# Import settings for config values
from app.core.config import settings # Added import

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Helper to Format History for Agent --- #
def _format_db_history_for_agent(history_records: Sequence[ChatMessage]) -> List[Dict[str, str]]:
    """Formats ChatMessage ORM objects into the list of dicts agent expects."""
    agent_history = []
    for record in history_records:
        # Add user message first
        if record.userContent:
             agent_history.append({"role": "user", "content": record.userContent})
        # Then add assistant message if it exists
        if record.assistantContent:
            agent_history.append({"role": "assistant", "content": record.assistantContent})
    return agent_history
# --- End Helper --- #

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    background_tasks: BackgroundTasks,
    chat_request: ChatRequest,
    session_id: Optional[str] = Query(None, description="Optional session ID for continuing a conversation"),
    db: AsyncSession = Depends(get_chat_db_session),
    current_user: CurrentUser = Depends(get_current_user)
):
    request_id = uuid.uuid4()
    # Get user_id and org_id reliably from the validated CurrentUser object
    user_id = current_user.user_id
    organization_id = current_user.org_id # Access directly

    # The check below is technically redundant if get_current_user enforces non-None,
    # but kept for extra safety / explicitness.
    if not organization_id:
        # This case should ideally not be hit if token validation is correct
        logger.error(f"[ReqID: {request_id}] Organization ID missing from validated user object for user {user_id}. Token/Security config issue?")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal configuration error: Organization ID not found.")

    logger.info(f"Starting chat request {request_id} for org {organization_id}, user {user_id}, session: {session_id}")

    # --- Session Handling & History Retrieval --- #
    session_uuid: Optional[uuid.UUID] = None
    db_history_records: Sequence[ChatMessage] = []
    agent_chat_history: List[Dict[str, str]] = []
    chat_session: Optional[ChatSession] = None # Store the session object

    try:
        # Use get_or_create_session from repository
        chat_session, created = await get_or_create_session(
            db=db,
            session_id=session_id,
            user_id=user_id,
            organization_id=organization_id
        )
        session_uuid = chat_session.sessionId
        session_id = str(session_uuid) # Update session_id string in case it was created

        if not created and session_uuid: # Only fetch history if it was an existing session
             # Retrieve history using the authorized function
             db_history_records = await get_session_history(
                 db=db,
                 session_id=session_id,
                 user_id=user_id,
                 organization_id=organization_id,
                 limit=settings.MAX_STATE_MESSAGES, # Use config for history limit
                 offset=0
             )
             logger.debug(f"Retrieved {len(db_history_records)} message objects from DB for history context, request {request_id}")
             # Format the retrieved history for the agent
             agent_chat_history = _format_db_history_for_agent(db_history_records)

    except RecordNotFound as e:
         logger.error(f"[ReqID: {request_id}] Session handling error: {e}. Could not find or create session.")
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat session error: {e}")
    except RepositoryError as e:
        logger.error(f"[ReqID: {request_id}] Repository error during session handling: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initialize chat session due to a database error.")
    except Exception as e: # Catch unexpected errors during session/history phase
        logger.error(f"[ReqID: {request_id}] Unexpected error retrieving/creating session {session_id} or history for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred handling the chat session.")
    # --- End Session Handling & History Retrieval --- #

    # --- Schedule Initial Log --- #
    if session_uuid: # Ensure we have a valid session ID before logging
        # Use add_task directly with the repository function and the current db session
        background_tasks.add_task(
            log_initial_chat_message,
            db=db,
            request_id=request_id,
            session_id=str(session_uuid),
            user_id=user_id,
            organization_id=organization_id,
            user_content=chat_request.message
        )
    else:
         # This state should ideally not be reachable if session handling is correct
         logger.error(f"[ReqID: {request_id}] session_uuid is None after session handling. Cannot log initial message.")
         # Consider raising an error here as it indicates a logic flaw
         raise HTTPException(status_code=500, detail="Internal error: Failed to establish session ID for logging.")
    # --- End Schedule Initial Log --- #

    # --- Call Langchain Agent --- #
    agent_response_data = {} # Initialize
    agent_status = "error"
    prompt_tokens = 0
    completion_tokens = 0

    try:
        # Pass the formatted history to the agent
        # Use timeout from settings
        result = await asyncio.wait_for(
            process_chat_message(
                organization_id=organization_id,
                message=chat_request.message,
                session_id=str(session_uuid), # Always pass session_id now
                chat_history=agent_chat_history, # Pass formatted history
                request_id=str(request_id),
            ),
            timeout=settings.AGENT_TIMEOUT_SECONDS # Use config for timeout
        )

        # Validate result structure (basic check)
        if isinstance(result, dict) and "status" in result and "data" in result:
             agent_status = result.get("status", "error")
             agent_response_data = result.get("data", {}) if agent_status == "success" else {}
             prompt_tokens = result.get("prompt_tokens", 0)
             completion_tokens = result.get("completion_tokens", 0)
             if agent_status != "success":
                 logger.warning(f"Agent processing returned status '{agent_status}' for request {request_id}. Result: {result.get('error', 'No error details')}")
                 # Raise an exception to be caught below, providing agent error details
                 raise Exception(f"Agent processing failed: {result.get('error', {}).get('message', 'Unknown agent error')}")
        else:
             logger.error(f"Invalid response structure from process_chat_message for request {request_id}. Result: {result}")
             raise Exception("Invalid response structure received from agent.")

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {settings.AGENT_TIMEOUT_SECONDS} seconds (ReqID: {request_id})")
        # Log timeout to DB if desired (might need a separate status or error field in ChatMessage)
        raise HTTPException(
             status_code=status.HTTP_408_REQUEST_TIMEOUT,
             detail={
                 "code": "REQUEST_TIMEOUT",
                 "message": f"The request took too long to process (>{settings.AGENT_TIMEOUT_SECONDS}s). Please try again.",
                 "details": {"request_id": str(request_id)}
             }
        )
    except Exception as e:
        # Handles both explicit raises from agent failure and unexpected errors
        logger.error(f"Error during agent processing: {str(e)} (ReqID: {request_id})", exc_info=True)
        # Log error to DB if desired
        # Use 500 for internal errors, potentially others based on agent error type if available
        raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail={
                 "code": "AGENT_PROCESSING_ERROR",
                 "message": "An error occurred while processing your request with the agent.",
                 "details": {"error": str(e), "request_id": str(request_id)}
             }
         )
    # --- End Call Langchain Agent --- #


    # --- Schedule Completion Log --- #
    # Log completion regardless of agent success/failure status, but ensure we have essential data
    if session_uuid:
        background_tasks.add_task(
            log_complete_chat_message,
            db=db,
            request_id=request_id,
            session_id=str(session_uuid),
            user_id=user_id,
            organization_id=organization_id,
            user_content=chat_request.message, # Log original user message
            assistant_content=agent_response_data.get("text") if agent_status == "success" else None, # Log agent text only on success
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        logger.debug(f"Scheduled background task to log complete message for request {request_id}.")
    else:
        # Should not be reachable if initial log succeeded
         logger.error(f"[ReqID: {request_id}] session_uuid is None before logging completion. Logic error suspected.")
         # Don't prevent response return, but log the error


    # --- Construct Final Response --- #
    # We already raised exceptions for errors, so if we reach here, agent call was successful
    final_response = ChatResponse(
        status=agent_status, # Should be "success"
        data=agent_response_data, # The data from the agent
        error=None, # No error in the successful case
        timestamp=datetime.now()
    )
    return final_response


# --- Add Feedback Endpoint ---
@router.post("/feedback", status_code=status.HTTP_201_CREATED, tags=["feedback"], response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_chat_db_session),
    current_user: CurrentUser = Depends(get_current_user)
) -> FeedbackResponse:
    """Submits feedback for a specific chat message."""
    # Get user_id and org_id reliably from the validated CurrentUser object
    user_id = current_user.user_id
    organization_id = current_user.org_id # Access directly

    # Redundant check, but safe
    if not organization_id:
        logger.error(f"Missing organization_id in token for user {user_id} during feedback submission.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Organization information missing or invalid token.")

    logger.info(f"Received feedback for request_id: {request.request_id}, rating: {request.rating} by user {user_id} for org {organization_id}")
    try:
        # Pass validated user_id and org_id from token
        feedback_record = await add_feedback(
            db=db,
            request_id=request.request_id,
            rating=request.rating,
            session_id=request.session_id,
            user_id=user_id,
            organization_id=organization_id
        )
        feedback_id_value = str(feedback_record.feedbackId) # Ensure it's string
        logger.info(f"Successfully saved feedback with ID: {feedback_id_value}")
        return FeedbackResponse(status="success", feedback_id=feedback_id_value)

    # --- Catch Custom Repository Exceptions ---
    except RecordNotFound as e:
        logger.warning(f"RecordNotFound saving feedback for {request.request_id} by user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    except SessionMismatchError as e: # This might be less relevant now with RecordNotFound check
        logger.warning(f"SessionMismatchError saving feedback for {request.request_id} by user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FeedbackIntegrityError as e:
        logger.error(f"FeedbackIntegrityError saving feedback for {request.request_id} by user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) # 409 Conflict is suitable

    except RepositoryError as e: # Catch generic repo errors
        logger.error(f"RepositoryError saving feedback for {request.request_id} by user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="A database error occurred while saving feedback.")

    except Exception as e:
        # Catch any other unexpected errors during API processing
        logger.error(f"Unexpected API error saving feedback for request_id {request.request_id}, user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred."
        )
    # --- End Exception Handling ---

# --- History Endpoint (Async) ---
@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["history"])
async def get_history(
    session_id: str = Path(..., description="The ID of the chat session to retrieve history for."),
    limit: int = Query(settings.DEFAULT_HISTORY_LIMIT, ge=1, le=settings.MAX_HISTORY_LIMIT, description="Maximum number of messages to return."),
    offset: int = Query(0, ge=0, description="Number of messages to skip."),
    db: AsyncSession = Depends(get_chat_db_session),
    current_user: CurrentUser = Depends(get_current_user)
) -> HistoryResponse:
    """Retrieves the chat message history for a given session ID with pagination."""
    # Get user_id and org_id reliably from the validated CurrentUser object
    user_id = current_user.user_id
    organization_id = current_user.org_id # Access directly

    # Redundant check, but safe
    if not organization_id:
        logger.error(f"Missing organization_id in token for user {user_id} during history retrieval.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Organization information missing or invalid token.")

    logger.info(f"Retrieving chat history for session_id: {session_id}, user {user_id}, org {organization_id}, limit: {limit}, offset: {offset}")
    try:
        # Validate UUID format before hitting DB
        try:
             session_uuid = uuid.UUID(session_id)
        except ValueError:
             logger.warning(f"Invalid session_id format '{session_id}' provided by user {user_id}")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session ID format.")

        history_orm = await get_session_history(
            db=db,
            session_id=str(session_uuid),
            user_id=user_id, # Use user_id from token
            limit=limit,
            offset=offset,
            organization_id=organization_id # Use org_id from token
        )

        # get_session_history now raises RecordNotFound if session doesn't exist for user/org
        # No need for the explicit check here anymore

        # Convert ORM objects to Pydantic models expected by HistoryResponse
        history_messages = [ChatMessageForHistory.model_validate(msg) for msg in history_orm]


        response = HistoryResponse(session_id=session_id, messages=history_messages) # Use validated messages
        logger.info(f"Successfully retrieved {len(history_messages)} messages for session_id: {session_id}, user {user_id}")
        return response

    except RecordNotFound as e:
         logger.warning(f"History not found for session {session_id}, user {user_id}, org {organization_id}: {e}")
         # Return 404 if the session itself is not found or not authorized
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RepositoryError as e:
         logger.error(f"Database error retrieving history for session {session_id}, user {user_id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to retrieve chat history due to a database error.")
    except HTTPException as http_err: # Re-raise specific HTTP exceptions
        raise http_err
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error retrieving history for session_id {session_id}, user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred while retrieving chat history."
        )
# --- End History Endpoint ---
