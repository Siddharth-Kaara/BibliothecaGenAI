import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Sequence

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Body, Path
# from pydantic import BaseModel # No longer directly needed here

# Import the refactored process_chat_message
from app.langchain.agent import process_chat_message
# Remove old memory imports
# from app.langchain.memory import add_messages_to_memory, get_memory_for_session
# Import schemas directly
from app.schemas.chat import ChatRequest, ChatResponse, ChatData, FeedbackRequest, FeedbackResponse, HistoryResponse, ChatMessageForHistory # Added ChatData

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
        # Add user message first using correct snake_case attribute
        if record.user_message_content:
             agent_history.append({"role": "user", "content": record.user_message_content})
        # Then add assistant message if it exists using correct snake_case attribute
        if record.assistant_message_content:
            agent_history.append({"role": "assistant", "content": record.assistant_message_content})
    return agent_history
# --- End Helper --- #

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    background_tasks: BackgroundTasks,
    chat_request: ChatRequest, # Contains message, session_id (optional)
    # session_id: Optional[str] = Query(None, ...), # REMOVED Query Parameter
    db: AsyncSession = Depends(get_chat_db_session),
    current_user: CurrentUser = Depends(get_current_user)
):
    request_id = uuid.uuid4()
    # Get user_id and org_id reliably from the validated CurrentUser object
    user_id = current_user.user_id
    organization_id = current_user.org_id

    # Ensure org_id is present (already validated by get_current_user)
    if not organization_id:
        logger.error(f"[ReqID: {request_id}] Organization ID missing from validated user object for user {user_id}. Token/Security config issue?")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal configuration error: Organization ID not found.")

    # --- Agent Call Variables (used in both branches) --- #
    agent_response_data = {} # Initialize
    agent_status = "error"
    prompt_tokens = 0
    completion_tokens = 0

    # --- Unified Stateful Logic (Handles New and Existing Sessions) --- #
    request_session_id_str = chat_request.session_id # Get session_id string from the request body (can be None)
    session_uuid_obj: Optional[uuid.UUID] = None
    chat_session: Optional[ChatSession] = None
    created_session: bool = False
    db_history_records: Sequence[ChatMessage] = []
    agent_chat_history: List[Dict[str, str]] = []

    # --- Session Handling, Validation & History Retrieval --- #
    try:
        # 1. Validate format if ID was provided
        if request_session_id_str:
            try:
                session_uuid_obj = uuid.UUID(request_session_id_str)
            except ValueError:
                logger.warning(f"[ReqID: {request_id}] Invalid session_id format '{request_session_id_str}' provided by user {user_id}.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session ID format.")
        
        # 2. Get or Create Session (session_uuid_obj will be None if starting new)
        chat_session, created_session = await get_or_create_session(
            db=db,
            session_id=session_uuid_obj, # Pass None or the validated UUID
            user_id=user_id,             
            organization_id=organization_id 
        )
        
        # 3. Handle Cases based on whether ID was provided and if session was found/created
        if request_session_id_str and chat_session is None:
            # ID was provided, but session not found by repo function
            logger.warning(f"[ReqID: {request_id}] Session ID '{request_session_id_str}' provided but not found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
            
        if chat_session is None: 
            # This *shouldn't* happen if session_id was None (repo should create or raise error) 
            # or if session_id was provided (caught above). Safety check.
            logger.error(f"[ReqID: {request_id}] chat_session is unexpectedly None after get_or_create_session call.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to establish chat session.")

        # 4. Authorization Check (only needed if session wasn't just created)
        if not created_session:
            if chat_session.user_id != user_id or chat_session.organization_id != organization_id:
                 logger.error(f"[ReqID: {request_id}] AuthZ Error: Session {chat_session.session_id} user/org mismatch. Req: {user_id}/{organization_id}. Sess: {chat_session.user_id}/{chat_session.organization_id}.")
                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.") # User doesn't need to know it exists but isn't theirs

        # 5. If we get here, chat_session is valid (either found+authorized or newly created)
        session_id_str_for_logging_and_agent = str(chat_session.session_id)
        logger.info(f"Processing STATEFUL chat request {request_id} using session {session_id_str_for_logging_and_agent} (Created: {created_session})")

        # 6. Retrieve history ONLY if the session wasn't newly created
        if not created_session:
            db_history_records = await get_session_history(
                db=db,
                session_id=session_id_str_for_logging_and_agent,
                user_id=user_id,
                organization_id=organization_id,
                limit=settings.MAX_STATE_MESSAGES,
                offset=0
            )
            logger.debug(f"Retrieved {len(db_history_records)} message objects from DB for history context, request {request_id}")
            agent_chat_history = _format_db_history_for_agent(db_history_records)

    except HTTPException as http_exc: # Re-raise specific HTTP exceptions (400, 404)
         raise http_exc
    except RepositoryError as e: # Catch errors from repo (e.g., DB error during creation)
        logger.error(f"[ReqID: {request_id}] Repository error during session handling: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to handle chat session due to a database error.")
    except Exception as e:
        logger.error(f"[ReqID: {request_id}] Unexpected error handling session or history for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred handling the chat session.")
    # --- End Session Handling --- #

    # --- Background Logging & Agent Call (Stateful - chat_session is guaranteed valid) --- #
    
    # Define a variable to hold assistant content for logging, default to None
    assistant_content_for_log: Optional[str] = None

    # Schedule Initial Log
    background_tasks.add_task(
        log_initial_chat_message,
        db=db,
        request_id=request_id,
        session_id=session_id_str_for_logging_and_agent, # Use the confirmed session ID
        user_id=user_id,
        organization_id=organization_id,
        user_content=chat_request.message
    )
    
    # Call Langchain Agent (Stateful)
    try:
        result = await asyncio.wait_for(
            process_chat_message(
                organization_id=organization_id,
                message=chat_request.message,
                session_id=session_id_str_for_logging_and_agent, # Use the confirmed session ID
                chat_history=agent_chat_history,
                request_id=str(request_id),
            ),
            timeout=settings.AGENT_TIMEOUT_SECONDS
        )

        # Validate and process agent result
        if isinstance(result, dict) and "status" in result and "data" in result:
            agent_status = result.get("status", "error")
            agent_response_data = result.get("data", {}) if agent_status == "success" else {}
            prompt_tokens = result.get("prompt_tokens", 0)
            completion_tokens = result.get("completion_tokens", 0)
            if agent_status != "success":
                error_detail = result.get('error', {}).get('message', 'Unknown agent error')
                logger.warning(f"Agent processing returned status '{agent_status}' for STATEFUL request {request_id}. Error: {error_detail}")
                raise Exception(f"Agent processing failed: {error_detail}")
        else:
            logger.error(f"Invalid response structure from process_chat_message for STATEFUL request {request_id}. Result: {result}")
            raise Exception("Invalid response structure received from agent.")

    except asyncio.TimeoutError:
        logger.warning(f"Agent processing timed out for STATEFUL request {request_id} after {settings.AGENT_TIMEOUT_SECONDS} seconds.")
        # SET assistant_content_for_log for timeout case
        assistant_content_for_log = "[Agent Timeout]"
        # DO NOT schedule log_complete_chat_message here
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Chat processing timed out.")

    except Exception as e:
        logger.error(f"Error during agent processing for STATEFUL request {request_id}: {e}", exc_info=True)
        # SET assistant_content_for_log for error case
        assistant_content_for_log = f"[Agent Error]: {str(e)[:500]}" # Log truncated error
        # DO NOT schedule log_complete_chat_message here
        # Determine appropriate status code based on error type if possible
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = "An error occurred during chat processing."
        raise HTTPException(status_code=status_code, detail=detail)
    # --- End Call Langchain Agent (Stateful) --- #

    # --- Construct Final Response (Stateful) --- #
    if agent_status == "success":
        # Construct response data
        response_data = ChatData(**agent_response_data)
        # Set assistant_content_for_log for success case
        assistant_content_for_log = response_data.text if response_data else None
        response = ChatResponse(
            status="success",
            request_id=request_id,
            data=response_data,
            session_id=chat_session.session_id # Use the confirmed session UUID
        )
        # DO NOT schedule log_complete_chat_message here anymore
        # It will be scheduled once after this if/else block
    else:
        # Handle agent error status (this path might be less likely if exceptions above raise HTTPException)
        # If an exception was raised above, this 'else' block might not be reached.
        # However, if agent_status is set to 'error' by process_chat_message without raising an exception
        # that gets caught by the outer try-except, this path is relevant.
        logger.info(f"[ReqID: {request_id}] Agent returned status '{agent_status}' but did not raise an exception caught by the main handler. Constructing error response.")
        if not assistant_content_for_log: # Ensure it's set if not already by an earlier specific error
             assistant_content_for_log = agent_response_data.get("text") or "[Agent Error - No specific message]"

        response = ChatResponse( # Use response instead of final_response for consistency
            status=agent_status,
            request_id=request_id, # Add request_id
            data=None, # No data on error
            error={"message": assistant_content_for_log}, # Include error message
            timestamp=datetime.now(timezone.utc), # Use timezone aware
            session_id=chat_session.session_id
        )

    # --- SINGLE, FINAL COMPLETION LOG --- #
    # This task will be scheduled regardless of success or caught agent error (that led to HTTPException)
    # It relies on assistant_content_for_log being set appropriately in all paths.
    # If an HTTPException was raised, the response is sent before this, but the task is still added.
    background_tasks.add_task(
        log_complete_chat_message,
        db=db,
        request_id=request_id,
        session_id=session_id_str_for_logging_and_agent,
        user_id=user_id,
        organization_id=organization_id,
        user_content=chat_request.message,
        assistant_content=assistant_content_for_log, # Use the determined content
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    logger.debug(f"[ReqID: {request_id}] Scheduled FINAL background task to log complete message.")

    logger.info(f"Completed STATEFUL chat request {request_id} with status: {response.status}")
    return response
    # --- END UNIFIED STATEFUL LOGIC --- #

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
    organization_id = current_user.org_id 

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
        feedback_id_value = str(feedback_record.feedback_id) 
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
