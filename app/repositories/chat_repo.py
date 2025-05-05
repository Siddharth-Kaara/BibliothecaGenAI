import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Sequence, Any, Dict

from sqlalchemy import select, update, desc, func, Row, TextClause, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
import sqlalchemy.exc

from app.models.chat import ChatSession, ChatMessage, InteractionFeedback
import logging

logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class RepositoryError(Exception):
    """Base class for repository exceptions."""
    pass

class RecordNotFound(RepositoryError):
    """Indicates a requested record was not found."""
    pass

class SessionMismatchError(RepositoryError):
    """Indicates a mismatch between expected and actual session ID."""
    pass

class FeedbackIntegrityError(RepositoryError):
    """Indicates an integrity issue adding/updating feedback (e.g., conflict)."""
    pass
# --- End Custom Exceptions ---

async def get_or_create_session(
    db: AsyncSession,
    session_id: Optional[uuid.UUID], 
    user_id: uuid.UUID,
    organization_id: str
) -> Tuple[ChatSession, bool]:
    """Gets an existing ChatSession or creates a new one.

    Args:
        db: The AsyncSession instance.
        session_id: The existing session ID (if any).
        user_id: The ID of the user.
        organization_id: The ID of the organization.

    Returns:
        A tuple containing the ChatSession object and a boolean indicating if it was created (True) or existing (False).
    """
    if session_id:
        stmt = select(ChatSession).where(ChatSession.sessionId == session_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        if session:
            session.updatedAt = datetime.now(timezone.utc)
            return session, False
        else:
            logger.warning(f"Requested session_id {session_id} not found, creating new one.")

    new_session = ChatSession(
        userId=user_id, 
        organizationId=organization_id
    )
    db.add(new_session)
    await db.flush()
    await db.refresh(new_session)
    logger.info(f"Created new chat session: {new_session.sessionId}")
    return new_session, True

async def update_session_timestamp(
    db: AsyncSession,
    session_id: uuid.UUID
):
    """Updates the updatedAt timestamp for a given session."""
    stmt = (
        update(ChatSession)
        .where(ChatSession.sessionId == session_id)
        .values(updatedAt=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False)
    )
    await db.execute(stmt)
    logger.debug(f"Updated timestamp for session {session_id}")

async def create_chat_session(db: AsyncSession, user_id: uuid.UUID, organization_id: str) -> ChatSession:
    """Creates a new chat session for a given user and organization."""
    try:
        new_session = ChatSession(
            sessionId=uuid.uuid4(),
            userId=user_id,
            organizationId=organization_id
        )
        db.add(new_session)
        await db.commit()
        logger.info(f"Created new chat session {new_session.sessionId} for user {user_id}")
        return new_session
    except Exception as e:
        error_msg = f"Unexpected error creating chat session for user {user_id}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RepositoryError(error_msg) from e

async def log_initial_chat_message(
    db: AsyncSession,
    request_id: uuid.UUID,
    session_id: str,
    user_id: uuid.UUID,
    organization_id: str,
    user_content: str
) -> None:
    """Logs the initial user message before the agent responds."""
    logger.debug(f"Logging initial user message for request {request_id} in session {session_id}")
    try:
        session_uuid = uuid.UUID(session_id)
        initial_message = ChatMessage(
            requestId=request_id,
            sessionId=session_uuid,
            userMessageContent=user_content,
            userMessageTimestamp=datetime.now(timezone.utc),
        )
        db.add(initial_message)
        await db.commit()
        logger.info(f"Successfully logged initial message for request {request_id}")
    except Exception as e:
        error_msg = f"Unexpected error logging initial message for request {request_id}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RepositoryError(error_msg) from e

async def log_complete_chat_message(
    db: AsyncSession,
    request_id: uuid.UUID,
    session_id: str,
    user_id: uuid.UUID,
    organization_id: str,
    user_content: str,
    assistant_content: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int]
) -> None:
    """Creates or updates a chat message record with the full interaction details."""
    logger.debug(f"Logging complete chat message details for request {request_id} in session {session_id}")
    try:
        session_uuid = uuid.UUID(session_id)
        
        stmt = select(ChatMessage).where(ChatMessage.requestId == request_id)
        result = await db.execute(stmt)
        existing_message = result.scalar_one_or_none()

        if existing_message:
            logger.debug(f"Updating existing chat message record for request {request_id}")
            existing_message.assistantMessageContent = assistant_content
            existing_message.assistantMessageTimestamp = datetime.now(timezone.utc)
            existing_message.promptTokens = prompt_tokens
            existing_message.completionTokens = completion_tokens
            db.add(existing_message)
        else:
            logger.warning(f"No initial log found for request {request_id}. Creating new complete record.")
            new_message = ChatMessage(
                requestId=request_id,
                sessionId=session_uuid,
                userMessageContent=user_content,
                userMessageTimestamp=datetime.now(timezone.utc),
                assistantMessageContent=assistant_content,
                assistantMessageTimestamp=datetime.now(timezone.utc),
                promptTokens=prompt_tokens,
                completionTokens=completion_tokens,
            )
            db.add(new_message)

        await db.commit()
        logger.info(f"Successfully logged complete message details for request {request_id}")
    except Exception as e:
        error_msg = f"Unexpected error logging complete message for request {request_id}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RepositoryError(error_msg) from e

async def get_messages_for_memory(db: AsyncSession, session_id: uuid.UUID, limit: int = 6) -> List[Tuple[str, str]]:
    """Retrieves the last N interaction pairs (user message, assistant message) for agent memory.
    
    Args:
        db: The AsyncSession instance.
        session_id: The session ID to retrieve messages for.
        limit: The maximum number of *messages* (not pairs) to retrieve. Should be even for pairs.

    Returns:
        A list of tuples, where each tuple is (user_message_content, assistant_message_content).
        Returns oldest pairs first.
    """
    if limit % 2 != 0:
        logger.warning(f"Memory limit {limit} is odd; fetching {limit+1} messages to ensure pair completion.")
        limit += 1
        
    stmt = (
        select(
            ChatMessage.userMessageContent,
            ChatMessage.assistantMessageContent
        )
        .where(ChatMessage.sessionId == session_id)
        .where(ChatMessage.assistantMessageContent.isnot(None))
        .order_by(desc(ChatMessage.userMessageTimestamp))
        .limit(limit // 2)
    )
    result = await db.execute(stmt)
    pairs = [(row.userMessageContent, row.assistantMessageContent) for row in result.fetchall()]
    return pairs[::-1] 

async def get_session_history(
    db: AsyncSession,
    session_id: str,
    user_id: uuid.UUID,
    organization_id: str,
    limit: int,
    offset: int
) -> Sequence[ChatMessage]:
    """Retrieves paginated chat history for a specific session, ensuring user ownership."""
    logger.debug(f"Retrieving history for session {session_id}, user {user_id}, org {organization_id}, limit {limit}, offset {offset}")
    try:
        session_uuid = uuid.UUID(session_id)
        stmt = (
            select(ChatMessage)
            .join(ChatSession, ChatMessage.sessionId == ChatSession.sessionId)
            .where(
                ChatMessage.sessionId == session_uuid,
                ChatSession.userId == user_id,
                ChatSession.organizationId == organization_id
            )
            .options(selectinload(ChatMessage.feedback))
            .order_by(ChatMessage.userMessageTimestamp.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        messages = result.scalars().all()
        
        if not messages and offset == 0:
            session_exists_stmt = select(ChatSession).where(
                ChatSession.sessionId == session_uuid,
                ChatSession.userId == user_id,
                ChatSession.organizationId == organization_id
            ).limit(1)
            exists_result = await db.execute(session_exists_stmt)
            if not exists_result.scalar_one_or_none():
                logger.warning(f"Session {session_id} not found for user {user_id} and org {organization_id}. Raising RecordNotFound.")
                raise RecordNotFound(f"Session {session_id} not found for the specified user and organization.")

        logger.info(f"Retrieved {len(messages)} messages for session {session_id}, user {user_id}")
        return messages
    except ValueError:
        logger.warning(f"Invalid session ID format provided: {session_id}")
        raise RecordNotFound("Invalid session ID format.")
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(f"Database error retrieving history for session {session_id}, user {user_id}: {e}", exc_info=True)
        raise RepositoryError("Failed to retrieve chat history due to database error.") from e
    except RecordNotFound:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving history for session {session_id}, user {user_id}: {e}", exc_info=True)
        raise RepositoryError("An unexpected error occurred while retrieving chat history.") from e

async def add_feedback(
    db: AsyncSession,
    request_id: uuid.UUID,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    organization_id: str,
    rating: int
) -> InteractionFeedback:
    """Adds or updates feedback for a given chat message request ID, ensuring user/session consistency."""
    logger.info(f"Adding feedback for request {request_id}, session {session_id}, user {user_id}, rating {rating}")
    try:
        # Query for the specific message, joining with session to verify ownership
        message_stmt = (
            select(ChatMessage)
            .join(ChatSession, ChatMessage.sessionId == ChatSession.sessionId)
            .where(
                ChatMessage.requestId == request_id,
                ChatMessage.sessionId == session_id,
                ChatSession.userId == user_id,
                ChatSession.organizationId == organization_id
            )
        )
        message_result = await db.execute(message_stmt)
        chat_message = message_result.scalar_one_or_none()
        if not chat_message:
            logger.warning(f"Feedback rejected: ChatMessage not found or not authorized for request {request_id}, session {session_id}, user {user_id}, org {organization_id}")
            raise RecordNotFound(f"Original interaction not found or not authorized for request ID {request_id}.")

        # Check for existing feedback for this specific request ID using the correct attribute name
        feedback_stmt = select(InteractionFeedback).where(InteractionFeedback.requestId == request_id)
        feedback_result = await db.execute(feedback_stmt)
        existing_feedback = feedback_result.scalar_one_or_none()

        if existing_feedback:
            # Compare UUID objects directly (DB is UUID, model is UUID, input is UUID)
            if existing_feedback.userId != user_id:
                 logger.error(f"Feedback integrity issue: Attempt by user {user_id} to update feedback originally submitted by user {existing_feedback.userId} for request {request_id}.")
                 raise RecordNotFound("Feedback record conflict or mismatch.")
            logger.info(f"Updating existing feedback for request {request_id}")
            existing_feedback.rating = rating
            feedback_to_return = existing_feedback
            db.add(feedback_to_return)
        else:
            logger.info(f"Creating new feedback for request {request_id}")
            # Create InteractionFeedback object - use model attribute names
            new_feedback = InteractionFeedback(
                feedbackId=uuid.uuid4(),
                requestId=request_id, # Use model attribute name
                userId=user_id, # Use model attribute name
                rating=rating,
            )
            db.add(new_feedback)
            feedback_to_return = new_feedback
            
        # Commit transaction after adding/updating
        await db.commit()
        logger.info(f"Successfully submitted feedback for request {request_id}")
        return feedback_to_return
    except ValueError:
        await db.rollback()
        logger.warning(f"Invalid session ID format provided for feedback: {session_id}")
        raise RecordNotFound("Invalid session ID format provided.")
    except RecordNotFound as e:
        await db.rollback()
        raise e
    except sqlalchemy.exc.IntegrityError as e:
        await db.rollback()
        logger.error(f"Database integrity error submitting feedback for request {request_id}: {e}", exc_info=True)
        raise FeedbackIntegrityError("Failed to submit feedback due to data consistency issue.") from e
    except sqlalchemy.exc.SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error submitting feedback for request {request_id}: {e}", exc_info=True)
        raise RepositoryError("Failed to submit feedback due to database error.") from e
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error submitting feedback for request {request_id}: {e}", exc_info=True)
        raise RepositoryError("An unexpected error occurred while submitting feedback.") from e