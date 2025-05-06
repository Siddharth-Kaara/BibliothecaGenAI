import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional # Ensure Optional is imported

from sqlalchemy import (
    ForeignKey, JSON, Text, SMALLINT, CheckConstraint, UniqueConstraint, Integer
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import text # Import text separately

# Import the Base from connection.py where it's defined
from app.db.connection import Base

# Define a default function for UUID generation if needed in Python
# Although the DB has a default, SQLAlchemy sometimes needs this for non-nullable PKs
def generate_uuid():
    return uuid.uuid4()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    # Column definitions matching the CREATE TABLE statement
    # Use Mapped for modern SQLAlchemy type hinting
    session_id: Mapped[uuid.UUID] = mapped_column('sessionId', UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column('userId', UUID(as_uuid=True), nullable=False, index=True)
    organization_id: Mapped[str] = mapped_column('organizationId', Text, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column('createdAt', TIMESTAMP(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP"), index=True)
    updated_at: Mapped[datetime] = mapped_column('updatedAt', TIMESTAMP(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))
    session_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column('metadata', JSONB, nullable=True)

    # Relationships (Points to the ChatMessage class)
    # Use List[] for one-to-many relationships
    messages: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession(session_id='{self.session_id}', user_id='{self.user_id}', org_id='{self.organization_id}')>"

class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    # Column definitions (Python: snake_case, DB: camelCase)
    request_id: Mapped[uuid.UUID] = mapped_column('requestId', UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column('sessionId', ForeignKey('chat_sessions.sessionId', ondelete="CASCADE"), nullable=False, index=True)
    user_message_content: Mapped[Optional[str]] = mapped_column('userMessageContent', Text, nullable=True)
    user_message_timestamp: Mapped[Optional[datetime]] = mapped_column('userMessageTimestamp', TIMESTAMP(timezone=True), nullable=True)
    assistant_message_content: Mapped[Optional[str]] = mapped_column('assistantMessageContent', Text, nullable=True)
    assistant_response_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column('assistantResponseMetadata', JSONB, nullable=True)
    assistant_message_timestamp: Mapped[Optional[datetime]] = mapped_column('assistantMessageTimestamp', TIMESTAMP(timezone=True), nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column('promptTokens', Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column('completionTokens', Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column('createdAt', TIMESTAMP(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP"), index=True)

    # Relationships (Points to ChatSession and InteractionFeedback classes)
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
    feedback_entries: Mapped[List["InteractionFeedback"]] = relationship("InteractionFeedback", back_populates="interaction", cascade="all, delete-orphan")

    def __repr__(self):
        user_msg = f"'{self.user_message_content[:20]}...'" if self.user_message_content else "None"
        asst_msg = f"'{self.assistant_message_content[:20]}...'" if self.assistant_message_content else "None"
        return f"<ChatMessage(request_id='{self.request_id}', session_id='{self.session_id}', user={user_msg}, assistant={asst_msg})>"


class InteractionFeedback(Base):
    __tablename__ = 'interaction_feedback'

    # Define constraints using actual database column names
    __table_args__ = (UniqueConstraint('requestId', 'userId', name='unique_user_request_feedback'), # Use DB column names
                      CheckConstraint('rating >= 1 AND rating <= 5', name='rating_check'))

    # Column definitions
    feedback_id: Mapped[uuid.UUID] = mapped_column('feedbackId', UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    request_id: Mapped[uuid.UUID] = mapped_column('requestId', ForeignKey('chat_messages.requestId', ondelete="SET NULL"), nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column('userId', UUID(as_uuid=True), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(SMALLINT, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column('createdAt', TIMESTAMP(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships (Points back to ChatMessage class)
    interaction: Mapped["ChatMessage"] = relationship("ChatMessage", back_populates="feedback_entries")

    def __repr__(self):
        return f"<InteractionFeedback(feedback_id='{self.feedback_id}', request_id='{self.request_id}', user_id='{self.user_id}', rating={self.rating})>"

# Ensure text is imported where server_default is used (it's imported above now)
# from sqlalchemy import text
