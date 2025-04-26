import logging
from typing import Dict
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

# Store histories in a dictionary keyed by session ID
# NOTE: For production, replace this with a persistent store like RedisChatMessageHistory
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates an in-memory chat message history for a given session ID.
    """
    if session_id not in store:
        logger.info(f"Creating new in-memory chat history for session {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        logger.debug(f"Using existing chat history for session {session_id}")
    return store[session_id]

# Removed add_messages_to_memory, clear_memory, get_chat_history as
# RunnableWithMessageHistory handles message addition, and clearing/getting
# would typically be done via the specific history object instance if needed.
