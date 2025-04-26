from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator, RootModel

# Chat Models

class ChatMessage(BaseModel):
    """A single message in a chat."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")

class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""
    # user_id: str = Field(..., description="Unique identifier for the user") # Removed
    organization_id: str = Field(..., description="Unique identifier for the user's organization to scope data access")
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")
    chat_history: Optional[List[ChatMessage]] = Field(None, description="Previous chat messages")

# Response Models

# Define the TableData structure
class TableData(BaseModel):
    """Structured table data with separate columns and rows."""
    columns: List[str] = Field(..., description="List of column headers")
    rows: List[List[Any]] = Field(..., description="List of rows, where each row is a list of values corresponding to the columns")

class Visualization(BaseModel):
    """Visualization data for chat responses."""
    type: str = Field(..., description="Type of visualization (bar, pie, line, etc.)")
    image_url: str = Field(..., description="URL or base64 data of the visualization image")

class Error(BaseModel):
    """Error details."""
    code: str = Field(..., description="Error code", examples=["AGENT_ERROR", "CONTEXT_LENGTH_EXCEEDED", "DATABASE_ERROR", "OPENAI_API_ERROR"])
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ChatData(BaseModel):
    """Data for a successful chat response with support for multiple outputs."""
    text: str = Field(..., description="Text response from the chatbot")
    tables: Optional[List[TableData]] = Field(None, description="List of table data (columns and rows), if applicable")
    visualizations: Optional[List[Visualization]] = Field(None, description="List of visualization data, if applicable")

class ChatResponse(BaseModel):
    """Response schema for the chat endpoint."""
    status: str = Field(..., description="Response status (success or error)")
    data: Optional[ChatData] = Field(None, description="Response data if status is success")
    error: Optional[Error] = Field(None, description="Error details if status is error")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    @model_validator(mode='after')
    def check_status_consistency(self):
        """Ensure error is present if status is error and data is present if status is success."""
        if self.status == "error" and self.error is None:
            raise ValueError("Error must be provided when status is error")
        if self.status == "success" and self.data is None:
            raise ValueError("Data must be provided when status is success")
        return self
