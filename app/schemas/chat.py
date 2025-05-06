from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, model_validator, RootModel, ConfigDict
import uuid

# Chat Models

class ChatMessage(BaseModel):
    """A single message in a chat."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")

class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""
    # user_id: str = Field(..., description="Unique identifier for the user") # Removed
    # organization_id: str = Field(..., description="Unique identifier for the user's organization to scope data access") # REMOVED - Get from token
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")
    chat_history: Optional[List[ChatMessage]] = Field(None, description="Previous chat messages")

# Response Models

# Define the TableData structure
class TableData(BaseModel):
    """Structured table data with separate columns and rows."""
    columns: List[str] = Field(..., description="List of column headers")
    rows: List[List[Any]] = Field(..., description="List of rows, where each row is a list of values corresponding to the columns")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the table (e.g., source, description).")

# --- New Chart Specification for API Response ---
class ApiChartSpecification(BaseModel):
    """Defines the specification for a chart to be rendered by the frontend."""
    type_hint: str = Field(..., description="Suggested chart type (e.g., 'bar', 'pie', 'line'). Frontend decides final implementation.")
    title: str = Field(..., description="The title for the chart.")
    x_column: str = Field(..., description="The name of the column from the data to use for the X-axis or labels.")
    y_column: str = Field(..., description="The name of the column from the data to use for the Y-axis or values.")
    color_column: Optional[str] = Field(default=None, description="Optional: The name of the column to use for grouping data by color/hue.")
    x_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the X-axis.")
    y_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the Y-axis.")
    data: TableData = Field(..., description="The actual data for the chart, matching the TableData structure.")

class Error(BaseModel):
    """Error details."""
    code: str = Field(..., description="Error code", examples=["AGENT_ERROR", "CONTEXT_LENGTH_EXCEEDED", "DATABASE_ERROR", "OPENAI_API_ERROR"])
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ChatData(BaseModel):
    """Data for a successful chat response with support for multiple outputs."""
    model_config = ConfigDict(from_attributes=True)

    text: str = Field(..., description="Text response from the chatbot")
    tables: Optional[List[TableData]] = Field(None, description="List of table data (columns and rows), if applicable")
    visualizations: List[ApiChartSpecification] = Field(default_factory=list)

class ChatResponse(BaseModel):
    """Response schema for the chat endpoint."""
    model_config = ConfigDict(from_attributes=True)

    request_id: Optional[uuid.UUID] = Field(None, description="Unique identifier for the specific request-response cycle, present on success.")
    status: Literal["success", "error", "partial_success"] = "success"
    data: Optional[ChatData] = None
    error: Optional[Error] = None
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    # Session ID must always be present for stateful chat responses
    session_id: uuid.UUID

    @model_validator(mode='after')
    def check_status_consistency(self):
        """Ensure error is present if status is error and data is present if status is success."""
        if self.status == "error" and self.error is None:
            raise ValueError("Error must be provided when status is error")
        if self.status == "success" and self.data is None:
            raise ValueError("Data must be provided when status is success")
        return self

# --- Add Feedback Schema ---
class FeedbackRequest(BaseModel):
    request_id: uuid.UUID = Field(..., description="The unique ID of the ChatMessage being rated.")
    rating: int = Field(..., ge=0, le=1, description="User rating (0=Thumbs Down, 1=Thumbs Up).") # Changed to 0 and 1
    comment: Optional[str] = Field(None, description="Optional user comment.")
    session_id: uuid.UUID = Field(..., description="The ID of the chat session for context.")

# Define the response schema for feedback submission
class FeedbackResponse(BaseModel):
    feedback_id: str # Changed to str as API returns stringified UUID
    message: str = "Feedback submitted successfully."
# --- End Feedback Schema ---

# --- History Schemas ---
class ChatMessageForHistory(BaseModel):
    # Mirrors ChatMessage initially, but allows for future field selection/modification
    request_id: uuid.UUID = Field(..., description="Unique ID for the specific user request/assistant response pair.")
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Timestamp of the message")
    # Add feedback info if needed in the future, e.g.:
    # rating: Optional[int] = None
    # feedback_comment: Optional[str] = None

    class Config:
        from_attributes = True # <-- Changed from orm_mode

class HistoryResponse(BaseModel):
    session_id: uuid.UUID = Field(..., description="The ID of the chat session.")
    messages: List[ChatMessageForHistory] = Field(..., description="List of chat messages in chronological order.")
# --- End History Schemas ---
