import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

def setup_logging():
    """Configure logging for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_dir / "bibliotheca_api.log",
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels, respecting the main debug level
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING) # Added to reduce font scan noise

    # Allow Uvicorn debug logs if main level is debug
    uvicorn_level = log_level if log_level == logging.DEBUG else logging.INFO
    logging.getLogger("uvicorn").setLevel(uvicorn_level)
    logging.getLogger("uvicorn.error").setLevel(uvicorn_level) # Also control uvicorn error logger
    logging.getLogger("uvicorn.access").setLevel(uvicorn_level) # And access logger

    # Allow OpenAI/LangChain debug logs if main level is debug
    # Note: LangChain might use other logger names too, but 'openai' is common for LLM calls
    openai_level = log_level if log_level == logging.DEBUG else logging.WARNING
    logging.getLogger("openai").setLevel(openai_level) # Base OpenAI client
    logging.getLogger("openai._base_client").setLevel(logging.WARNING) # Reduce spam from raw requests
    logging.getLogger("httpcore").setLevel(logging.WARNING) # Reduce spam from HTTP library
    
    # Potentially add other LangChain/LangGraph loggers if needed, e.g., 'langchain', 'langgraph'
    logging.getLogger("langchain").setLevel(log_level) # Explicitly set langchain level
    logging.getLogger("langgraph").setLevel(log_level) # Explicitly set langgraph level
    # Force the agent logger to INFO to reduce verbosity, regardless of main log_level
    logging.getLogger("app.langchain.agent").setLevel(logging.INFO)

    return root_logger
