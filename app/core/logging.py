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
    # Lower the level of noisy libraries unless LOG_LEVEL is DEBUG
    default_external_level = logging.WARNING if log_level > logging.DEBUG else logging.INFO
    logging.getLogger("httpx").setLevel(default_external_level)
    logging.getLogger("openai").setLevel(default_external_level) # Base OpenAI client
    logging.getLogger("openai._base_client").setLevel(logging.WARNING) # Reduce spam from raw requests
    logging.getLogger("httpcore").setLevel(logging.WARNING) # Reduce spam from HTTP library
    logging.getLogger("langchain").setLevel(logging.INFO if log_level > logging.DEBUG else logging.DEBUG) # Langchain core
    logging.getLogger("langgraph").setLevel(logging.INFO if log_level > logging.DEBUG else logging.DEBUG) # Langgraph core
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # --- Configure Uvicorn Loggers --- #
    # Get Uvicorn loggers
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    
    # DO NOT clear existing handlers - let Uvicorn use its defaults
    # uvicorn_logger.handlers.clear()
    # uvicorn_error_logger.handlers.clear()
    # uvicorn_access_logger.handlers.clear()

    # Set levels based on the main log level
    uvicorn_level = log_level # Allow Uvicorn DEBUG if main level is DEBUG
    uvicorn_logger.setLevel(uvicorn_level)
    uvicorn_error_logger.setLevel(uvicorn_level)
    uvicorn_access_logger.setLevel(uvicorn_level) # Access logs often desired even if INFO

    # Prevent Uvicorn loggers from propagating to OUR root logger handlers
    uvicorn_logger.propagate = False
    uvicorn_error_logger.propagate = False
    uvicorn_access_logger.propagate = False
    # --- End Uvicorn Configuration --- #

    # --- Configure Application Loggers --- #
    # Ensure our core agent/tool logs are visible based on main LOG_LEVEL
    logging.getLogger("app").setLevel(log_level) # Set base level for our app
    logging.getLogger("app.langchain.agent").setLevel(log_level) # Explicitly match main level
    logging.getLogger("app.langchain.tools").setLevel(log_level) # Set base for tools
    # Reduce verbosity of DB connection logs unless main level is DEBUG
    db_conn_level = logging.INFO if log_level > logging.DEBUG else logging.DEBUG
    logging.getLogger("app.db.connection").setLevel(db_conn_level)
    # Example: If a specific tool is too noisy, set its level higher here
    # logging.getLogger("app.langchain.tools.sql_tool").setLevel(logging.INFO) 
    # --- End Application Logger Configuration --- #

    return root_logger
