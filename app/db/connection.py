import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from sqlalchemy import text, inspect, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from contextlib import asynccontextmanager
from fastapi import HTTPException

from app.core.config import settings
from app.db.schema_definitions import SCHEMA_DEFINITIONS

logger = logging.getLogger(__name__)

# Dictionary to store database engines - async only
async_db_engines: Dict[str, AsyncEngine] = {}

# Globally accessible session factory - initialized after engine creation
# Using a more specific name for clarity
chat_management_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

# Create Base for SQLAlchemy models
Base = declarative_base()

# List of database names considered essential for the application to start
ESSENTIAL_DATABASES = ["report_management", "chat_management"] 

async def get_async_db_engine(db_name: str) -> Optional[AsyncEngine]:
    """Get SQLAlchemy async engine for a specific database.
    
    Args:
        db_name: Name of the database to connect to
        
    Returns:
        Async database engine or None if not found
    """
    if db_name not in async_db_engines:
        logger.warning(f"Async database engine for '{db_name}' not found")
        return None
    
    return async_db_engines[db_name]

async def create_db_engines():
    """Create async database engines for all configured databases."""
    global async_db_engines, chat_management_session_factory
    logger.info(f"Found {len(settings.POSTGRES_SERVERS)} database configurations to process.")
    
    all_engines_created = True # Flag to track success
    
    for db_config in settings.POSTGRES_SERVERS:
        db_name = db_config["name"]
        db_url = db_config["url"]
        
        logger.debug(f"Attempting to configure async engine for: {db_config}") 
        
        try:
            # Create async engine
            logger.info(f"Creating async database engine for '{db_name}'")
            
            # Create async engine with asyncpg
            async_engine = _create_async_engine(db_url)
            
            # Test async connection
            try:
                async with async_engine.connect() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    # Do not await fetchone() - it's not a coroutine in SQLAlchemy async API
                    row = result.fetchone()
                    if row and row[0] == 1:
                        logger.info(f"Successfully connected to database '{db_name}' (async)")
                    else:
                        raise ValueError(f"Connection test to '{db_name}' failed: Unexpected result {row}")
            except Exception as e:
                logger.error(f"Async connection test failed for '{db_name}': {str(e)}")
                raise
            
            if db_name not in SCHEMA_DEFINITIONS:
                logger.warning(f"Database '{db_name}' is connected but has no schema definition")
            
            async_db_engines[db_name] = async_engine
            logger.info(f"Async engine for '{db_name}' successfully created and added.")
            
        except Exception as e:
            logger.error(f"Failed to create async database engine for '{db_name}': {str(e)}")
            all_engines_created = False # Mark failure
            # Check if this failed database is essential
            if db_name in ESSENTIAL_DATABASES:
                logger.critical(f"CRITICAL: Failed to connect to essential database '{db_name}'. Application cannot start properly.")
                # Raising an error here will stop the application startup
                raise RuntimeError(f"Failed to connect to essential database: {db_name}") from e
            else:
                 logger.warning(f"Continuing startup despite failure for non-essential database '{db_name}'.")

    if not all_engines_created:
        logger.warning("Some database engines failed to create during startup.")

    # --- Initialize the chat_management session factory --- 
    chat_engine = async_db_engines.get("chat_management")
    if chat_engine:
        chat_management_session_factory = async_sessionmaker(
            bind=chat_engine, 
            expire_on_commit=False, 
            class_=AsyncSession
        )
        logger.info("Successfully created chat_management_session_factory.")
    else:
        if "chat_management" in ESSENTIAL_DATABASES:
             logger.critical("chat_management engine not found after creation attempt, but it's essential. Session factory cannot be created.")
        else:
             logger.warning("chat_management engine not found. Background tasks requiring this DB may fail.")
    # --- End factory initialization ---

def _create_async_engine(db_url: str) -> AsyncEngine:
    """Create an asynchronous SQLAlchemy engine."""
    # Convert the standard PostgreSQL URL to the async version
    async_url = db_url.replace('postgresql://', 'postgresql+asyncpg://')
    
    # Get pool settings, providing higher defaults suitable for concurrency
    pool_size = getattr(settings, 'DB_POOL_SIZE', 100) # Increased default for concurrency
    max_overflow = getattr(settings, 'DB_MAX_OVERFLOW', 200) # Higher overflow buffer
    pool_timeout = getattr(settings, 'DB_POOL_TIMEOUT', 30) # Faster timeout for full pool
    pool_recycle = getattr(settings, 'DB_POOL_RECYCLE', 600) # Recycle connections every 10min
    
    logger.info(f"Creating async engine for {async_url.split('@')[1] if '@' in async_url else '?'} with pool_size={pool_size}, max_overflow={max_overflow}, pool_timeout={pool_timeout}s")

    return create_async_engine(
        async_url,
        pool_size=pool_size,
        max_overflow=max_overflow, # Use configured or higher default value
        pool_timeout=pool_timeout,   # Increase timeout slightly
        pool_recycle=pool_recycle,
        echo=settings.DEBUG, # Keep echo tied to main DEBUG setting
    )

async def get_table_metadata_async(db_name: str) -> Dict[str, List[Dict]]:
    """Get metadata about tables in a database asynchronously.
    
    Args:
        db_name: Name of the database to inspect
        
    Returns:
        Dictionary with tables metadata
    """
    engine = await get_async_db_engine(db_name)
    if not engine:
        return {"tables": []}
    
    # While inspection itself isn't async-native, we can use it within an async context
    async with engine.connect() as conn:
        # Get the underlying connection for inspection
        inspector = inspect(conn)
        tables = []
        
        # Run the table_names method in a sync context
        table_names = await conn.run_sync(inspector.get_table_names)
        
        for table_name in table_names:
            columns = []
            # Run the get_columns method in a sync context
            table_columns = await conn.run_sync(lambda: inspector.get_columns(table_name))
            for column in table_columns:
                columns.append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                })
            
            # Run the pk_constraint method in a sync context
            pk_constraint = await conn.run_sync(lambda: inspector.get_pk_constraint(table_name))
            pk_columns = pk_constraint['constrained_columns']
            
            fk_columns = []
            # Run the foreign_keys method in a sync context
            foreign_keys = await conn.run_sync(lambda: inspector.get_foreign_keys(table_name))
            for fk in foreign_keys:
                fk_columns.append({
                    "column": fk["constrained_columns"][0],
                    "referred_table": fk["referred_table"],
                    "referred_column": fk["referred_columns"][0],
                })
            
            tables.append({
                "name": table_name,
                "columns": columns,
                "primary_keys": pk_columns,
                "foreign_keys": fk_columns,
            })
        
        return {"tables": tables}

async def get_async_session_maker(db_name: str) -> Optional[async_sessionmaker]:
    """Get async session maker for a specific database.
    
    Args:
        db_name: Name of the database
        
    Returns:
        Async session maker or None if engine not found
    """
    engine = await get_async_db_engine(db_name)
    if not engine:
        return None
    
    return async_sessionmaker(expire_on_commit=False, bind=engine)

async def get_async_session(db_name: str) -> Optional[AsyncSession]:
    """Get an async session for a specific database.
    
    Args:
        db_name: Name of the database
        
    Returns:
        AsyncSession or None if engine not found
    """
    session_maker = await get_async_session_maker(db_name)
    if not session_maker:
        return None
    
    return session_maker()

async def validate_schema_definitions():
    """Validate that all SCHEMA_DEFINITIONS tables and columns exist in the actual database."""
    for db_name, engine in async_db_engines.items():
        if db_name not in SCHEMA_DEFINITIONS:
            logger.warning(f"No schema definition for connected database '{db_name}'")
            continue
        
        try:
            async with engine.connect() as conn:
                # For async connections, we need to create an inspector within a run_sync call
                # rather than creating an inspector on the async connection directly
                async def get_tables_and_validate():

                    # Get all table names using inspector inside run_sync
                    def get_table_names(connection):
                        inspector = inspect(connection)
                        return inspector.get_table_names()
                    
                    actual_tables = await conn.run_sync(get_table_names)
                    defined_tables_info = SCHEMA_DEFINITIONS[db_name]["tables"]
                    
                    # Get the physical table names (e.g., "5", "8") defined in the schema
                    defined_physical_tables = [info.get("name") for info in defined_tables_info.values() if info.get("name")]
                    
                    # Check for defined PHYSICAL tables that don't exist in the database
                    missing_physical_tables = [table for table in defined_physical_tables if table not in actual_tables]
                    if missing_physical_tables:
                        # Log using physical names
                        logger.warning(f"Database '{db_name}' is missing these defined tables (physical names): {', '.join(missing_physical_tables)}")
                    
                    # Check for existing tables that aren't represented by a PHYSICAL name in the schema definitions
                    extra_tables = [table for table in actual_tables if table not in defined_physical_tables]
                    if extra_tables:
                        logger.debug(f"Database '{db_name}' has tables not mapped in schema definitions (physical names): {', '.join(extra_tables)}")
                    
                    # Check columns for each defined table that exists physically
                    # Iterate through the logical names (keys) to access column definitions
                    for logical_name, table_info in defined_tables_info.items():
                        physical_name = table_info.get("name")
                        if physical_name and physical_name in actual_tables:
                            # Get columns for this table using inspector inside run_sync
                            def get_columns(connection, table_name):
                                inspector = inspect(connection)
                                return inspector.get_columns(table_name)
                            
                            actual_columns_result = await conn.run_sync(lambda conn: get_columns(conn, physical_name))
                            actual_columns = [col["name"] for col in actual_columns_result]
                            defined_columns = [col["name"] for col in table_info["columns"]]
                            
                            # Check for defined columns that don't exist in the table
                            missing_columns = [col for col in defined_columns if col not in actual_columns]
                            if missing_columns:
                                logger.warning(f"Table '{db_name}.{physical_name}' (logical: {logical_name}) is missing these defined columns: {', '.join(missing_columns)}")
                            
                            # Check for existing columns that aren't in the schema definition
                            extra_columns = [col for col in actual_columns if col not in defined_columns]
                            if extra_columns:
                                logger.debug(f"Table '{db_name}.{physical_name}' (logical: {logical_name}) has columns not in schema definitions: {', '.join(extra_columns)}")
                        elif not physical_name:
                            logger.warning(f"Schema definition for logical table '{logical_name}' in db '{db_name}' is missing a physical 'name' attribute.")
                
                # Execute the validation function
                await get_tables_and_validate()
                
        except Exception as e:
            logger.error(f"Error validating schema for database '{db_name}': {str(e)}")

@asynccontextmanager
async def get_async_db_connection(db_name: str):
    """Async context manager for obtaining and automatically releasing database connections.
    
    Args:
        db_name: Name of the database to connect to
        
    Yields:
        AsyncConnection: The database connection
    """
    engine = await get_async_db_engine(db_name)
    if not engine:
        raise ValueError(f"No async database engine found for '{db_name}'")
    
    async with engine.connect() as conn:
        try:
            yield conn
        finally:
            # Connection will be closed automatically when exiting the context
            pass

@asynccontextmanager
async def get_async_db_session(db_name: str):
    """Async context manager for obtaining and automatically managing database sessions.
    
    Args:
        db_name: Name of the database to connect to
        
    Yields:
        AsyncSession: The database session
    """
    session = await get_async_session(db_name)
    if session is None:
        raise ValueError(f"No async session could be created for '{db_name}'")
    
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
        logger.debug(f"Async session closed for database: {db_name}")

# Initialize database engines
# Note: This is now an async function, so it needs to be called during application startup
# Don't call it directly on module import

# --- FastAPI Dependency for Chat DB Session --- 
async def get_chat_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get an AsyncSession for the 'chat_management' database."""
    # Use the globally initialized factory
    if chat_management_session_factory is None:
        logger.error("chat_management_session_factory is not initialized. Cannot provide session.")
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Database service for chat management is not available."
        )
    
    async with chat_management_session_factory() as session:
        try:
            yield session
            # Only commit here if the endpoint doesn't handle its own commits/rollbacks explicitly.
            # Since repository functions are called with this session, they might handle commit/rollback.
            # Let's assume the endpoint or repo function handles transaction completion.
            # await session.commit() # <-- REMOVED - Let caller manage transaction
        except Exception:
            await session.rollback() # Rollback on any exception during the request using this session
            raise # Re-raise the exception to be handled by FastAPI error handlers
        finally:
            # Session is automatically closed by the context manager `async with`
            logger.debug("Async session closed for database: chat_management in dependency")
# --- End FastAPI Dependency ---
