import logging
from typing import Dict, List, Optional
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import settings
from app.db.schema_definitions import SCHEMA_DEFINITIONS

logger = logging.getLogger(__name__)

# Dictionary to store database engines
db_engines: Dict[str, Engine] = {}

# Create Base for SQLAlchemy models
Base = declarative_base()

# List of database names considered essential for the application to start
ESSENTIAL_DATABASES = ["report_management"] 

def get_db_engine(db_name: str) -> Optional[Engine]:
    """Get SQLAlchemy engine for a specific database.
    
    Args:
        db_name: Name of the database to connect to
        
    Returns:
        Database engine or None if not found
    """
    if db_name not in db_engines:
        logger.warning(f"Database engine for '{db_name}' not found")
        return None
    
    return db_engines[db_name]

def create_db_engines():
    """Create database engines for all configured databases."""
    global db_engines
    logger.info(f"Found {len(settings.POSTGRES_SERVERS)} database configurations to process.")
    
    all_engines_created = True # Flag to track success
    
    for db_config in settings.POSTGRES_SERVERS:
        db_name = db_config["name"]
        db_url = db_config["url"]
        
        logger.debug(f"Attempting to configure engine for: {db_config}") 
        
        try:
            logger.info(f"Creating database engine for '{db_name}'")
            engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                echo=settings.DEBUG,
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info(f"Successfully connected to database '{db_name}'")
            
            if db_name not in SCHEMA_DEFINITIONS:
                logger.warning(f"Database '{db_name}' is connected but has no schema definition")
            
            db_engines[db_name] = engine
            logger.info(f"Engine for '{db_name}' successfully created and added to db_engines.") 
            
        except Exception as e:
            logger.error(f"Failed to create database engine for '{db_name}': {str(e)}")
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

def get_table_metadata(db_name: str) -> Dict[str, List[Dict]]:
    """Get metadata about tables in a database.
    
    Args:
        db_name: Name of the database to inspect
        
    Returns:
        Dictionary with tables metadata
    """
    engine = get_db_engine(db_name)
    if not engine:
        return {"tables": []}
    
    inspector = inspect(engine)
    tables = []
    
    for table_name in inspector.get_table_names():
        columns = []
        for column in inspector.get_columns(table_name):
            columns.append({
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column["nullable"],
            })
        
        pk_columns = inspector.get_pk_constraint(table_name)['constrained_columns']
        fk_columns = []
        for fk in inspector.get_foreign_keys(table_name):
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

def get_session_maker(db_name: str) -> Optional[sessionmaker]:
    """Get session maker for a specific database.
    
    Args:
        db_name: Name of the database
        
    Returns:
        Session maker or None if engine not found
    """
    engine = get_db_engine(db_name)
    if not engine:
        return None
    
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def validate_schema_definitions():
    """Validate that all SCHEMA_DEFINITIONS tables and columns exist in the actual database."""
    for db_name, engine in db_engines.items():
        if db_name not in SCHEMA_DEFINITIONS:
            logger.warning(f"No schema definition for connected database '{db_name}'")
            continue
        
        try:
            inspector = inspect(engine)
            actual_tables = inspector.get_table_names()
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
                logger.info(f"Database '{db_name}' has tables not mapped in schema definitions (physical names): {', '.join(extra_tables)}")
            
            # Check columns for each defined table that exists physically
            # Iterate through the logical names (keys) to access column definitions
            for logical_name, table_info in defined_tables_info.items():
                physical_name = table_info.get("name")
                if physical_name and physical_name in actual_tables:
                    actual_columns = [col["name"] for col in inspector.get_columns(physical_name)]
                    defined_columns = [col["name"] for col in table_info["columns"]]
                    
                    # Check for defined columns that don't exist in the table
                    missing_columns = [col for col in defined_columns if col not in actual_columns]
                    if missing_columns:
                        logger.warning(f"Table '{db_name}.{physical_name}' (logical: {logical_name}) is missing these defined columns: {', '.join(missing_columns)}")
                    
                    # Check for existing columns that aren't in the schema definition
                    extra_columns = [col for col in actual_columns if col not in defined_columns]
                    if extra_columns:
                        logger.info(f"Table '{db_name}.{physical_name}' (logical: {logical_name}) has columns not in schema definitions: {', '.join(extra_columns)}")
                elif not physical_name:
                    logger.warning(f"Schema definition for logical table '{logical_name}' in db '{db_name}' is missing a physical 'name' attribute.")
        
        except Exception as e:
            logger.error(f"Error validating schema for database '{db_name}': {str(e)}")

# Initialize database engines on module import
create_db_engines()

# Validate schema definitions if configured to do so
if settings.VALIDATE_SCHEMA_ON_STARTUP and db_engines:
    validate_schema_definitions()
