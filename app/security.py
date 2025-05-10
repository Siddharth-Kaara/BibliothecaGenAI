import logging
from uuid import UUID, uuid4
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)

# OAuth2 scheme to extract token from Authorization: Bearer <token> header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is not actually used for JWT

class TokenPayload(BaseModel):
    userId: str # Standard claim for subject (user ID) 
    organizationId: Optional[str] = None # Custom claim for organization ID 
    # Add other expected claims like exp, iss, aud if needed for validation

class CurrentUser(BaseModel):
    user_id: UUID
    org_id: str 

async def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    """
    Dependency function to decode and validate JWT token, extract user ID and org ID.
    Raises HTTPException 401 for invalid/expired tokens or missing/invalid IDs.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    invalid_user_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid user identifier in token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Prepare decode options - add audience/issuer validation if configured
        options = {

            "verify_exp": True # Always verify expiration
        }
        
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY, # Use SECRET key for HS256 verification
            algorithms=[settings.JWT_ALGORITHM], # Should be ["HS256"]
            options=options
        )
        
        # Validate payload structure using Pydantic
        token_data = TokenPayload(**payload)
        
        # --- User ID Validation ---
        user_id_str = token_data.userId 
        if not user_id_str:
            logger.warning("Token validation failed: 'userId' claim missing.") 
            raise invalid_user_exception
        try:
            user_id_uuid = UUID(user_id_str)
        except ValueError:
            logger.warning(f"Token validation failed: 'userId' claim ('{user_id_str}') is not a valid UUID.") # Changed from 'sub'
            raise invalid_user_exception

        # --- Organization ID Validation ---
        org_id_str = token_data.organizationId
        if not org_id_str: 
             logger.warning("Token validation failed: 'organizationId' claim missing.") # Changed from 'org_id'
             raise credentials_exception 

        # Return validated user and organization IDs
        return CurrentUser(user_id=user_id_uuid, org_id=org_id_str) 

    except JWTError as e:
        logger.warning(f"Token validation failed: {e}")
        raise credentials_exception from e
    except ValidationError as e:
        logger.warning(f"Token payload validation failed: {e}")
        raise credentials_exception from e
    except Exception as e: 
        logger.error(f"Unexpected error during token validation: {e}", exc_info=True)
        raise credentials_exception from e 