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
    sub: str # Standard claim for subject (user ID)
    org_id: Optional[str] = None # Custom claim for organization ID
    # Add other expected claims like exp, iss, aud if needed for validation

class CurrentUser(BaseModel):
    user_id: UUID
    org_id: str # Changed Optional[str]=None to str, assuming org_id is mandatory

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
            # "verify_aud": True, # Uncomment if JWT_AUDIENCE is set
            # "verify_iss": True, # Uncomment if JWT_ISSUER is set
            "verify_exp": True # Always verify expiration
        }
        
        payload = jwt.decode(
            token,
            settings.JWT_PUBLIC_KEY, # Use PUBLIC key for RS256 verification
            algorithms=[settings.JWT_ALGORITHM], # Should be ["RS256"]
            # audience=settings.JWT_AUDIENCE, # Uncomment if needed
            # issuer=settings.JWT_ISSUER,    # Uncomment if needed
            options=options
        )
        
        # Validate payload structure using Pydantic
        token_data = TokenPayload(**payload)
        
        # --- User ID Validation ---
        user_id_str = token_data.sub
        if not user_id_str:
            logger.warning("Token validation failed: 'sub' claim missing.")
            raise invalid_user_exception
        try:
            user_id_uuid = UUID(user_id_str)
        except ValueError:
            logger.warning(f"Token validation failed: 'sub' claim ('{user_id_str}') is not a valid UUID.")
            raise invalid_user_exception
        # --- End User ID Validation ---

        # --- Organization ID Validation ---
        org_id_str = token_data.org_id 
        if not org_id_str: # Make org_id mandatory
             logger.warning("Token validation failed: 'org_id' claim missing.")
             raise credentials_exception # Use generic credentials exception

        # Optional: Validate org_id format if needed (e.g., if it should be UUID)
        # try:
        #     org_id_uuid = UUID(org_id_str)
        # except ValueError:
        #     logger.warning(f"Token validation failed: 'org_id' claim ('{org_id_str}') is not a valid UUID.")
        #     raise credentials_exception
        # --- End Organization ID Validation ---

        # Return validated user and organization IDs
        return CurrentUser(user_id=user_id_uuid, org_id=org_id_str) # Include org_id

    except JWTError as e:
        logger.warning(f"Token validation failed: {e}")
        raise credentials_exception from e
    except ValidationError as e:
        logger.warning(f"Token payload validation failed: {e}")
        raise credentials_exception from e
    except Exception as e: # Catch any other unexpected error during validation
        logger.error(f"Unexpected error during token validation: {e}", exc_info=True)
        raise credentials_exception from e 