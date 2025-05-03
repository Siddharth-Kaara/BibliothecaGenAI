import uuid
import datetime
import decimal


def json_default(obj):
    """JSON serializer handling specific types and number formatting.

    Handles:
    - UUID -> str
    - datetime/date/time -> ISO format str
    - float representing whole number (e.g., 234.0) -> int (234)
    - Decimal representing whole number -> int
    - Other Decimal -> str (to preserve precision)
    """
    if isinstance(obj, uuid.UUID):
        # Convert UUID to string
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    # --- NEW: Handle floats representing whole numbers ---
    if isinstance(obj, float) and obj.is_integer():
        return int(obj) # Convert 234.0 -> 234
    # --- Handle Decimals carefully ---
    if isinstance(obj, decimal.Decimal):
        # If it's a whole number Decimal, convert to int
        if obj % 1 == 0:
            return int(obj)
        # Otherwise, convert to string to preserve precision (avoids float conversion)
        else:
            return str(obj)
    # Let the base class default method raise the TypeError for other types
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def clean_json_response(response: str) -> str:
    """Clean JSON response string from LLM output, removing markdown code fences."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip() 