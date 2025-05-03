import uuid
import datetime
import decimal


def json_default(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, uuid.UUID):
        # Convert UUID to string
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        # Keep this for potential future use or if params contain dates
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        # Convert Decimal to float for JSON serialization
        # Use str(obj) if exact decimal representation as string is needed
        return float(obj)
    # Let the base class default method raise the TypeError
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