from typing import Union

from pydantic import BaseModel, TypeAdapter

from kubectl_mcp_tool.schemas import ErrorResponse


def response_schema(model: type[BaseModel]) -> dict:
    """Declare a success model or shared error, with MCP's object schema root."""
    schema = TypeAdapter(Union[model, ErrorResponse]).json_schema()
    schema["type"] = "object"
    return schema


def structured_response(data: dict, model: type[BaseModel]) -> dict:
    """Validate data against a Pydantic model and return a clean dict.

    FastMCP emits both structuredContent and TextContent when output_schema is set.
    This function validates the data before FastMCP sees it.
    """
    response_model = ErrorResponse if data.get("success") is False else model
    validated = response_model.model_validate(data)
    return validated.model_dump()
