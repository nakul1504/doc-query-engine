import logging

from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from src.util.logging_utils import get_logger

logger = get_logger("exception_handler")
logger.setLevel(logging.INFO)


async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles custom HTTP exceptions and returns a JSONResponse with the appropriate status code and error message.

    Args:
        request (Request): The request object used for obtaining additional information.
        exc (HTTPException): The HTTPException instance that was raised.

    Returns:
        JSONResponse: A JSON response with the corresponding status code and error message.
    """
    logger.error(f"HTTP exception occurred for the endpoint: {request.url.path}")
    logger.error(f"HTTP exception occurred: {str(exc)}", exc_info=True)
    error_message = exc.detail["message"]
    content = {
        "status": 0,
        "message": error_message,
        "code": exc.status_code,
    }

    return JSONResponse(status_code=exc.status_code, content=content)


async def general_exception_handler(request: Request, exc: Exception):
    """
    An exception handler for general exceptions that logs the internal server error message and returns a JSON response with a status code of 500 and an error message.
    """
    logger.exception(f"An internal server error occurred {str(exc)}", exc_info=True)
    content = {"status": 0, "message": "Something went wrong", "code": 500}
    return JSONResponse(status_code=500, content=content)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    An exception handler for validation errors that logs the error and returns a JSON response with status code 422 and error messages.

    Args:
        request (Request): The request object used for obtaining additional information.
        exc (RequestValidationError): The RequestValidationError instance that was raised.

    Returns:
        JSONResponse: A JSON response with status code 422 and error messages.
    """
    logger.error(f"Validation error occurred for the endpoint: {request.url.path}")
    logger.error(f"Validation error occurred: {str(exc)}", exc_info=True)
    validation_errors = exc.errors()
    error_messages = []

    for error in validation_errors:
        field = (
            error["loc"][-1]
            if not isinstance(error["loc"][-1], int)
            else error["loc"][-2]
        )
        msg = error["msg"]
        error_messages.append(f"'{field}' {msg.capitalize()}")

    content = {
        "status": 0,
        "message": "Invalid input data",
        "error": error_messages or ["Invalid input data"],
        "code": 422,
    }

    return JSONResponse(status_code=422, content=content)
