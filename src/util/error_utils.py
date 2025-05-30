from fastapi import HTTPException


def raise_http_exception(status_code: int, message: str):
    """
    Raise an HTTPException with a specified status code and message.

    Args:
        status_code (int): The HTTP status code for the exception.
        message (str): The error message to include in the exception details.

    Raises:
        HTTPException: An exception with the given status code and message.
    """
    error_message = {"message": message}
    raise HTTPException(status_code=status_code, detail=error_message)
