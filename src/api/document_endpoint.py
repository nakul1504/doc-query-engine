import logging

from fastapi import APIRouter, Depends
from starlette import status
from starlette.responses import JSONResponse

from src.service.auth_service import authenticate_user
from src.service.document_service import DocumentService
from src.util.logging_utils import get_logger

logger = get_logger("document_endpoint")
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/v1", tags=["Documents"])


@router.get("/list-documents")
async def list_documents(user_id: str = Depends(authenticate_user)):
    """
    Retrieve a list of documents for the authenticated user.

    Args:
        user_id (str): The ID of the authenticated user, obtained via dependency injection.

    Returns:
        JSONResponse: A JSON response containing the status, message, document data,
        and HTTP status code 200.
    """
    document_data = await DocumentService.get_documents_by_user(user_id=user_id)

    content = {
        "status": 1,
        "message": "User documents fetched successfully",
        "document_data": document_data,
        "code": status.HTTP_200_OK
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)
