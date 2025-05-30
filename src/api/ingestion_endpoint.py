import logging

from fastapi import APIRouter, UploadFile, File, Depends
from starlette import status
from starlette.responses import JSONResponse

from src.service.auth_service import authenticate_user
from src.service.ingestion_service import IngestionService
from src.util.logging_utils import get_logger

logger = get_logger("ingestion_endpoint")
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...), user_id: str = Depends(authenticate_user)):
    """
    Handles the ingestion of a document by processing the uploaded file and returning
    a JSON response with the document ID and a success message.

    Args:
        file (UploadFile): The file to be ingested, uploaded by the user.
        user_id (str): The ID of the authenticated user, obtained via dependency injection.

    Returns:
        JSONResponse: A response containing the status, message, document ID, and HTTP status code.
    """
    document_data = await IngestionService.process_document(file=file, user_id=user_id)

    content = {
        "status": 1,
        "message": "Document ingested successfully",
        "document_id": document_data.get("document_id", ""),
        "code": status.HTTP_201_CREATED
    }

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=content)
