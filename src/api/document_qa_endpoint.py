import logging

from fastapi import APIRouter, Depends
from starlette import status
from starlette.responses import JSONResponse

from src.models.request.qa_request import QARequest
from src.service.auth_service import authenticate_user
from src.service.document_qa_service import DocumentQAService
from src.util.logging_utils import get_logger

logger = get_logger("document_qa_endpoint")
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/v1", tags=["Q&A"])


@router.post("/qa")
async def ask_question(payload: QARequest, user_id: str = Depends(authenticate_user)):
    """
    Handles POST requests to the /qa endpoint to process a question-answering request.

    Validates the incoming QARequest payload and generates an answer based on the
    provided question and document ID. The user ID is authenticated using a dependency
    injection pattern. Returns a JSON response containing the answer and a success message.

    Args:
        payload (QARequest): The request payload containing the question and document ID.
        user_id (str): The authenticated user ID, obtained via dependency injection.

    Returns:
        JSONResponse: A response object with the generated answer and HTTP status code 200.
    """
    await DocumentQAService.validate_qa_request(payload)
    result = await DocumentQAService.generate_answer_by_id(payload.question, payload.document_id)

    content = {
        "status": 1,
        "message": "Answer to user query generated successfully",
        "result": result.get("answer", {}),
        "code": status.HTTP_200_OK
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


