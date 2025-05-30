import logging
from fastapi import APIRouter
from starlette import status
from starlette.responses import JSONResponse

from src.models.request.user_login_request import UserLoginRequest
from src.models.request.user_register_request import UserRegisterRequest
from src.service.auth_service import generate_user_auth_tokens
from src.service.user_service import UserService
from src.util.logging_utils import get_logger

logger = get_logger("document_endpoint")
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/v1", tags=["User Registration and Authentication"])


@router.post("/register")
async def register_user(payload: UserRegisterRequest):
    """
    Register a new user.

    This endpoint registers a new user by accepting a payload containing
    user registration details. It delegates the registration process to
    the UserService and returns a JSON response indicating the success
    of the operation.

    Args:
        payload (UserRegisterRequest): The user registration details.

    Returns:
        JSONResponse: A response with a status code of 201 and a message
        indicating successful registration.
    """
    await UserService.register_user_details(payload)

    content = {
        "status":1,
        "message": "User registered successfully",
        "code": status.HTTP_201_CREATED
    }

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=content)


@router.post("/login")
async def login_user(payload: UserLoginRequest):

    user_id = await UserService.validate_user_login_details(payload)

    access_token, refresh_token = await generate_user_auth_tokens(user_id)

    content = {
        "status": 1,
        "message": "User logged in successfully",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "code": status.HTTP_200_OK
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)
