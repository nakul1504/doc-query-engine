import logging
from sqlalchemy.exc import IntegrityError
from uuid import uuid4

from sqlalchemy import select

from src.core.database import SessionLocal
from src.models.entities.user import User
from src.models.request.user_login_request import UserLoginRequest
from src.models.request.user_register_request import UserRegisterRequest
from src.service.auth_service import get_password_hash, verify_password
from src.util.error_utils import raise_http_exception
from src.util.logging_utils import get_logger

logger = get_logger("user_service")
logger.setLevel(logging.INFO)

class UserService:
    """
    A service class for handling user registration and login validation.

    Methods:
        register_user_details(payload: UserRegisterRequest):
            Asynchronously registers a new user with the provided details.
            Raises an HTTP exception if the email is already registered.

        validate_user_login_details(payload: UserLoginRequest):
            Asynchronously validates user login credentials.
            Returns the user ID if validation is successful.
            Raises an HTTP exception if credentials are invalid.
    """
    @staticmethod
    async def register_user_details(payload: UserRegisterRequest):
        async with SessionLocal() as session:
            user = User(
                id=str(uuid4()),
                email=payload.email,
                hashed_password=get_password_hash(payload.password)
            )
            session.add(user)
            try:
                await session.commit()
            except IntegrityError:
                raise_http_exception(status_code=400, message="Email already registered")

            logger.info("User registered successfully")

    @staticmethod
    async def validate_user_login_details(payload: UserLoginRequest):
        async with SessionLocal() as session:
            result = await session.execute(select(User).where(User.email == payload.email))
            user = result.scalar_one_or_none()
            if not user or not verify_password(payload.password, user.hashed_password):
                raise_http_exception(status_code=401, message="Invalid credentials")

            logger.info("User login details validated successfully")
            return user.id
