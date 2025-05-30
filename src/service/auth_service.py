from datetime import datetime, timedelta, timezone

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials
import jwt
from passlib.context import CryptContext
from sqlalchemy.future import select

from config import REFRESH_TOKEN_EXPIRATION_DAYS, ACCESS_TOKEN_EXPIRATION_DAYS, SECRET_KEY, ALGORITHM
from src.core.database import SessionLocal
from src.models.entities.user import User
from src.util.custom_http_bearer_utils import CustomHTTPBearer
from src.util.error_utils import raise_http_exception

TOKEN_TYPE_FIELD = "token_type"
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = CustomHTTPBearer()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_jwt_token(user_id: str, is_refresh: bool = False):
    payload = {
        "sub": user_id,
        "exp": (
            datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRATION_DAYS)
            if is_refresh else datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRATION_DAYS)
        ),
        TOKEN_TYPE_FIELD: REFRESH_TOKEN_TYPE if is_refresh else ACCESS_TOKEN_TYPE,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def authenticate_user(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Authenticate a user based on the provided JWT token.

    This function verifies the JWT token's validity, checks the token type,
    and ensures the token has not expired. It retrieves the user associated
    with the token from the database and validates the token against the
    stored access or refresh token.

    Args:
        token (HTTPAuthorizationCredentials): The JWT token credentials
        provided by the client, extracted using the custom bearer scheme.

    Returns:
        str: The user ID if authentication is successful.

    Raises:
        HTTPException: If the token is missing, invalid, expired, or if the
        user is not found.
    """
    if not token or not token.credentials:
        raise_http_exception(status_code=401, message="Token missing or empty")
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        exp = payload.get("exp")
        token_type = payload.get(TOKEN_TYPE_FIELD)

        if not user_id:
            raise_http_exception(status_code=401, message="Invalid token")

        async with SessionLocal() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

            if not user:
                raise_http_exception(status_code=404, message="User not found")

            if token_type == ACCESS_TOKEN_TYPE and user.access_token != token.credentials:
                raise_http_exception(status_code=401, message="Invalid access token")
            elif token_type == REFRESH_TOKEN_TYPE and user.refresh_token != token.credentials:
                raise_http_exception(status_code=401, message="Invalid refresh token")
            elif token_type not in (ACCESS_TOKEN_TYPE, REFRESH_TOKEN_TYPE):
                raise_http_exception(status_code=401, message="Invalid token type")

            expiration_time = datetime.fromtimestamp(exp, tz=timezone.utc)
            if datetime.utcnow().replace(tzinfo=timezone.utc) > expiration_time:
                raise_http_exception(status_code=401, message="Token has expired")

            return user.id

    except jwt.ExpiredSignatureError:
        raise_http_exception(status_code=401, message="Token has expired")
    except jwt.InvalidTokenError:
        raise_http_exception(status_code=401, message="Invalid token")


async def generate_user_auth_tokens(user_id: str):
    """
    Generate and store JWT access and refresh tokens for a user.

    This asynchronous function creates new JWT access and refresh tokens
    for the specified user ID. It updates the user's record in the database
    with the newly generated tokens and commits the changes.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        tuple: A tuple containing the access token and refresh token.
    """
    access_token = create_jwt_token(user_id)
    refresh_token = create_jwt_token(user_id, is_refresh=True)

    async with SessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user:
            user.access_token = access_token
            user.refresh_token = refresh_token
            await session.commit()
    return access_token, refresh_token
