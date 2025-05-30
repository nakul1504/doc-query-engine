import logging

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request

logger = logging.getLogger("auth")


class CustomHTTPBearer(HTTPBearer):
    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        authorization: str = request.headers.get("Authorization")
        if authorization:
            logger.info("Authorization header present")
            return await super().__call__(request)
        logger.warning("Authorization header not present.")
        return None
