import re
import uuid
from datetime import datetime, timezone
from random import randrange

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.util.logging_utils import request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = RequestIDMiddleware.generate_time_based_uuid()
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @staticmethod
    def get_current_timestamp():
        """
        The function will generate the current timestamp based on local timezone configurations
        :return: timestamp
        """
        current_time_stamp = datetime.now(timezone.utc)
        return current_time_stamp.isoformat()

    @staticmethod
    def generate_time_based_uuid():
        """
        This function will generate the uuid based on time
        :return: UUID
        """
        timestamp = RequestIDMiddleware.get_current_timestamp()
        uid = re.sub(r"[:\.\-\+TZ\s]", "", timestamp)
        rn_num = str(randrange(10**11, 10**12))
        uid = uuid.UUID(f"{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{rn_num}")
        return uid.hex
