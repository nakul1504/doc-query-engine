import logging

from fastapi import APIRouter

from src.core.database import create_tables
from src.service.scheduler_service import add_scheduler_job
from src.util.logging_utils import get_logger

logger = get_logger("database_events")
logger.setLevel(logging.INFO)

router = APIRouter()


@router.on_event("startup")
async def startup():
    """
    Handles the startup event for the FastAPI application.

    This function is triggered when the application starts up and ensures
    that the necessary database tables are created by calling the
    `create_tables` function.
    """
    await create_tables()
    await add_scheduler_job()
