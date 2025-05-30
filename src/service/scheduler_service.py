import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import INDEX_CLEANUP_JOB_INTERVAL_HOURS
from src.service.document_qa_service import DocumentQAService
from src.util.logging_utils import get_logger

logger = get_logger("document_qa_endpoint")
logger.setLevel(logging.INFO)

scheduler = AsyncIOScheduler()


def cleanup_job():
    cleaned = DocumentQAService.clean_old_indexes()
    logger.info(f"FAISS index cleanup done. Remaining indexes: {list(cleaned.keys())}")


async def add_scheduler_job():
    """
    Adds a cleanup job to the scheduler that runs at a specified interval.

    The job is responsible for cleaning up old indexes using the DocumentQAService.
    The interval for the job is determined by the INDEX_CLEANUP_JOB_INTERVAL_HOURS
    configuration. If the scheduler is not already running, it will be started and
    a log message will be recorded.
    """
    scheduler.add_job(cleanup_job, "interval", hours=INDEX_CLEANUP_JOB_INTERVAL_HOURS, next_run_time=None)
    if not scheduler.running:
        logger.info("Scheduler is up and running")
        scheduler.start()
