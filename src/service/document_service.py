import logging

from sqlalchemy.future import select

from src.core.database import SessionLocal
from src.models.entities.document_entity import Document
from src.util.logging_utils import get_logger

logger = get_logger("document_service")
logger.setLevel(logging.INFO)

class DocumentService:
    """
    Fetches documents associated with a specific user.

    Args:
        user_id (str): The ID of the user whose documents are to be retrieved.

    Returns:
        list: A list of dictionaries, each containing the document ID and title.
    """
    @staticmethod
    async def get_documents_by_user(user_id: str):
        logger.info(f"Fetching documents for user {user_id}")
        async with SessionLocal() as session:
            result = await session.execute(select(Document).where(Document.owner_id == user_id))
            return [
                {"document_id": doc.id, "document_title": doc.title} for doc in result.scalars().all()
            ]
