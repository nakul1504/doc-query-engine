import logging
import uuid
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import SENTENCE_TRANSFORMER_MODEL
from src.core.database import SessionLocal
from src.models.entities.document_entity import Document
from src.util.logging_utils import get_logger

logger = get_logger("embeddings_service")
logger.setLevel(logging.INFO)
class EmbeddingsService:
    """
    EmbeddingsService class provides functionality to embed text using a pre-trained
    sentence transformer model and store the resulting embedding in the database.

    Attributes:
        model (SentenceTransformer): An instance of the SentenceTransformer initialized
            with a specified model for generating embeddings.

    Methods:
        embed_and_store_document(text: str, title: str, owner_id: str) -> str:
            Asynchronously embeds the given text, stores it in the database as a Document
            entity, and returns the generated document ID.

            Args:
                text (str): The content of the document to be embedded.
                title (str): The title of the document.
                owner_id (str): The ID of the owner of the document.

            Returns:
                str: The unique identifier of the stored document.
    """
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    @staticmethod
    async def embed_and_store_document(text: str, title: str, owner_id: str) -> str:
        doc_id = str(uuid.uuid4())
        embedding = EmbeddingsService.model.encode([text])[0].tolist()

        async with SessionLocal() as session:
            doc = Document(id=doc_id, title=title, content=text, embedding=embedding, owner_id=owner_id)
            session.add(doc)
            await session.commit()

        return doc_id
