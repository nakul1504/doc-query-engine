import logging

from fastapi import UploadFile
from PyPDF2 import PdfReader
from io import BytesIO

from src.service.embeddings_service import EmbeddingsService
from src.util.error_utils import raise_http_exception
from src.util.logging_utils import get_logger

logger = get_logger("ingestion_service")
logger.setLevel(logging.INFO)

class IngestionService:
    """
        Asynchronously processes an uploaded document file by validating its type,
        extracting text content, and storing it with embeddings.

        Args:
            file (UploadFile): The uploaded document file, expected to be a .txt or .pdf.
            user_id (str): The ID of the user uploading the document.

        Returns:
            dict: A dictionary containing the unique document ID.

        Raises:
            HTTPException: If the file type is unsupported, the file is empty, or
                           there are errors in decoding or parsing the file.
        """
    @staticmethod
    async def process_document(file: UploadFile, user_id: str):
        if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
            raise_http_exception(
                status_code=400, message="Only .txt and .pdf files are supported"
            )

        content = await file.read()
        if not content:
            raise_http_exception(status_code=400, message="File is empty")

        if file.filename.endswith(".txt"):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                logger.error("Error while decoding text file as UTF-8")
                raise_http_exception(
                    status_code=400, message="Failed to decode text file as UTF-8"
                )
        elif file.filename.endswith(".pdf"):
            try:
                reader = PdfReader(BytesIO(content))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                logger.error("Failed to parse PDF file")
                raise_http_exception(
                    status_code=400, message="Failed to parse PDF file"
                )

        doc_id = await EmbeddingsService.embed_and_store_document(
            text=text, title=file.filename, owner_id=user_id
        )
        return {
            "document_id": doc_id
        }
