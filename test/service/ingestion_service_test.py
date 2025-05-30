import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import UploadFile, HTTPException

from src.service.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_process_file_txt_success():
    """Test successful processing of a valid text file."""

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.txt"
    mock_file.read = AsyncMock(return_value=b"Sample text content")
    mock_user_id = "user-abc"

    with patch("src.service.embeddings_service.EmbeddingsService.embed_and_store_document") as mock_embed:
        mock_embed.return_value = "12345678-1234-5678-1234-567812345678"

        result = await IngestionService.process_document(mock_file, user_id=mock_user_id)

        assert result == {
            "document_id": "12345678-1234-5678-1234-567812345678"
        }

        mock_file.read.assert_awaited_once()
        mock_embed.assert_awaited_once_with(
            text="Sample text content",
            title="test_document.txt",
            owner_id=mock_user_id
        )


@pytest.mark.asyncio
async def test_process_file_pdf_success():
    """Test successful processing of a valid PDF file."""

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.pdf"
    mock_file.read = AsyncMock(return_value=b"PDF content")
    mock_user_id = "user-xyz"

    mock_pdf_reader = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 content"
    mock_pdf_reader.pages = [mock_page1, mock_page2]

    with patch("src.service.ingestion_service.PdfReader", return_value=mock_pdf_reader), \
         patch("src.service.embeddings_service.EmbeddingsService.embed_and_store_document") as mock_embed:

        mock_embed.return_value = "12345678-1234-5678-1234-567812345678"

        result = await IngestionService.process_document(mock_file, user_id=mock_user_id)

        assert result == {
            "document_id": "12345678-1234-5678-1234-567812345678"
        }

        mock_file.read.assert_awaited_once()
        mock_embed.assert_awaited_once_with(
            text="Page 1 content\nPage 2 content",
            title="test_document.pdf",
            owner_id=mock_user_id
        )


@pytest.mark.asyncio
async def test_process_file_unsupported_extension():
    """Test rejection of a file with unsupported extension."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.docx"
    mock_user_id = "user-xyz"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await IngestionService.process_document(mock_file, user_id=mock_user_id)

    assert excinfo.value.status_code == 400
    assert "Only .txt and .pdf files are supported" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_process_file_empty_file():
    """Test rejection of an empty file."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "empty.txt"
    mock_file.read = AsyncMock(return_value=b"")
    mock_user_id = "user-xyz"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await IngestionService.process_document(mock_file, user_id=mock_user_id)

    assert excinfo.value.status_code == 400
    assert "File is empty" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_process_file_txt_decode_error():
    """Test handling of text file with decoding error."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.txt"
    mock_file.read = AsyncMock(return_value=b"\x80\x81")
    mock_user_id = "user-xyz"

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await IngestionService.process_document(mock_file, user_id=mock_user_id)

    assert excinfo.value.status_code == 400
    assert "Failed to decode text file as UTF-8" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_process_file_pdf_parse_error():
    """Test handling of PDF file with parsing error."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "corrupt.pdf"
    mock_file.read = AsyncMock(return_value=b"Not a real PDF")
    mock_user_id = "user-xyz"

    with patch('src.service.ingestion_service.PdfReader', side_effect=Exception("PDF parsing error")):
        # Act & Assert
        with pytest.raises(HTTPException) as excinfo:
            await IngestionService.process_document(mock_file, user_id=mock_user_id)

        assert excinfo.value.status_code == 400
        assert "Failed to parse PDF file" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_process_file_pdf_empty_pages():
    """Test processing PDF with empty pages."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "empty_pages.pdf"
    mock_file.read = AsyncMock(return_value=b"PDF content")

    mock_pdf_reader = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = ""
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = None
    mock_pdf_reader.pages = [mock_page1, mock_page2]
    mock_user_id = "user-xyz"

    with patch('src.service.ingestion_service.PdfReader', return_value=mock_pdf_reader), \
            patch('src.service.embeddings_service.EmbeddingsService.embed_and_store_document') as mock_embed:
        mock_embed.return_value = "12345678-1234-5678-1234-567812345678"

        # Act
        result = await IngestionService.process_document(mock_file, user_id=mock_user_id)

        # Assert
        assert result == {
            "document_id": "12345678-1234-5678-1234-567812345678"
        }
        mock_embed.assert_awaited_once_with(text="\n", title="empty_pages.pdf", owner_id=mock_user_id)


@pytest.mark.asyncio
async def test_process_file_embedding_service_error():
    """Test handling of errors from the embedding service."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.txt"
    mock_file.read = AsyncMock(return_value=b"Sample text content")
    mock_user_id = "user-xyz"

    with patch('src.service.embeddings_service.EmbeddingsService.embed_and_store_document') as mock_embed:
        mock_embed.side_effect = Exception("Embedding service error")

        # Act & Assert
        with pytest.raises(Exception, match="Embedding service error"):
            await IngestionService.process_document(mock_file, user_id=mock_user_id)


@pytest.mark.asyncio
async def test_process_file_large_txt():
    """Test processing of a large text file."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "large_document.txt"
    large_content = b"A" * 1000000  # 1MB of content
    mock_file.read = AsyncMock(return_value=large_content)
    mock_user_id = "user-xyz"

    with patch('src.service.embeddings_service.EmbeddingsService.embed_and_store_document') as mock_embed:
        mock_embed.return_value = "12345678-1234-5678-1234-567812345678"

        # Act
        result = await IngestionService.process_document(mock_file, user_id=mock_user_id)

        # Assert
        assert result == {
            "document_id": "12345678-1234-5678-1234-567812345678"
        }
        mock_embed.assert_awaited_once_with(text="A" * 1000000, title="large_document.txt", owner_id=mock_user_id)


@pytest.mark.asyncio
async def test_process_file_unicode_txt():
    """Test processing of a text file with Unicode characters."""
    # Arrange
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "unicode_document.txt"
    unicode_content = "Unicode text: ￯﾿ﾤ￯ﾾﾽ￯ﾾﾠ￯﾿ﾥ￯ﾾﾥ￯ﾾﾽ, ￯﾿ﾣ￯ﾾﾁ￯ﾾﾓ￯﾿ﾣ￯ﾾﾂ￯ﾾﾓ￯﾿ﾣ￯ﾾﾁ￯ﾾﾫ￯﾿ﾣ￯ﾾﾁ￯ﾾﾡ￯﾿ﾣ￯ﾾﾁ￯ﾾﾯ, ￯﾿ﾬ￯ﾾﾕ￯ﾾﾈ￯﾿ﾫ￯ﾾﾅ￯ﾾﾕ￯﾿ﾭ￯ﾾﾕ￯ﾾﾘ￯﾿ﾬ￯ﾾﾄ￯ﾾﾸ￯﾿ﾬ￯ﾾﾚ￯ﾾﾔ".encode(
        'utf-8')
    mock_file.read = AsyncMock(return_value=unicode_content)
    mock_user_id = "user-xyz"

    with patch('src.service.embeddings_service.EmbeddingsService.embed_and_store_document') as mock_embed:
        mock_embed.return_value = "12345678-1234-5678-1234-567812345678"

        # Act
        result = await IngestionService.process_document(mock_file,user_id=mock_user_id)

        # Assert
        assert result == {
            "document_id": "12345678-1234-5678-1234-567812345678"
        }
        mock_embed.assert_awaited_once_with(
            text="Unicode text: ￯﾿ﾤ￯ﾾﾽ￯ﾾﾠ￯﾿ﾥ￯ﾾﾥ￯ﾾﾽ, ￯﾿ﾣ￯ﾾﾁ￯ﾾﾓ￯﾿ﾣ￯ﾾﾂ￯ﾾﾓ￯﾿ﾣ￯ﾾﾁ￯ﾾﾫ￯﾿ﾣ￯ﾾﾁ￯ﾾﾡ￯﾿ﾣ￯ﾾﾁ￯ﾾﾯ, ￯﾿ﾬ￯ﾾﾕ￯ﾾﾈ￯﾿ﾫ￯ﾾﾅ￯ﾾﾕ￯﾿ﾭ￯ﾾﾕ￯ﾾﾘ￯﾿ﾬ￯ﾾﾄ￯ﾾﾸ￯﾿ﾬ￯ﾾﾚ￯ﾾﾔ",
            title="unicode_document.txt", owner_id=mock_user_id)