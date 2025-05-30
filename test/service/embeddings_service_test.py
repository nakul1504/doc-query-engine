import pytest
import uuid
from unittest.mock import patch, AsyncMock
import numpy as np

from src.service.embeddings_service import EmbeddingsService


@pytest.mark.asyncio
async def test_embed_and_store_document_success():
    """Test successful embedding and storing of a document with valid inputs."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
         patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
         patch('src.service.embeddings_service.SessionLocal') as mock_session_context:

        mock_encode.return_value = [np.array([0.1] * 1024)]

        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = "Sample document text"
        title = "Sample Title"
        owner_id = "user-abc"

        doc_id = await EmbeddingsService.embed_and_store_document(text, title, owner_id)

        assert doc_id == "12345678-1234-5678-1234-567812345678"
        mock_encode.assert_called_once_with([text])
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

        doc_arg = mock_session.add.call_args[0][0]
        assert doc_arg.id == doc_id
        assert doc_arg.title == title
        assert doc_arg.content == text
        assert doc_arg.owner_id == owner_id
        assert doc_arg.embedding == [0.1] * 1024


@pytest.mark.asyncio
async def test_embed_and_store_document_empty_text():
    """Test embedding and storing with empty text."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
         patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
         patch('src.service.embeddings_service.SessionLocal') as mock_session_context:

        mock_encode.return_value = [np.array([0.0] * 1024)]

        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = ""
        title = "Empty Document"
        owner_id = "user-xyz"

        doc_id = await EmbeddingsService.embed_and_store_document(text, title, owner_id)

        assert doc_id == "12345678-1234-5678-1234-567812345678"
        mock_encode.assert_called_once_with([text])

        doc_arg = mock_session.add.call_args[0][0]
        assert doc_arg.content == ""
        assert doc_arg.owner_id == owner_id


@pytest.mark.asyncio
async def test_embed_and_store_document_long_text():
    """Test embedding and storing with very long text."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
         patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
         patch('src.service.embeddings_service.SessionLocal') as mock_session_context:

        mock_encode.return_value = [np.array([0.2] * 1024)]

        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = "A" * 10000
        title = "Long Document"
        owner_id = "user-long"

        doc_id = await EmbeddingsService.embed_and_store_document(text, title, owner_id)

        assert doc_id == "12345678-1234-5678-1234-567812345678"
        mock_encode.assert_called_once_with([text])

        doc_arg = mock_session.add.call_args[0][0]
        assert doc_arg.content == text
        assert len(doc_arg.content) == 10000
        assert doc_arg.owner_id == owner_id


@pytest.mark.asyncio
async def test_embed_and_store_document_special_characters():
    """Test embedding and storing with text containing special characters."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
            patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
            patch('src.service.embeddings_service.SessionLocal') as mock_session_context:
        mock_encode.return_value = [np.array([0.3] * 1024)]

        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = "Special characters: !@#$%^&*()_+{}|:<>?[];\',./`~"
        title = "Special Chars Document"
        owner_id = "user-long"

        # Act
        doc_id = await EmbeddingsService.embed_and_store_document(text, title, owner_id)

        # Assert
        assert doc_id == "12345678-1234-5678-1234-567812345678"
        mock_encode.assert_called_once_with([text])

        # Verify the document was created with the special characters
        doc_arg = mock_session.add.call_args[0][0]
        assert doc_arg.content == text
        assert doc_arg.owner_id == owner_id


@pytest.mark.asyncio
async def test_embed_and_store_document_model_error():
    """Test handling of errors from the embedding model."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
            patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode:
        mock_encode.side_effect = Exception("Model error")

        text = "Sample document text"
        title = "Sample Title"
        owner_id = "user-long"

        # Act & Assert
        with pytest.raises(Exception, match="Model error"):
            await EmbeddingsService.embed_and_store_document(text, title, owner_id)


@pytest.mark.asyncio
async def test_embed_and_store_document_db_error():
    """Test handling of database errors."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
            patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
            patch('src.service.embeddings_service.SessionLocal') as mock_session_context:
        mock_encode.return_value = [np.array([0.5] * 1024)]

        mock_session = AsyncMock()
        mock_session.commit.side_effect = Exception("Database error")
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = "Sample document text"
        title = "Sample Title"
        owner_id = "user-long"

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await EmbeddingsService.embed_and_store_document(text, title, owner_id)


@pytest.mark.asyncio
async def test_embed_and_store_document_unicode_text():
    """Test embedding and storing with Unicode text."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
            patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
            patch('src.service.embeddings_service.SessionLocal') as mock_session_context:
        mock_encode.return_value = [np.array([0.6] * 1024)]

        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = "Unicode text: ￤ﾽﾠ￥ﾥﾽ, ￣ﾁﾓ￣ﾂﾓ￣ﾁﾫ￣ﾁﾡ￣ﾁﾯ, ￬ﾕﾈ￫ﾅﾕ￭ﾕﾘ￬ﾄﾸ￬ﾚﾔ"
        title = "Unicode Document"
        owner_id = "user-long"

        # Act
        doc_id = await EmbeddingsService.embed_and_store_document(text, title, owner_id)

        # Assert
        assert doc_id == "12345678-1234-5678-1234-567812345678"
        mock_encode.assert_called_once_with([text])

        # Verify the document was created with the Unicode text
        doc_arg = mock_session.add.call_args[0][0]
        assert doc_arg.content == text
        assert doc_arg.owner_id == owner_id


@pytest.mark.asyncio
async def test_embed_and_store_document_embedding_shape():
    """Test that the embedding has the expected shape."""
    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')), \
            patch('src.service.embeddings_service.EmbeddingsService.model.encode') as mock_encode, \
            patch('src.service.embeddings_service.SessionLocal') as mock_session_context:
        # Create a realistic embedding with the expected dimensionality
        embedding = np.random.rand(1024).astype(np.float32)
        mock_encode.return_value = [embedding]

        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        text = "Sample document text"
        title = "Sample Title"
        owner_id = "user-long"

        # Act
        doc_id = await EmbeddingsService.embed_and_store_document(text, title, owner_id)

        # Assert
        doc_arg = mock_session.add.call_args[0][0]
        assert len(doc_arg.embedding) == 1024
        assert isinstance(doc_arg.embedding, list)