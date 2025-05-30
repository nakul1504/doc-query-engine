from datetime import datetime, timedelta

import pytest

import json
from unittest.mock import patch, AsyncMock, MagicMock, mock_open, Mock

from fastapi import HTTPException

from src.models.request.qa_request import QARequest
from src.service.document_qa_service import DocumentQAService

from src.models.entities.document_entity import Document as DocumentModel


@pytest.fixture
def mock_document():
    """Fixture for a mock document."""
    return DocumentModel(
        id="12345678-1234-5678-1234-567812345678",
        title="Test Document",
        content="This is a test document with some content for testing. It contains information about testing.",
        embedding=[0.1] * 1024
    )


@pytest.mark.asyncio
async def test_validate_qa_request_valid():
    """Test validation with valid QA request."""
    # Arrange
    qa_request = QARequest(
        question="What is this document about?",
        document_id="12345678-1234-5678-1234-567812345678"
    )

    # Act
    # No exception should be raised
    await DocumentQAService.validate_qa_request(qa_request)

    # Assert - if we get here without exception, the test passes


@pytest.mark.asyncio
async def test_validate_qa_request_empty_question():
    """Test validation with empty question."""
    # Arrange
    qa_request = QARequest(
        question="",
        document_id="12345678-1234-5678-1234-567812345678"
    )

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await DocumentQAService.validate_qa_request(qa_request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["message"] == "Question cannot be empty"


@pytest.mark.asyncio
async def test_validate_qa_request_empty_document_id():
    """Test validation with empty document ID."""

    qa_request = QARequest(
        question="What is this document about?",
        document_id=""
    )

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await DocumentQAService.validate_qa_request(qa_request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["message"] == "Document ID is required"


@pytest.mark.asyncio
async def test_generate_answer_document_not_found():
    """Test generating answer when document is not found."""

    question = "What is this document about?"
    document_id = "non-existent-id"

    mock_session = AsyncMock()
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None  # ✅ simulate sync method
    mock_session.execute = AsyncMock(return_value=mock_result)

    with patch("src.service.document_qa_service.SessionLocal") as mock_session_local:
        mock_session_local.return_value.__aenter__.return_value = mock_session

        result = await DocumentQAService.generate_answer_by_id(question, document_id)

        # Assert
        assert result == {"answer": f"No document found with id {document_id}"}
        mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_success():
    """Test successful answer generation."""

    question = "What is this document about?"
    document_id = "test-doc-id"
    expected_answer = "This document is about testing."

    # Setup mock document
    mock_document = MagicMock()
    mock_document.id = document_id
    mock_document.content = (
        "This is a test document with some content for testing. "
        "It contains information about testing."
    )

    # Mock DB session and result
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_document
    mock_session.execute.return_value = mock_result

    # Mock spaCy NLP response
    mock_doc_nlp = MagicMock()
    mock_doc_nlp.sents = [
        MagicMock(text="This is a test document with some content for testing."),
        MagicMock(text="It contains information about testing.")
    ]

    # Mock FAISS + RAG
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = expected_answer

    with patch("src.service.document_qa_service.SessionLocal") as mock_session_local, \
         patch.object(DocumentQAService, "nlp", return_value=mock_doc_nlp) as mock_nlp, \
         patch.object(DocumentQAService, "load_faiss_index", return_value=mock_vectorstore) as mock_load_index, \
         patch("src.service.document_qa_service.FAISS"), \
         patch("src.service.document_qa_service.RetrievalQA") as mock_retrieval_qa:

        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_retrieval_qa.from_chain_type.return_value = mock_rag_chain

        # Act
        result = await DocumentQAService.generate_answer_by_id(question, document_id)

    # ✅ Assert
    assert result == {"answer": expected_answer}
    mock_session.execute.assert_called_once()
    mock_nlp.assert_called_once()
    mock_load_index.assert_called_once_with(document_id)
    mock_vectorstore.as_retriever.assert_called_once()
    mock_rag_chain.invoke.assert_called_once_with(question)
    mock_retrieval_qa.from_chain_type.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_create_new_index():
    """Test answer generation when FAISS index doesn't exist."""

    question = "What is this document about?"
    document_id = "test-doc-id"
    expected_answer = "This document is about testing."

    # Setup mock document
    mock_document = MagicMock()
    mock_document.id = document_id
    mock_document.content = (
        "This is a test document with some content for testing. "
        "It contains information about testing."
    )

    # Mock DB session and scalar_one_or_none to return our document
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_document
    mock_session.execute.return_value = mock_result

    # Mock spaCy doc
    mock_doc_nlp = MagicMock()
    mock_doc_nlp.sents = [
        MagicMock(text="This is a test document with some content for testing."),
        MagicMock(text="It contains information about testing.")
    ]

    # Mock FAISS and Retrieval chain
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = expected_answer

    with patch("src.service.document_qa_service.SessionLocal") as mock_session_local, \
         patch.object(DocumentQAService, "nlp", return_value=mock_doc_nlp), \
         patch.object(DocumentQAService, "load_faiss_index", return_value=None) as mock_load_index, \
         patch.object(DocumentQAService, "save_faiss_index") as mock_save_index, \
         patch("src.service.document_qa_service.FAISS") as mock_faiss, \
         patch("src.service.document_qa_service.RetrievalQA") as mock_retrieval_qa:

        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_faiss.from_documents.return_value = mock_vectorstore
        mock_retrieval_qa.from_chain_type.return_value = mock_rag_chain

        result = await DocumentQAService.generate_answer_by_id(question, document_id)

    # ✅ Assert output
    assert result == {"answer": expected_answer}
    mock_load_index.assert_called_once_with(document_id)
    mock_faiss.from_documents.assert_called_once()
    mock_save_index.assert_called_once_with(mock_vectorstore, document_id)


@pytest.mark.asyncio
async def test_generate_answer_long_document_chunking():
    """Test chunking of long documents."""

    question = "What is this document about?"
    document_id = "test-document-id"

    # Mock document with long content
    long_content = " ".join([f"This is sentence number {i}." for i in range(100)])

    mock_document = MagicMock()
    mock_document.id = document_id
    mock_document.content = long_content

    # Mock DB session result
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_document
    mock_session.execute.return_value = mock_result

    # Mock spaCy output: 100 sentence objects
    mock_doc_nlp = MagicMock()
    mock_doc_nlp.sents = [MagicMock(text=f"This is sentence number {i}.") for i in range(100)]

    # Mock FAISS, retriever, RAG chain
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = "Answer about testing"

    with patch("src.service.document_qa_service.SessionLocal") as mock_session_local, \
         patch.object(DocumentQAService, "nlp", return_value=mock_doc_nlp), \
         patch.object(DocumentQAService, "load_faiss_index", return_value=mock_vectorstore), \
         patch("src.service.document_qa_service.FAISS"), \
         patch("src.service.document_qa_service.LangchainDoc") as mock_langchain_doc, \
         patch("src.service.document_qa_service.RetrievalQA") as mock_retrieval_qa:

        # Setup session context
        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_retrieval_qa.from_chain_type.return_value = mock_rag_chain

        result = await DocumentQAService.generate_answer_by_id(question, document_id)

    # ✅ Assertions
    assert isinstance(result, dict)
    assert result["answer"] == "Answer about testing"

    # ✅ Multiple chunks should be created → multiple calls to LangchainDoc
    assert mock_langchain_doc.call_count > 1


@pytest.mark.asyncio
async def test_load_faiss_index_exists():
    """Test loading FAISS index when it exists."""

    document_id = "12345678-1234-5678-1234-567812345678"

    mock_vectorstore = MagicMock()

    with patch("src.service.document_qa_service.Path.exists", return_value=True), \
         patch("src.service.document_qa_service.FAISS.load_local", return_value=mock_vectorstore) as mock_load_local, \
         patch("src.service.document_qa_service.DocumentQAService.load_metadata", return_value={}), \
         patch("src.service.document_qa_service.DocumentQAService.save_metadata") as mock_save_metadata:

        result = DocumentQAService.load_faiss_index(document_id)

    assert result == mock_vectorstore
    mock_load_local.assert_called_once_with(
        str(DocumentQAService.INDEX_DIR / f"{document_id}.faiss"),
        DocumentQAService.embedding_model,
        allow_dangerous_deserialization=True
    )
    mock_save_metadata.assert_called_once()


@pytest.mark.asyncio
async def test_load_faiss_index_not_exists():
    """Test loading FAISS index when it doesn't exist."""
    # Arrange
    document_id = "12345678-1234-5678-1234-567812345678"

    # Act
    with patch('src.service.document_qa_service.Path.exists') as mock_exists:
        mock_exists.return_value = False
        result = DocumentQAService.load_faiss_index(document_id)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_save_faiss_index():
    """Test saving FAISS index."""
    document_id = "12345678-1234-5678-1234-567812345678"
    mock_vectorstore = MagicMock()
    mock_metadata = {}

    with patch('src.service.document_qa_service.Path.exists') as mock_exists, \
            patch('src.service.document_qa_service.Path.mkdir') as mock_mkdir, \
            patch('builtins.open', mock_open()) as mock_file, \
            patch('json.dump') as mock_json_dump:
        mock_exists.return_value = False

        DocumentQAService.save_faiss_index(mock_vectorstore, document_id)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_vectorstore.save_local.assert_called_once()
    mock_json_dump.assert_called_once()


@pytest.mark.parametrize(
    "metadata_content,expected_result,expected_removed",
    [
        # Case 1: All documents are recent, none should be removed
        (
            {
                "doc1": {"last_access": (datetime(2023, 1, 8)).isoformat()},
                "doc2": {"last_access": (datetime(2023, 1, 6)).isoformat()},
            },
            {
                "doc1": {"last_access": (datetime(2023, 1, 8)).isoformat()},
                "doc2": {"last_access": (datetime(2023, 1, 6)).isoformat()},
            },
            []
        ),
        # Case 2: Some documents are old, should be removed
        (
            {
                "doc1": {"last_access": (datetime(2023, 1, 8)).isoformat()},
                "doc2": {"last_access": (datetime(2022, 12, 30)).isoformat()},
            },
            {
                "doc1": {"last_access": (datetime(2023, 1, 8)).isoformat()},
            },
            ["doc2"]
        ),
        # Case 3: All documents are old, all should be removed
        (
            {
                "doc1": {"last_access": (datetime(2022, 12, 28)).isoformat()},
                "doc2": {"last_access": (datetime(2022, 12, 30)).isoformat()},
            },
            {},
            ["doc1", "doc2"]
        ),
        # Case 4: Empty metadata, nothing to remove
        (
            {},
            {},
            []
        ),
        # Case 5: Edge case - exactly at threshold (should not be removed)
        (
            {
                "doc1": {"last_access": (datetime(2023, 1, 1)).isoformat()},
            },
            {
                "doc1": {"last_access": (datetime(2023, 1, 1)).isoformat()},
            },
            []
        ),
    ]
)
def test_clean_old_indexes(metadata_content, expected_result, expected_removed):
    """Test cleaning old indexes with various metadata scenarios."""

    fake_now = datetime(2023, 1, 8)  # Fixed current time
    with patch('src.service.document_qa_service.DocumentQAService.load_metadata', return_value=metadata_content), \
         patch('src.service.document_qa_service.DocumentQAService.save_metadata') as mock_save, \
         patch('os.remove') as mock_remove, \
         patch('pathlib.Path.exists', return_value=True), \
         patch('src.service.document_qa_service.datetime') as mock_datetime:

        mock_datetime.now.return_value = fake_now
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        # Act
        result = DocumentQAService.clean_old_indexes()

        # Assert
        assert result == expected_result
        mock_save.assert_called_once_with(expected_result)

        # Check that the correct files were removed
        assert mock_remove.call_count == len(expected_removed)
        for doc_id in expected_removed:
            mock_remove.assert_any_call(DocumentQAService.INDEX_DIR / f"{doc_id}.faiss")


def test_clean_old_indexes_file_not_found():
    """Test handling when index file doesn't exist."""
    metadata = {
        "doc1": {"last_access": (datetime.now() - timedelta(days=10)).isoformat()},
    }

    with patch('src.service.document_qa_service.DocumentQAService.load_metadata', return_value=metadata), \
            patch('src.service.document_qa_service.DocumentQAService.save_metadata') as mock_save, \
            patch('os.remove') as mock_remove, \
            patch('pathlib.Path.exists', return_value=False):
        # Act
        result = DocumentQAService.clean_old_indexes()

        # Assert
        assert result == {}
        mock_save.assert_called_once_with({})
        mock_remove.assert_not_called()  # File doesn't exist, so remove shouldn't be called


def test_clean_old_indexes_load_metadata_error():
    """Test handling when metadata loading fails."""
    with patch('src.service.document_qa_service.DocumentQAService.load_metadata',
               side_effect=Exception("Failed to load metadata")), \
            patch('src.service.document_qa_service.DocumentQAService.save_metadata') as mock_save:
        # Act & Assert
        with pytest.raises(Exception, match="Failed to load metadata"):
            DocumentQAService.clean_old_indexes()

        # Verify save_metadata was not called
        mock_save.assert_not_called()


def test_clean_old_indexes_remove_error():
    """Test handling when file removal fails."""
    metadata = {
        "doc1": {"last_access": (datetime.now() - timedelta(days=10)).isoformat()},
    }

    with patch('src.service.document_qa_service.DocumentQAService.load_metadata', return_value=metadata), \
            patch('src.service.document_qa_service.DocumentQAService.save_metadata') as mock_save, \
            patch('os.remove', side_effect=OSError("Permission denied")), \
            patch('pathlib.Path.exists', return_value=True):
        # Act & Assert
        with pytest.raises(OSError, match="Permission denied"):
            DocumentQAService.clean_old_indexes()

        # Verify save_metadata was not called due to the error
        mock_save.assert_not_called()


def test_clean_old_indexes_save_metadata_error():
    """Test handling when saving metadata fails."""
    metadata = {
        "doc1": {"last_access": (datetime.now() - timedelta(days=10)).isoformat()},
    }

    with patch('src.service.document_qa_service.DocumentQAService.load_metadata', return_value=metadata), \
            patch('src.service.document_qa_service.DocumentQAService.save_metadata',
                  side_effect=Exception("Failed to save metadata")), \
            patch('os.remove') as mock_remove, \
            patch('pathlib.Path.exists', return_value=True):
        # Act & Assert
        with pytest.raises(Exception, match="Failed to save metadata"):
            DocumentQAService.clean_old_indexes()

        # Verify remove was called before the save error
        mock_remove.assert_called_once()


@pytest.mark.parametrize(
    "threshold_days, metadata_input, expected_output",
    [
        # Test with custom threshold of 3 days
        (
            3,
            {
                "doc1": {"last_access": (datetime(2023, 1, 6)).isoformat()},  # 2 days old
                "doc2": {"last_access": (datetime(2023, 1, 4)).isoformat()},  # 4 days old
            },
            {
                "doc1": {"last_access": (datetime(2023, 1, 6)).isoformat()},
            },
        ),
        # Test with custom threshold of 14 days
        (
            14,
            {
                "doc1": {"last_access": (datetime(2022, 12, 29)).isoformat()},  # 10 days old
                "doc2": {"last_access": (datetime(2022, 12, 24)).isoformat()},  # 15 days old
            },
            {
                "doc1": {"last_access": (datetime(2022, 12, 29)).isoformat()},
            },
        ),
    ]
)
def test_clean_old_indexes_custom_threshold(threshold_days, metadata_input, expected_output):
    """Test cleaning old indexes with a custom threshold."""
    fixed_now = datetime(2023, 1, 8)

    with patch('src.service.document_qa_service.DocumentQAService.INACTIVITY_THRESHOLD_DAYS', threshold_days), \
         patch('src.service.document_qa_service.DocumentQAService.load_metadata', return_value=metadata_input), \
         patch('src.service.document_qa_service.DocumentQAService.save_metadata') as mock_save, \
         patch('os.remove'), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('src.service.document_qa_service.datetime') as mock_datetime:

        mock_datetime.now.return_value = fixed_now
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        result = DocumentQAService.clean_old_indexes()

        assert result == expected_output
        mock_save.assert_called_once_with(expected_output)