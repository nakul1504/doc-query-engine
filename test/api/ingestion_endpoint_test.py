from http import HTTPStatus

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import io

from starlette import status

from main import app
from src.service.auth_service import authenticate_user

client = TestClient(app)


@pytest.mark.asyncio
async def test_ingest_document_success():
    """Test successful ingestion of a document."""

    mock_user_id = "user-123"
    mock_document_id = {"document_id": "12345678-1234-5678-1234-567812345678"}
    mock_response = mock_document_id

    app.dependency_overrides[authenticate_user] = lambda: mock_user_id

    # Create a fake .txt file
    file_data = io.BytesIO(b"This is a test document.")
    files = {"file": ("test.txt", file_data, "text/plain")}

    with patch(
            "src.service.ingestion_service.IngestionService.process_document",
            new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = mock_document_id

        # Make the request
        response = client.post("/api/v1/ingest", files=files)

        # Validate response
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "status": 1,
            "message": "Document ingested successfully",
            "document_id": mock_document_id.get("document_id"),
            "code": status.HTTP_201_CREATED
        }

        mock_process.assert_called_once()
        uploaded_file = mock_process.call_args.kwargs["file"]
        assert uploaded_file.filename == "test.txt"
        assert uploaded_file.content_type == "text/plain"
        assert mock_process.call_args.kwargs["user_id"] == mock_user_id

    # Clean up overrides after test
    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_ingest_document_pdf_success():
    """Test successful ingestion of a PDF document."""

    mock_document_id = {"document_id": "12345678-1234-5678-1234-567812345678"}
    mock_user_id = "user-123"
    app.dependency_overrides[authenticate_user] = lambda: mock_user_id

    file_data = io.BytesIO(b"This is a test document.")
    file_name = "test.pdf"
    content_type = "application/pdf"
    files = {"file": (file_name, file_data, content_type)}

    with patch(
            "src.service.ingestion_service.IngestionService.process_document",
            new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = mock_document_id

        response = client.post("/api/v1/ingest", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "status": 1,
            "message": "Document ingested successfully",
            "document_id": mock_document_id.get("document_id"),
            "code": status.HTTP_201_CREATED
        }

        mock_process.assert_called_once()
        file_arg = mock_process.call_args.kwargs["file"]
        assert file_arg.filename == file_name
        assert file_arg.content_type == content_type
        assert mock_process.call_args.kwargs["user_id"] == mock_user_id

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_ingest_document_no_file():
    """Test the endpoint behavior when no file is provided."""

    mock_user_id = "user-123"
    app.dependency_overrides[authenticate_user] = lambda: mock_user_id
    response = client.post("/api/v1/ingest")
    assert response.status_code == 422
    app.dependency_overrides = {}
