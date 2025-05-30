import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from main import app
from src.service.auth_service import authenticate_user


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_list_documents_success(client):
    mock_user_id = "user-123"
    mock_documents = [
        {"id": "12345678-1234-5678-1234-567812345678", "title": "Document 1"},
        {"id": "87654321-4321-8765-4321-876543210987", "title": "Document 2"}
    ]

    app.dependency_overrides[authenticate_user] = lambda: mock_user_id

    with patch(
            "src.service.document_service.DocumentService.get_documents_by_user",
            new_callable=AsyncMock
    ) as mock_get_docs:
        mock_get_docs.return_value = mock_documents

        response = client.get("/api/v1/list-documents", headers={"Authorization": "Bearer faketoken"})

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "status": 1,
            "message": "User documents fetched successfully",
            "document_data": mock_documents,
            "code": 200
        }

        mock_get_docs.assert_called_once_with(user_id=mock_user_id)

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    """Test when user has no documents."""

    mock_user_id = "user-123"
    mock_documents = []

    app.dependency_overrides[authenticate_user] = lambda: mock_user_id

    with patch(
            "src.service.document_service.DocumentService.get_documents_by_user",
            new_callable=AsyncMock
    ) as mock_get_docs:
        mock_get_docs.return_value = mock_documents

        response = client.get("/api/v1/list-documents", headers={"Authorization": "Bearer faketoken"})

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "status": 1,
            "message": "User documents fetched successfully",
            "document_data": [],
            "code": 200
        }

        mock_get_docs.assert_called_once_with(user_id=mock_user_id)

    app.dependency_overrides = {}
