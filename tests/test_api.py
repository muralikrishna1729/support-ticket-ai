import sys
import os
from unittest.mock import patch, MagicMock
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from fastapi.testclient import TestClient
from src.app import app 

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_home_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.fixture(autouse=True)
def mock_pipeline():
    with patch("src.services.ticket_service.pipeline", MagicMock()) as mock:
        mock.predict.return_value = {
            "category": "Technical Support",
            "issue_type": "Incident",
            "auto_response": "Test response",
            "clean_text": "test",
            "confidence": 0.9,
            "needs_review": False,
        }
        yield mock