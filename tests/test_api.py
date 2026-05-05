"""
Integration tests for the FastAPI recommendation endpoint.

These tests mock the Recommender so they run without artifacts on disk.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    mock_rec = MagicMock()
    mock_rec.recommend.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mock_rec.model.user_embedding.num_embeddings = 943

    with patch("api.app.recommender", mock_rec):
        from api.app import app
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_recommend_valid_user(client):
    resp = client.post("/recommend", json={"user_id": 0, "top_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 0
    assert len(body["recommendations"]) == 10


def test_recommend_invalid_user(client):
    resp = client.post("/recommend", json={"user_id": 99999, "top_k": 10})
    assert resp.status_code == 400


def test_recommend_top_k_respected(client):
    from unittest.mock import MagicMock, patch

    mock_rec = MagicMock()
    mock_rec.recommend.return_value = [1, 2, 3]
    mock_rec.model.user_embedding.num_embeddings = 943

    with patch("api.app.recommender", mock_rec):
        from api.app import app
        from fastapi.testclient import TestClient

        c = TestClient(app)
        resp = c.post("/recommend", json={"user_id": 0, "top_k": 3})
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) == 3
