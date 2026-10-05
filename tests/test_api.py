from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.cooccurrence import CooccurrenceRecommender
from src.models.pipeline import HybridFallbackPipeline
from src.models.popularity import PopularityRecommender


@pytest.fixture(scope="module")
def mock_pipeline_instance():
    items_df = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "title": ["A", "B", "C", "D", "E", "F"],
        "clean_title": ["A", "B", "C", "D", "E", "F"],
        "genres": ["Action", "Action", "Comedy", "Drama", "Drama", "Action"],
        "genres_clean": ["Action", "Action", "Comedy", "Drama", "Drama", "Action"],
        "tags_clean": ["", "", "", "", "", ""],
        "content_corpus": ["A Action", "B Action", "C Comedy", "D Drama", "E Drama", "F Action"],
        "year": [2000, 2001, 2002, 2003, 2004, 2005]
    })
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [1, 2, 1],
        "rating": [5.0, 4.0, 3.0],
        "timestamp": [1, 2, 3]
    })
    pop = PopularityRecommender()
    cooc = CooccurrenceRecommender()
    cb = ContentBasedRecommender()
    cf = CollaborativeRecommender(n_factors=2)
    pipeline = HybridFallbackPipeline(cf, cb, cooc, pop)
    pipeline.fit(ratings_df, items_df)
    return pipeline

@pytest.fixture(scope="module", autouse=True)
def mock_load(mock_pipeline_instance):
    with patch("src.api.app.load_pipeline_artifacts", return_value=mock_pipeline_instance):
        yield

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_ui_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "RecSys demo" in response.text


def test_recommend_user_collaborative(client):
    response = client.get("/recommend/user/1?model_type=collaborative&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["count"] <= 10
    assert len(data["recommendations"]) == data["count"]
    assert "item_id" in data["recommendations"][0]
    assert "score" in data["recommendations"][0]


def test_recommend_user_cold_fallback(client):
    # User 999999 is nonexistent -> must trigger fallback to popularity
    response = client.get("/recommend/user/999999?model_type=collaborative&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is True
    assert "fallback" in data["strategy_used"]
    assert len(data["recommendations"]) == data["count"]


def test_recommend_similar_items(client):
    # Movie 1 is Toy Story
    response = client.get("/recommend/item/1?model_type=content_based&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == 1
    assert data["count"] <= 10
    assert len(data["similar_items"]) == data["count"]


def test_recommend_popular_endpoint(client):
    response = client.get("/recommend/popular?top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] <= 10
    assert data["strategy_used"] == "popularity"


def test_prometheus_metrics_endpoint(client):
    # Make a request to generate metrics
    client.get("/recommend/user/1?model_type=collaborative&top_k=3")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "recsys_requests_total" in response.text
    assert "recsys_request_latency_seconds" in response.text
