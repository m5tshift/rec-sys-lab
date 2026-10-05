import pandas as pd
import pytest

from src.evaluation.metrics import RecSysMetrics
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.cooccurrence import CooccurrenceRecommender
from src.models.pipeline import HybridFallbackPipeline
from src.models.popularity import PopularityRecommender


@pytest.fixture
def mock_data():
    items_df = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale", "Heat"],
            "clean_title": ["Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale", "Heat"],
            "genres": ["Adventure|Animation", "Adventure", "Comedy|Romance", "Comedy|Drama", "Action|Crime"],
            "genres_clean": ["Adventure Animation", "Adventure", "Comedy Romance", "Comedy Drama", "Action Crime"],
            "tags_clean": ["pixar animation", "board game", "sequel", "romance", "heist crime"],
            "content_corpus": [
                "Toy Story Adventure Animation pixar animation",
                "Jumanji Adventure board game",
                "Grumpier Old Men Comedy Romance sequel",
                "Waiting to Exhale Comedy Drama romance",
                "Heat Action Crime heist crime",
            ],
            "year": [1995, 1995, 1995, 1995, 1995],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            "movieId": [1, 2, 5, 1, 2, 3, 2, 3, 4, 1, 5],
            "rating": [5.0, 4.0, 4.5, 4.0, 5.0, 3.0, 3.5, 4.0, 4.5, 5.0, 4.0],
            "timestamp": list(range(100, 111)),
        }
    )

    return items_df, ratings_df


def test_popularity_recommender(mock_data):
    items_df, ratings_df = mock_data
    model = PopularityRecommender(m=2.0)
    model.fit(ratings_df, items_df)

    recs = model.recommend(user_id=1, top_k=2, exclude_seen=True)
    assert len(recs) <= 2
    # User 1 has seen 1, 2, 5. Recs should only contain 3 or 4
    for m_id, _ in recs:
        assert m_id in {3, 4}


def test_cooccurrence_recommender(mock_data):
    items_df, ratings_df = mock_data
    model = CooccurrenceRecommender(min_rating=3.0)
    model.fit(ratings_df, items_df)

    sims = model.similar_items(item_id=1, top_k=3)
    assert isinstance(sims, list)
    if sims:
        assert sims[0][0] != 1


def test_content_based_recommender(mock_data):
    items_df, ratings_df = mock_data
    model = ContentBasedRecommender()
    model.fit(ratings_df, items_df)

    recs = model.recommend(user_id=1, top_k=2, exclude_seen=True)
    assert isinstance(recs, list)
    sims = model.similar_items(item_id=1, top_k=2)
    assert len(sims) <= 2
    for m_id, score in sims:
        assert m_id != 1
        assert 0.0 <= score <= 1.0


def test_collaborative_recommender(mock_data):
    items_df, ratings_df = mock_data
    model = CollaborativeRecommender(n_factors=2)
    model.fit(ratings_df, items_df)

    # Known user
    recs = model.recommend(user_id=1, top_k=2, exclude_seen=True)
    assert isinstance(recs, list)

    # Cold user should return empty list from collaborative model directly
    cold_recs = model.recommend(user_id=999, top_k=2)
    assert cold_recs == []


def test_pipeline_cold_user_fallback(mock_data):
    items_df, ratings_df = mock_data
    pop = PopularityRecommender(m=2.0)
    cooc = CooccurrenceRecommender()
    cb = ContentBasedRecommender()
    cf = CollaborativeRecommender(n_factors=2)

    pipeline = HybridFallbackPipeline(cf, cb, cooc, pop)
    pipeline.fit(ratings_df, items_df)

    # Request collaborative for cold user -> should activate fallback
    resp = pipeline.recommend(user_id=999, model_type="collaborative", top_k=3)
    assert resp["is_fallback"] is True
    assert "fallback" in resp["strategy_used"]
    assert len(resp["recommendations"]) > 0


def test_metrics_precision_recall_ndcg():
    actual = {1, 3, 5}
    predicted = [1, 2, 3, 4, 6]

    # Hits in top 3: 1, 3 -> 2 hits
    p3 = RecSysMetrics.precision_at_k(actual, predicted, k=3)
    assert p3 == pytest.approx(2 / 3.0)

    # Recall in top 5: 1, 3 -> 2 out of 3 actual
    r5 = RecSysMetrics.recall_at_k(actual, predicted, k=5)
    assert r5 == pytest.approx(2 / 3.0)

    # NDCG
    ndcg = RecSysMetrics.ndcg_at_k(actual, predicted, k=5)
    assert 0.0 < ndcg <= 1.0

    # Coverage
    coverage = RecSysMetrics.catalog_coverage([[1, 2], [2, 3]], total_items_count=10)
    assert coverage == pytest.approx(3 / 10.0)
