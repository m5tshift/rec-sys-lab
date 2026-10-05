import time
from typing import Any

import pandas as pd

from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.cooccurrence import CooccurrenceRecommender
from src.models.popularity import PopularityRecommender


class HybridFallbackPipeline:
    """Production-grade pipeline executing personalized models with hierarchical fallback.

    Hierarchy:
        1. Primary Model (Collaborative Filtering or Content-Based)
        2. Co-occurrence fallback (if user has sparse positive history)
        3. Genre Popularity fallback (if user has preference or filter)
        4. Global Bayesian Popularity fallback (guarantees SLA and prevents zero-result screens)
    """

    def __init__(
        self,
        collaborative: CollaborativeRecommender,
        content_based: ContentBasedRecommender,
        cooccurrence: CooccurrenceRecommender,
        popularity: PopularityRecommender,
    ):
        self.collaborative = collaborative
        self.content_based = content_based
        self.cooccurrence = cooccurrence
        self.popularity = popularity
        self.items_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "HybridFallbackPipeline":
        self.items_df = items_df.copy()
        self.collaborative.fit(train_df, items_df)
        self.content_based.fit(train_df, items_df)
        self.cooccurrence.fit(train_df, items_df)
        self.popularity.fit(train_df, items_df)
        return self

    def recommend(
        self,
        user_id: int,
        model_type: str = "collaborative",
        top_k: int = 10,
        exclude_seen: bool = True,
        filter_genre: str | None = None,
    ) -> dict[str, Any]:
        """Produce recommendations with guaranteed fallback response."""
        start_time = time.perf_counter()
        used_strategy = model_type
        is_fallback = False
        recs: list[tuple[int, float]] = []

        # 1. Attempt primary model
        if model_type == "collaborative":
            recs = self.collaborative.recommend(
                user_id=user_id, top_k=top_k, exclude_seen=exclude_seen, filter_genre=filter_genre
            )
        elif model_type == "content_based":
            recs = self.content_based.recommend(
                user_id=user_id, top_k=top_k, exclude_seen=exclude_seen, filter_genre=filter_genre
            )
        elif model_type == "cooccurrence":
            recs = self.cooccurrence.recommend(
                user_id=user_id, top_k=top_k, exclude_seen=exclude_seen, filter_genre=filter_genre
            )
        elif model_type == "popularity":
            recs = self.popularity.recommend(
                user_id=user_id, top_k=top_k, exclude_seen=exclude_seen, filter_genre=filter_genre
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 2. Check if primary model failed or returned insufficient recommendations
        if len(recs) < top_k and model_type != "popularity":
            is_fallback = True
            needed = top_k - len(recs)
            existing_ids = {m_id for m_id, _ in recs}

            # Try cooccurrence if user had any ratings
            if model_type != "cooccurrence":
                cooc_recs = self.cooccurrence.recommend(
                    user_id=user_id, top_k=top_k, exclude_seen=exclude_seen, filter_genre=filter_genre
                )
                for m_id, score in cooc_recs:
                    if m_id not in existing_ids:
                        recs.append((m_id, score))
                        existing_ids.add(m_id)
                        if len(recs) >= top_k:
                            break

            # If still insufficient, fallback to Bayesian Popularity
            if len(recs) < top_k:
                needed = top_k - len(recs)
                pop_recs = self.popularity.recommend(
                    user_id=user_id,
                    top_k=needed * 2,
                    exclude_seen=exclude_seen,
                    filter_genre=filter_genre,
                )
                for m_id, score in pop_recs:
                    if m_id not in existing_ids:
                        recs.append((m_id, score))
                        existing_ids.add(m_id)
                        if len(recs) >= top_k:
                            break
                used_strategy = f"{model_type}_fallback_popularity"
            else:
                used_strategy = f"{model_type}_fallback_cooccurrence"

        recs = recs[:top_k]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Enrich item titles and genres if metadata available
        enriched_items = []
        if self.items_df is not None:
            items_lookup = self.items_df.set_index("movieId").to_dict(orient="index")
            for m_id, score in recs:
                meta = items_lookup.get(m_id, {})
                enriched_items.append(
                    {
                        "item_id": int(m_id),
                        "score": round(float(score), 4),
                        "title": meta.get("title", f"Movie #{m_id}"),
                        "genres": meta.get("genres", ""),
                        "year": meta.get("year", 0),
                    }
                )

        return {
            "user_id": user_id,
            "model_requested": model_type,
            "strategy_used": used_strategy,
            "is_fallback": is_fallback,
            "count": len(enriched_items),
            "recommendations": enriched_items,
            "latency_ms": round(latency_ms, 2),
        }
