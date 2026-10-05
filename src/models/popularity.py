
import pandas as pd

from src.config import BAYESIAN_PRIOR_CONFIDENCE
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender using Bayesian Average / IMDb Weighted Rating.

    Formula:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        R = average rating of the item
        v = number of ratings for the item
        m = minimum votes required to be considered (prior weight)
        C = the mean rating across the whole catalog
    """

    def __init__(self, m: float = BAYESIAN_PRIOR_CONFIDENCE):
        super().__init__(name="PopularityRecommender")
        self.m = m
        self.global_mean: float = 0.0
        self.ranking_table: pd.DataFrame = pd.DataFrame()

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "PopularityRecommender":
        self.items_df = items_df.copy()
        self._record_user_history(train_df)

        self.global_mean = float(train_df["rating"].mean())

        # Aggregate counts and mean ratings per item
        stats = (
            train_df.groupby("movieId")
            .agg(vote_count=("rating", "count"), vote_average=("rating", "mean"))
            .reset_index()
        )

        merged = items_df.merge(stats, on="movieId", how="left")
        merged["vote_count"] = merged["vote_count"].fillna(0)
        merged["vote_average"] = merged["vote_average"].fillna(self.global_mean)

        v = merged["vote_count"]
        r = merged["vote_average"]
        c = self.global_mean
        m = self.m

        merged["score"] = (v / (v + m)) * r + (m / (v + m)) * c
        self.ranking_table = merged.sort_values(by="score", ascending=False)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        filter_genre: str | None = None,
    ) -> list[tuple[int, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        candidates = self.ranking_table

        if filter_genre:
            candidates = candidates[
                candidates["genres_clean"].str.contains(filter_genre, case=False, na=False)
            ]

        if exclude_seen and user_id in self.user_history:
            seen_set = set(self.user_history[user_id])
            candidates = candidates[~candidates["movieId"].isin(seen_set)]

        top_candidates = candidates.head(top_k)
        return list(zip(top_candidates["movieId"].astype(int), top_candidates["score"].astype(float)))

    def similar_items(self, item_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Fallback for item-to-item: return top popular in the same primary genre."""
        if not self.is_fitted or self.items_df is None:
            raise RuntimeError("Model is not fitted yet.")

        target = self.items_df[self.items_df["movieId"] == item_id]
        if target.empty:
            return self.recommend(user_id=-1, top_k=top_k, exclude_seen=False)

        genres = str(target.iloc[0]["genres_clean"]).split()
        primary_genre = genres[0] if genres else None
        candidates = self.ranking_table[self.ranking_table["movieId"] != item_id]

        if primary_genre:
            candidates = candidates[
                candidates["genres_clean"].str.contains(primary_genre, case=False, na=False)
            ]

        top_candidates = candidates.head(top_k)
        return list(zip(top_candidates["movieId"].astype(int), top_candidates["score"].astype(float)))
