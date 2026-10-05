from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.items_df: pd.DataFrame | None = None
        self.user_history: dict[int, list[int]] = {}

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "BaseRecommender":
        """Fit model on training interactions and item metadata."""

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        filter_genre: str | None = None,
    ) -> list[tuple[int, float]]:
        """Recommend top_k items for given user_id.

        Returns list of tuples: (item_id, predicted_score)
        """

    def similar_items(
        self, item_id: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Find items similar to given item_id."""
        raise NotImplementedError(f"{self.name} does not support item-to-item similarity")

    def _record_user_history(self, train_df: pd.DataFrame) -> None:
        """Cache user interaction history for excluding already seen items."""
        history = train_df.groupby("userId")["movieId"].apply(list).to_dict()
        self.user_history = history
