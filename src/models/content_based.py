
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using TF-IDF over movie titles, genres, and user tags.

    User profiles are constructed as rating-weighted combinations of item TF-IDF vectors.
    Recommendations are scored via cosine similarity between the user profile and item vectors.
    """

    def __init__(self, max_features: int = 5000):
        super().__init__(name="ContentBasedRecommender")
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            sublinear_tf=True,
        )
        self.tfidf_matrix: np.ndarray | None = None
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.user_profiles: dict[int, np.ndarray] = {}

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "ContentBasedRecommender":
        self.items_df = items_df.copy()
        self._record_user_history(train_df)

        corpus = self.items_df["content_corpus"].fillna("")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        movie_ids = self.items_df["movieId"].tolist()
        self.item_to_idx = {m: i for i, m in enumerate(movie_ids)}
        self.idx_to_item = {i: m for i, m in enumerate(movie_ids)}

        # Precompute user profiles
        user_means = train_df.groupby("userId")["rating"].mean().to_dict()

        for u_id, group in train_df.groupby("userId"):
            u_mean = user_means.get(u_id, 3.5)
            rated_movie_indices = []
            weights = []

            for _, row in group.iterrows():
                m_id = int(row["movieId"])
                if m_id in self.item_to_idx:
                    idx = self.item_to_idx[m_id]
                    # Center rating: positive above mean, negative below mean
                    weight = float(row["rating"]) - u_mean
                    rated_movie_indices.append(idx)
                    weights.append(weight)

            if not rated_movie_indices:
                continue

            sub_matrix = self.tfidf_matrix[rated_movie_indices]
            weights_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)

            # Weighted sum of item vectors
            weighted_profile = sub_matrix.multiply(weights_arr).sum(axis=0)
            norm = np.linalg.norm(weighted_profile)
            if norm > 0:
                weighted_profile = weighted_profile / norm

            self.user_profiles[u_id] = np.asarray(weighted_profile).flatten()

        self.is_fitted = True
        return self

    def similar_items(self, item_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Model is not fitted yet.")

        if item_id not in self.item_to_idx:
            return []

        idx = self.item_to_idx[item_id]
        target_vec = self.tfidf_matrix[idx]
        sim_scores = linear_kernel(target_vec, self.tfidf_matrix).flatten()

        # Sort descending and exclude self
        top_indices = np.argsort(sim_scores)[::-1]
        results = []
        for i in top_indices:
            if i == idx:
                continue
            results.append((self.idx_to_item[i], float(sim_scores[i])))
            if len(results) >= top_k:
                break
        return results

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        filter_genre: str | None = None,
    ) -> list[tuple[int, float]]:
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Model is not fitted yet.")

        if user_id not in self.user_profiles:
            return []

        user_vec = self.user_profiles[user_id].reshape(1, -1)
        sim_scores = linear_kernel(user_vec, self.tfidf_matrix).flatten()

        seen_set = set(self.user_history.get(user_id, [])) if exclude_seen else set()

        sorted_indices = np.argsort(sim_scores)[::-1]
        results: list[tuple[int, float]] = []

        for idx in sorted_indices:
            m_id = self.idx_to_item[idx]
            if m_id in seen_set:
                continue

            if filter_genre and self.items_df is not None:
                match = self.items_df[self.items_df["movieId"] == m_id]
                if not match.empty:
                    movie_genre = str(match.iloc[0]["genres_clean"])
                    if filter_genre.lower() not in movie_genre.lower():
                        continue

            results.append((m_id, float(sim_scores[idx])))
            if len(results) >= top_k:
                break

        return results
