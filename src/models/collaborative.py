
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SVD_N_COMPONENTS
from src.models.base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """Collaborative Filtering Recommender using Matrix Factorization via SVD.

    Decomposes the demeaned user-item rating matrix:
        R_demeaned ~ U * Sigma * V^T
    Item similarities are computed in the learned latent factor space V.
    """

    def __init__(self, n_factors: int = SVD_N_COMPONENTS):
        super().__init__(name="CollaborativeRecommender")
        self.n_factors = n_factors
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

        self.user_means: np.ndarray = np.array([])
        self.global_mean: float = 0.0
        self.u_factors: np.ndarray | None = None
        self.vt_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None  # V matrix for item similarities

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "CollaborativeRecommender":
        self.items_df = items_df.copy()
        self._record_user_history(train_df)

        self.global_mean = float(train_df["rating"].mean())

        # Construct indices
        user_ids = sorted(train_df["userId"].unique())
        item_ids = sorted(items_df["movieId"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.idx_to_user = {i: u for i, u in enumerate(user_ids)}
        self.item_to_idx = {m: i for i, m in enumerate(item_ids)}
        self.idx_to_item = {i: m for i, m in enumerate(item_ids)}

        n_users = len(user_ids)
        n_items = len(item_ids)

        # Build sparse matrix
        valid_ratings = train_df[train_df["movieId"].isin(self.item_to_idx)]
        rows = [self.user_to_idx[u] for u in valid_ratings["userId"]]
        cols = [self.item_to_idx[m] for m in valid_ratings["movieId"]]
        vals = valid_ratings["rating"].values.astype(np.float32)

        ratings_sparse = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # Compute user means for demeaned matrix
        user_sums = np.array(ratings_sparse.sum(axis=1)).flatten()
        user_counts = np.diff(ratings_sparse.indptr)
        self.user_means = np.zeros(n_users, dtype=np.float32)
        nonzero_mask = user_counts > 0
        self.user_means[nonzero_mask] = user_sums[nonzero_mask] / user_counts[nonzero_mask]
        self.user_means[~nonzero_mask] = self.global_mean

        # Construct demeaned sparse matrix: only center existing entries
        demeaned_data = vals - self.user_means[rows]
        demeaned_matrix = csr_matrix((demeaned_data, (rows, cols)), shape=(n_users, n_items))

        # Perform SVD
        k = min(self.n_factors, min(n_users, n_items) - 1)
        u, s, vt = svds(demeaned_matrix, k=k)

        # Sort singular values descending
        sort_order = np.argsort(s)[::-1]
        s = s[sort_order]
        u = u[:, sort_order]
        vt = vt[sort_order, :]

        # Scale latent representations
        self.u_factors = u
        self.vt_factors = np.diag(s) @ vt
        self.item_factors = vt.T  # Shape: (n_items, k)

        self.is_fitted = True
        return self

    def predict_user_ratings(self, user_idx: int) -> np.ndarray:
        """Compute predicted ratings vector for user across all items."""
        u_vec = self.u_factors[user_idx, :].reshape(1, -1)
        # Reconstruct: user_mean + u_vec * (Sigma * Vt)
        pred = self.user_means[user_idx] + (u_vec @ self.vt_factors).flatten()
        return pred

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        filter_genre: str | None = None,
    ) -> list[tuple[int, float]]:
        if not self.is_fitted or self.u_factors is None:
            raise RuntimeError("Model is not fitted yet.")

        if user_id not in self.user_to_idx:
            # Cold user
            return []

        u_idx = self.user_to_idx[user_id]
        scores = self.predict_user_ratings(u_idx)

        seen_set = set(self.user_history.get(user_id, [])) if exclude_seen else set()

        sorted_indices = np.argsort(scores)[::-1]
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

            results.append((m_id, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

    def similar_items(self, item_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.is_fitted or self.item_factors is None:
            raise RuntimeError("Model is not fitted yet.")

        if item_id not in self.item_to_idx:
            return []

        item_idx = self.item_to_idx[item_id]
        target_factor = self.item_factors[item_idx, :].reshape(1, -1)

        sims = cosine_similarity(target_factor, self.item_factors).flatten()
        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            if idx == item_idx:
                continue
            results.append((self.idx_to_item[idx], float(sims[idx])))
            if len(results) >= top_k:
                break

        return results
