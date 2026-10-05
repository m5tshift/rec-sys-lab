
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender


class CooccurrenceRecommender(BaseRecommender):
    """Heuristic recommender based on item co-occurrence (People also watched/bought).

    Builds an item-to-item co-occurrence matrix from users' positive interactions
    and calculates normalized cosine similarity between items.
    """

    def __init__(self, min_rating: float = 3.0, top_n_history: int = 5):
        super().__init__(name="CooccurrenceRecommender")
        self.min_rating = min_rating
        self.top_n_history = top_n_history
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.sim_matrix: csr_matrix | None = None
        self.user_positive_history: dict[int, list[tuple[int, float]]] = {}

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "CooccurrenceRecommender":
        self.items_df = items_df.copy()
        self._record_user_history(train_df)

        # Filter positive interactions
        pos_df = train_df[train_df["rating"] >= self.min_rating]

        # Unique mappings
        all_items = sorted(items_df["movieId"].unique())
        self.item_to_idx = {m: i for i, m in enumerate(all_items)}
        self.idx_to_item = {i: m for i, m in enumerate(all_items)}

        user_ids = sorted(pos_df["userId"].unique())
        user_to_idx = {u: i for i, u in enumerate(user_ids)}

        # Record user positive interactions
        for u_id, group in pos_df.groupby("userId"):
            sorted_history = (
                group.sort_values(by="rating", ascending=False)[["movieId", "rating"]]
                .values.tolist()
            )
            self.user_positive_history[u_id] = [(int(m), float(r)) for m, r in sorted_history]

        # Build binary user-item matrix of positive interactions
        valid_pos = pos_df[pos_df["movieId"].isin(self.item_to_idx)]
        rows = [user_to_idx[u] for u in valid_pos["userId"]]
        cols = [self.item_to_idx[m] for m in valid_pos["movieId"]]
        data = np.ones(len(rows), dtype=np.float32)

        n_users = len(user_to_idx)
        n_items = len(self.item_to_idx)
        ui_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        # Item-item co-occurrence: C = X^T * X
        cooc = ui_matrix.T.dot(ui_matrix)

        # Normalize with cosine similarity: S_ij = C_ij / sqrt(C_ii * C_jj)
        item_counts = np.array(cooc.diagonal()).flatten()
        sqrt_counts = np.sqrt(item_counts)
        inv_sqrt = np.zeros_like(sqrt_counts)
        mask = sqrt_counts > 0
        inv_sqrt[mask] = 1.0 / sqrt_counts[mask]

        from scipy.sparse import diags
        d_mat = diags(inv_sqrt)
        self.sim_matrix = d_mat.dot(cooc).dot(d_mat).tocsr()

        # Zero out diagonal
        self.sim_matrix.setdiag(0.0)
        self.sim_matrix.eliminate_zeros()

        self.is_fitted = True
        return self

    def similar_items(self, item_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.is_fitted or self.sim_matrix is None:
            raise RuntimeError("Model is not fitted yet.")

        if item_id not in self.item_to_idx:
            return []

        item_idx = self.item_to_idx[item_id]
        row = self.sim_matrix.getrow(item_idx)
        if row.nnz == 0:
            return []

        col_indices = row.indices
        scores = row.data

        top_indices = np.argsort(scores)[::-1][:top_k]
        result = [
            (self.idx_to_item[col_indices[idx]], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        return result

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        filter_genre: str | None = None,
    ) -> list[tuple[int, float]]:
        if not self.is_fitted or self.sim_matrix is None:
            raise RuntimeError("Model is not fitted yet.")

        if user_id not in self.user_positive_history:
            return []

        # Use recent top-rated items from user's history
        recent_items = self.user_positive_history[user_id][: self.top_n_history]
        aggregated_scores: dict[int, float] = {}

        seen_set = set(self.user_history.get(user_id, [])) if exclude_seen else set()

        for m_id, weight in recent_items:
            sims = self.similar_items(m_id, top_k=top_k * 3)
            for sim_id, sim_score in sims:
                if sim_id in seen_set:
                    continue
                aggregated_scores[sim_id] = aggregated_scores.get(sim_id, 0.0) + (sim_score * weight)

        if not aggregated_scores:
            return []

        candidates = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

        if filter_genre and self.items_df is not None:
            genre_filtered = []
            for m_id, score in candidates:
                match = self.items_df[self.items_df["movieId"] == m_id]
                if not match.empty:
                    genres = str(match.iloc[0]["genres_clean"])
                    if filter_genre.lower() in genres.lower():
                        genre_filtered.append((m_id, score))
            candidates = genre_filtered

        return candidates[:top_k]
