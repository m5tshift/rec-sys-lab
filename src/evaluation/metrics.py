import math


class RecSysMetrics:
    """Standard evaluation metrics for recommendation systems."""

    @staticmethod
    def precision_at_k(actual: set[int], predicted: list[int], k: int) -> float:
        """Precision@K: Proportion of recommended items in top-k that are relevant."""
        if k <= 0 or not predicted:
            return 0.0
        top_k = predicted[:k]
        hits = len(set(top_k) & actual)
        return hits / float(k)

    @staticmethod
    def recall_at_k(actual: set[int], predicted: list[int], k: int) -> float:
        """Recall@K: Proportion of relevant items captured in top-k recommendations."""
        if not actual or not predicted or k <= 0:
            return 0.0
        top_k = predicted[:k]
        hits = len(set(top_k) & actual)
        return hits / float(len(actual))

    @staticmethod
    def average_precision_at_k(actual: set[int], predicted: list[int], k: int) -> float:
        """Average Precision@K (AP@K) for a single user."""
        if not actual or not predicted or k <= 0:
            return 0.0

        top_k = predicted[:k]
        score = 0.0
        hits = 0

        for i, item in enumerate(top_k):
            if item in actual:
                hits += 1
                score += hits / (i + 1.0)

        return score / min(len(actual), k)

    @staticmethod
    def ndcg_at_k(actual: set[int], predicted: list[int], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K (binary relevance)."""
        if not actual or not predicted or k <= 0:
            return 0.0

        top_k = predicted[:k]
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in actual:
                dcg += 1.0 / math.log2(i + 2.0)

        # Ideal DCG: all actual items at top positions
        idcg = sum(1.0 / math.log2(i + 2.0) for i in range(min(len(actual), k)))
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def catalog_coverage(recommended_item_sets: list[list[int]], total_items_count: int) -> float:
        """Catalog Coverage: Fraction of catalog recommended to at least one user."""
        if total_items_count == 0:
            return 0.0
        unique_recommended = set()
        for item_list in recommended_item_sets:
            unique_recommended.update(item_list)
        return len(unique_recommended) / float(total_items_count)

    @staticmethod
    def novelty(
        recommended_item_sets: list[list[int]],
        item_frequencies: dict[int, float],
        total_interactions: int,
    ) -> float:
        """Novelty: Average self-information of recommended items across all recommendations.

        Novelty(item) = -log2(P(item))
        Higher values mean less popular, more surprising recommendations.
        """
        if not recommended_item_sets or total_interactions == 0:
            return 0.0

        total_novelty = 0.0
        count = 0

        for recs in recommended_item_sets:
            for item in recs:
                freq = item_frequencies.get(item, 1)
                p_item = freq / float(total_interactions)
                self_info = -math.log2(max(p_item, 1e-9))
                total_novelty += self_info
                count += 1

        return total_novelty / max(count, 1)
