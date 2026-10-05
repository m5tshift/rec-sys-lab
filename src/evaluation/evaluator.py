import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import TOP_K_DEFAULT
from src.evaluation.metrics import RecSysMetrics
from src.models.base import BaseRecommender


class RecSysEvaluator:
    """Evaluates recommender models on held-out test data."""

    def __init__(self, k: int = TOP_K_DEFAULT, min_relevant_rating: float = 3.5):
        self.k = k
        self.min_relevant_rating = min_relevant_rating

    def evaluate_model(
        self,
        model: BaseRecommender,
        test_df: pd.DataFrame,
        total_items_count: int,
        item_frequencies: dict[int, float],
        total_train_interactions: int,
        limit_users: int = 0,
    ) -> dict[str, float]:
        """Run full evaluation suite on a given model."""
        # Ground truth: positive test ratings per user
        test_pos = test_df[test_df["rating"] >= self.min_relevant_rating]
        user_test_ground_truth: dict[int, set[int]] = (
            test_pos.groupby("userId")["movieId"].apply(set).to_dict()
        )

        test_users = list(user_test_ground_truth.keys())
        if limit_users > 0:
            test_users = test_users[:limit_users]

        precisions: list[float] = []
        recalls: list[float] = []
        aps: list[float] = []
        ndcgs: list[float] = []
        all_recs: list[list[int]] = []

        for u_id in test_users:
            actual = user_test_ground_truth[u_id]
            if not actual:
                continue

            predictions = model.recommend(user_id=u_id, top_k=self.k, exclude_seen=True)
            pred_item_ids = [m_id for m_id, _ in predictions]

            all_recs.append(pred_item_ids)
            precisions.append(RecSysMetrics.precision_at_k(actual, pred_item_ids, self.k))
            recalls.append(RecSysMetrics.recall_at_k(actual, pred_item_ids, self.k))
            aps.append(RecSysMetrics.average_precision_at_k(actual, pred_item_ids, self.k))
            ndcgs.append(RecSysMetrics.ndcg_at_k(actual, pred_item_ids, self.k))

        coverage = RecSysMetrics.catalog_coverage(all_recs, total_items_count)
        nov = RecSysMetrics.novelty(all_recs, item_frequencies, total_train_interactions)

        return {
            "model": model.name,
            f"precision@{self.k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
            f"recall@{self.k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
            f"map@{self.k}": round(float(np.mean(aps)) if aps else 0.0, 4),
            f"ndcg@{self.k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
            "catalog_coverage": round(coverage, 4),
            "novelty_bits": round(nov, 2),
            "evaluated_users": len(test_users),
        }

    @staticmethod
    def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
        """Save results to JSON and CSV formats."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_file = output_path.with_suffix(".json")
        csv_file = output_path.with_suffix(".csv")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {json_file} and {csv_file}")
