import time

import joblib
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.evaluator import RecSysEvaluator
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.cooccurrence import CooccurrenceRecommender
from src.models.pipeline import HybridFallbackPipeline
from src.models.popularity import PopularityRecommender


def run_training_and_evaluation() -> None:
    print("=" * 60)
    print("Starting RecSys Training & Offline Evaluation Pipeline")
    print("=" * 60)

    # 1. Load Data
    t0 = time.time()
    loader = DataLoader()
    movies_df, ratings_df, tags_df = loader.load_raw_data()
    print(f"Loaded: {len(movies_df)} movies, {len(ratings_df)} ratings, {len(tags_df)} tags.")

    # 2. Preprocess
    preprocessor = DataPreprocessor()
    items_df = preprocessor.prepare_item_metadata(movies_df, tags_df)
    train_df, test_df = preprocessor.split_train_test(ratings_df)
    print(f"Data split: Train={len(train_df)} rows, Test={len(test_df)} rows.")

    # Save processed data for API quick loading
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    items_df.to_pickle(processed_dir / "items.pkl")
    train_df.to_pickle(processed_dir / "train.pkl")
    test_df.to_pickle(processed_dir / "test.pkl")

    # 3. Model initialization
    print("\nTraining models...")
    popularity_model = PopularityRecommender()
    cooccurrence_model = CooccurrenceRecommender()
    content_model = ContentBasedRecommender()
    collaborative_model = CollaborativeRecommender()

    # Fit all models
    t_start = time.time()
    popularity_model.fit(train_df, items_df)
    print(f"Fitted PopularityRecommender in {time.time() - t_start:.2f}s")

    t_start = time.time()
    cooccurrence_model.fit(train_df, items_df)
    print(f"Fitted CooccurrenceRecommender in {time.time() - t_start:.2f}s")

    t_start = time.time()
    content_model.fit(train_df, items_df)
    print(f"Fitted ContentBasedRecommender in {time.time() - t_start:.2f}s")

    t_start = time.time()
    collaborative_model.fit(train_df, items_df)
    print(f"Fitted CollaborativeRecommender in {time.time() - t_start:.2f}s")

    # Fit Hybrid Pipeline
    pipeline = HybridFallbackPipeline(
        collaborative=collaborative_model,
        content_based=content_model,
        cooccurrence=cooccurrence_model,
        popularity=popularity_model,
    )
    pipeline.items_df = items_df

    # Save fitted models to disk
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "pipeline.pkl")
    print(f"Saved fitted pipeline to {MODELS_DIR / 'pipeline.pkl'}")

    # 4. Evaluation
    print("\nRunning offline evaluation...")
    evaluator = RecSysEvaluator(k=10)
    total_items = len(items_df)
    item_frequencies = train_df["movieId"].value_counts().to_dict()
    total_train_interactions = len(train_df)

    models_to_evaluate = [
        popularity_model,
        cooccurrence_model,
        content_model,
        collaborative_model,
    ]

    results = []
    for model in models_to_evaluate:
        print(f"Evaluating {model.name}...")
        metrics = evaluator.evaluate_model(
            model=model,
            test_df=test_df,
            total_items_count=total_items,
            item_frequencies=item_frequencies,
            total_train_interactions=total_train_interactions,
        )
        results.append(metrics)

    # Save results
    results_dir = PROJECT_ROOT / "artifacts"
    evaluator.save_results(results, results_dir / "evaluation_results")

    # Display comparison table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("OFFLINE BENCHMARK COMPARISON (Top-10)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"Pipeline finished in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    run_training_and_evaluation()
