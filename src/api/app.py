import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import (
    HealthResponse,
    ItemSimilarityResponse,
    RecommendationResponse,
    RecommendedItem,
)
from src.config import MODELS_DIR
from src.models.pipeline import HybridFallbackPipeline

# Prometheus Metrics Instrumentation
REQUEST_COUNT = Counter(
    "recsys_requests_total",
    "Total count of recommendation requests",
    ["endpoint", "model", "status"],
)
REQUEST_LATENCY = Histogram(
    "recsys_request_latency_seconds",
    "Histogram of request latencies in seconds",
    ["endpoint", "model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
FALLBACK_COUNT = Counter(
    "recsys_fallback_total",
    "Total count of fallback activations",
    ["requested_model", "fallback_strategy"],
)

pipeline_instance: HybridFallbackPipeline | None = None


def load_pipeline_artifacts() -> HybridFallbackPipeline:
    """Load pre-trained pipeline or train on the fly."""
    pipeline_file = MODELS_DIR / "pipeline.pkl"
    if not pipeline_file.exists():
        print("Pipeline file not found. Running training on the fly...")
        from src.train_and_eval import run_training_and_evaluation

        run_training_and_evaluation()

    loaded = joblib.load(pipeline_file)
    print("Loaded RecSys pipeline from disk.")
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    global pipeline_instance
    pipeline_instance = load_pipeline_artifacts()
    yield


app = FastAPI(
    title="RecSys Personalization API",
    description="Educational recommendation demo with model comparison, fallback and Prometheus metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
def serve_ui():
    """Serve the interactive RecSys Demo UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>RecSys API is running. Open <a href='/docs'>/docs</a> for Swagger UI.</h1>")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint for orchestrators (k8s / docker)."""
    return HealthResponse(
        status="healthy",
        service="recsys-api",
        models_loaded=pipeline_instance is not None,
        version="1.0.0",
    )


@app.get("/metrics", tags=["Observability"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recommend/user/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend_user(
    user_id: int,
    model_type: str = Query(
        "collaborative",
        description="Model to use: collaborative, content_based, cooccurrence, popularity",
    ),
    top_k: int = Query(10, ge=1, le=50, description="Number of items to recommend"),
    exclude_seen: bool = Query(True, description="Exclude items already consumed by user"),
    filter_genre: str | None = Query(None, description="Optional genre constraint"),
):
    """Generate personalized recommendations for a user with hierarchical fallback."""
    if pipeline_instance is None:
        REQUEST_COUNT.labels(endpoint="recommend_user", model=model_type, status="503").inc()
        raise HTTPException(status_code=503, detail="Models not loaded")

    valid_models = {"collaborative", "content_based", "cooccurrence", "popularity"}
    if model_type not in valid_models:
        REQUEST_COUNT.labels(endpoint="recommend_user", model=model_type, status="400").inc()
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Must be one of {valid_models}")

    t0 = time.perf_counter()
    try:
        result = pipeline_instance.recommend(
            user_id=user_id,
            model_type=model_type,
            top_k=top_k,
            exclude_seen=exclude_seen,
            filter_genre=filter_genre,
        )
        duration = time.perf_counter() - t0

        # Prometheus tracking
        REQUEST_COUNT.labels(endpoint="recommend_user", model=model_type, status="200").inc()
        REQUEST_LATENCY.labels(endpoint="recommend_user", model=model_type).observe(duration)

        if result.get("is_fallback"):
            FALLBACK_COUNT.labels(
                requested_model=model_type, fallback_strategy=result.get("strategy_used")
            ).inc()

        return result
    except (ValueError, KeyError, RuntimeError) as e:
        REQUEST_COUNT.labels(endpoint="recommend_user", model=model_type, status="500").inc()
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/recommend/item/{item_id}", response_model=ItemSimilarityResponse, tags=["Recommendations"])
def recommend_similar_items(
    item_id: int,
    model_type: str = Query("content_based", description="Model: content_based, collaborative, cooccurrence"),
    top_k: int = Query(10, ge=1, le=50),
):
    """Find similar items (Item-to-Item) based on content or co-consumption."""
    if pipeline_instance is None:
        REQUEST_COUNT.labels(endpoint="recommend_similar_items", model=model_type, status="503").inc()
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        t0 = time.perf_counter()
        sim_items = []

        if model_type == "content_based":
            sim_items = pipeline_instance.content_based.similar_items(item_id, top_k=top_k)
        elif model_type == "collaborative":
            sim_items = pipeline_instance.collaborative.similar_items(item_id, top_k=top_k)
        elif model_type == "cooccurrence":
            sim_items = pipeline_instance.cooccurrence.similar_items(item_id, top_k=top_k)
        else:
            REQUEST_COUNT.labels(endpoint="recommend_similar_items", model=model_type, status="400").inc()
            raise HTTPException(
                status_code=400,
                detail="Model type for item similarity must be one of: content_based, collaborative, cooccurrence",
            )

        duration = time.perf_counter() - t0
        REQUEST_COUNT.labels(endpoint="recommend_similar_items", model=model_type, status="200").inc()
        REQUEST_LATENCY.labels(endpoint="recommend_similar_items", model=model_type).observe(duration)

        enriched = []
        if pipeline_instance.items_df is not None:
            items_lookup = pipeline_instance.items_df.set_index("movieId").to_dict(orient="index")
            for m_id, score in sim_items:
                meta = items_lookup.get(m_id, {})
                enriched.append(
                    RecommendedItem(
                        item_id=int(m_id),
                        score=round(float(score), 4),
                        title=meta.get("title", f"Movie #{m_id}"),
                        genres=meta.get("genres", ""),
                        year=meta.get("year", 0),
                    )
                )

        return ItemSimilarityResponse(
            item_id=item_id,
            model_type=model_type,
            count=len(enriched),
            similar_items=enriched,
            latency_ms=round(duration * 1000.0, 2),
        )
    except (ValueError, KeyError, RuntimeError) as e:
        REQUEST_COUNT.labels(endpoint="recommend_similar_items", model=model_type, status="500").inc()
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/recommend/popular", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend_popular(
    top_k: int = Query(10, ge=1, le=50),
    filter_genre: str | None = Query(None, description="Optional genre constraint"),
):
    """Retrieve global or genre-filtered popular items using Bayesian IMDb rating."""
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        return pipeline_instance.recommend(
            user_id=-1,
            model_type="popularity",
            top_k=top_k,
            exclude_seen=False,
            filter_genre=filter_genre,
        )
    except (ValueError, KeyError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail="Internal server error") from e
