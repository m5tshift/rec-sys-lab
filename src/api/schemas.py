from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    item_id: int = Field(..., description="Unique movie identifier")
    score: float = Field(..., description="Relevance / predicted score")
    title: str = Field(..., description="Movie title")
    genres: str = Field(..., description="Genres associated with movie")
    year: int = Field(..., description="Release year")


class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="Target user identifier")
    model_requested: str = Field(..., description="Algorithm requested by client")
    strategy_used: str = Field(..., description="Actual strategy employed (primary or fallback)")
    is_fallback: bool = Field(..., description="True if fallback logic was triggered")
    count: int = Field(..., description="Number of recommendations returned")
    recommendations: list[RecommendedItem] = Field(..., description="Ranked list of recommended items")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class ItemSimilarityResponse(BaseModel):
    item_id: int = Field(..., description="Source item identifier")
    model_type: str = Field(..., description="Algorithm used for item similarity")
    count: int = Field(..., description="Number of similar items returned")
    similar_items: list[RecommendedItem] = Field(..., description="Ranked list of similar items")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool
    version: str
