from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

# Dataset configuration
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_NAME = "ml-latest-small"

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_DEFAULT = 10
BAYESIAN_PRIOR_CONFIDENCE = 50.0  # Minimum vote count weight m in IMDb formula
SVD_N_COMPONENTS = 20
COOCCURRENCE_MIN_SUPPORT = 5

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
