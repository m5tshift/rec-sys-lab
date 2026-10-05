import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATASET_NAME, MOVIELENS_URL, RAW_DATA_DIR


class DataLoader:
    """Handles downloading and loading the MovieLens dataset."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, dataset_name: str = DATASET_NAME):
        self.raw_dir = raw_dir
        self.dataset_name = dataset_name
        self.extracted_path = self.raw_dir / self.dataset_name

    def ensure_dataset(self) -> None:
        """Download and unpack the dataset if not already present."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.extracted_path.exists() and (self.extracted_path / "movies.csv").exists():
            return

        print(f"Downloading dataset from {MOVIELENS_URL}...")
        req = urllib.request.Request(
            MOVIELENS_URL,
            headers={"User-Agent": "RecSys-Lab-Loader/1.0"}
        )
        with urllib.request.urlopen(req) as response:
            zip_content = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
            zip_ref.extractall(self.raw_dir)
        print(f"Dataset extracted to {self.extracted_path}")

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load movies, ratings, and tags dataframes."""
        self.ensure_dataset()

        movies_path = self.extracted_path / "movies.csv"
        ratings_path = self.extracted_path / "ratings.csv"
        tags_path = self.extracted_path / "tags.csv"

        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        tags_df = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame()

        return movies_df, ratings_df, tags_df
