import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import RANDOM_STATE, TEST_SIZE


class DataPreprocessor:
    """Preprocesses MovieLens metadata and interactions into model-ready structures."""

    def __init__(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
        self.test_size = test_size
        self.random_state = random_state

        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}

    @staticmethod
    def extract_year(title: str) -> tuple[str, int]:
        """Extract clean title and 4-digit release year."""
        match = re.search(r"\((\d{4})\)", title)
        year = int(match.group(1)) if match else 0
        clean_title = re.sub(r"\s*\(\d{4}\)", "", title).strip()
        return clean_title, year

    def prepare_item_metadata(
        self, movies_df: pd.DataFrame, tags_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create enriched item metadata dataframe with tags and cleaned genres."""
        items = movies_df.copy()

        # Extract year and clean title
        titles_years = items["title"].apply(self.extract_year)
        items["clean_title"] = [t[0] for t in titles_years]
        items["year"] = [t[1] for t in titles_years]

        # Clean genres: replace pipe with space
        items["genres_clean"] = items["genres"].fillna("").apply(
            lambda g: "" if g == "(no genres listed)" else g.replace("|", " ")
        )

        # Aggregate user tags per item
        if not tags_df.empty:
            tags_agg = (
                tags_df.groupby("movieId")["tag"]
                .apply(lambda tags: " ".join(tags.astype(str).str.lower()))
                .reset_index()
                .rename(columns={"tag": "tags_clean"})
            )
            items = items.merge(tags_agg, on="movieId", how="left")
            items["tags_clean"] = items["tags_clean"].fillna("")
        else:
            items["tags_clean"] = ""

        # Build combined content corpus
        items["content_corpus"] = (
            items["clean_title"]
            + " "
            + items["genres_clean"]
            + " "
            + items["tags_clean"]
        ).str.strip()

        return items

    def split_train_test(
        self, ratings_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions temporally per user to evaluate future recommendations."""
        ratings_sorted = ratings_df.sort_values(by=["userId", "timestamp"])

        train_records = []
        test_records = []

        for _, user_group in ratings_sorted.groupby("userId"):
            n_ratings = len(user_group)
            if n_ratings < 5:
                # Keep small profiles in train to prevent complete zero training
                train_records.append(user_group)
            else:
                n_test = max(1, int(n_ratings * self.test_size))
                train_records.append(user_group.iloc[:-n_test])
                test_records.append(user_group.iloc[-n_test:])

        train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
        test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()
        return train_df, test_df


    def build_interaction_matrix(
        self, ratings_df: pd.DataFrame, binary: bool = False
    ) -> csr_matrix:
        """Build sparse user-item interaction matrix."""
        valid_ratings = ratings_df[
            ratings_df["userId"].isin(self.user_to_idx)
            & ratings_df["movieId"].isin(self.item_to_idx)
        ]

        rows = [self.user_to_idx[u] for u in valid_ratings["userId"]]
        cols = [self.item_to_idx[m] for m in valid_ratings["movieId"]]
        data = (
            np.ones(len(valid_ratings), dtype=np.float32)
            if binary
            else valid_ratings["rating"].values.astype(np.float32)
        )

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
