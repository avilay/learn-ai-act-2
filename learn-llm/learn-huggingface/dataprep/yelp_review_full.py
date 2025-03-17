from dataprep.tosql import HuggingFaceDataset, RowBuilder, SplitInfo
from datasets import Dataset, load_dataset
from typing import Any


class YelpReviewFull(HuggingFaceDataset):
    def dbname(self) -> str:
        return "yelp-review-full.db"

    def row_builder(self) -> RowBuilder:
        def build_row(i: int, dataset: Dataset) -> dict[str, Any]:
            return {
                "id": i + 1,
                "content": dataset[i]["text"],
                "label": dataset[i]["label"],
            }

        return build_row

    def splits(self) -> dict[str, SplitInfo]:
        return {
            "train": SplitInfo("train", "trainset", "train_features.pkl"),
            "test": SplitInfo("test", "testset", "test_features.pkl"),
        }

    def insert_sql(self, table_name: str) -> str:
        return f"INSERT INTO {table_name} VALUES (:id, :content, :label)"

    def create_sql_template(self) -> str:
        return """
CREATE TABLE IF NOT EXISTS {0} (
    id INTEGER PRIMARY KEY,
    content TEXT,
    label INTEGER
)
"""

    def dataset(self) -> Dataset:
        return load_dataset("yelp_review_full")
