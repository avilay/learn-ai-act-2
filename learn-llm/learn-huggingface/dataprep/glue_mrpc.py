from typing import Any

from dataprep.tosql import HuggingFaceDataset, RowBuilder, SplitInfo
from datasets import Dataset, load_dataset


class GlueMRPC(HuggingFaceDataset):
    def dbname(self) -> str:
        return "glue_mrpc.db"

    def row_builder(self) -> RowBuilder:
        def build_row(i: int, dataset: Dataset) -> dict[str, Any]:
            return {
                "id": i + 1,
                "sentence1": dataset[i]["sentence1"],
                "sentence2": dataset[i]["sentence2"],
                "label": dataset[i]["label"],
            }

        return build_row

    def splits(self) -> dict[str, SplitInfo]:
        return {
            "train": SplitInfo("train", "trainset", "train_features.pkl"),
            "validation": SplitInfo("validation", "valset", "val_features.pkl"),
            "test": SplitInfo("test", "testset", "test_features.pkl"),
        }

    def insert_sql(self, table_name) -> str:
        return f"INSERT INTO {table_name} VALUES (:id, :sentence1, :sentence2, :label)"

    def create_sql_template(self) -> str:
        return """
CREATE TABLE IF NOT EXISTS {0} (
    id INTEGER PRIMARY KEY,
    sentence1 TEXT,
    sentence2 TEXT,
    label INTEGER
)
"""

    def dataset(self) -> Dataset:
        return load_dataset("glue", "mrpc")
