from typing import Any

from dataprep.tosql import HuggingFaceDataset, RowBuilder, SplitInfo
from datasets import Dataset, load_dataset


class DialogSummary(HuggingFaceDataset):
    def dbname(self) -> str:
        return "dialogsum.db"

    def row_builder(self) -> RowBuilder:
        def build_row(i: int, dataset: Dataset) -> dict[str, Any]:
            return {
                "id": i + 1,
                "dsid": dataset[i]["id"],
                "dialog": dataset[i]["dialogue"],
                "summary": dataset[i]["summary"],
                "topic": dataset[i]["topic"],
            }

        return build_row

    def splits(self) -> dict[str, SplitInfo]:
        return {
            "train": SplitInfo("train", "trainset", "train_features.pkl"),
            "validation": SplitInfo("validation", "valset", "val_features.pkl"),
            "test": SplitInfo("test", "testset", "test_features.pkl"),
        }

    def insert_sql(self, table_name: str) -> str:
        return (
            f"INSERT INTO {table_name} VALUES (:id, :dsid, :dialog, :summary, :topic)"
        )

    def create_sql_template(self) -> str:
        return """
CREATE TABLE IF NOT EXISTS {0} (
    id INTEGER PRIMARY KEY,
    dsid VARCHAR(12),
    dialog TEXT,
    summary TEXT,
    topic VARCHAR(45)
)
"""

    def dataset(self) -> Dataset:
        return load_dataset("knkarthick/dialogsum")
