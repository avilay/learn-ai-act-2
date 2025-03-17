from typing import Any

from dataprep.tosql import HuggingFaceDataset, RowBuilder, SplitInfo
from datasets import Dataset, load_dataset


class Quora(HuggingFaceDataset):
    def dbname(self) -> str:
        return "quora.db"

    def row_builder(self) -> RowBuilder:
        def build_row(i: int, dataset: Dataset) -> dict[str, Any]:
            return {
                "id": i + 1,
                "question_1_id": dataset[i]["questions"]["id"][0],
                "question_2_id": dataset[i]["questions"]["id"][1],
                "question_1": dataset[i]["questions"]["text"][0],
                "question_2": dataset[i]["questions"]["text"][1],
                "is_duplicate": dataset[i]["is_duplicate"],
            }

        return build_row

    def splits(self) -> dict[str, SplitInfo]:
        return {"train": SplitInfo("train", "trainset", "train_features.pkl")}

    def insert_sql(self, table_name: str) -> str:
        return f"INSERT INTO {table_name} VALUES (:id, :question_1_id, :question_2_id, :question_1, :question_2, :is_duplicate)"

    def create_sql_template(self) -> str:
        return """
CREATE TABLE IF NOT EXISTS {0} (
    id INTEGER PRIMARY KEY,
    question_1_id VARCHAR(10),
    question_2_id VARCHAR(10),
    question_1 TEXT,
    question_2 TEXT,
    is_duplicate BOOLEAN
)
"""

    def dataset(self) -> Dataset:
        return load_dataset("quora")
