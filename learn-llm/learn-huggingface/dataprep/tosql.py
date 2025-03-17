#!/usr/bin/env python

from collections import namedtuple
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Callable, Protocol

from cprint import danger_print
from datasets import Dataset
from tqdm.auto import tqdm

SplitInfo = namedtuple("SplitInfo", ["split_name", "table_name", "feature_file"])
RowBuilder = Callable[[int, Dataset], dict[str, Any]]


def validate(outdir: Path, dbname: str) -> bool:
    if not outdir.exists():
        danger_print(f"{outdir} should already exist!")
        return False
    if not outdir.is_dir():
        danger_print(f"{outdir} should be a directory!")
        return False
    if (outdir / dbname).exists():
        danger_print(f"The {dbname} file already exists in {outdir}!")
        return False
    return True


def create(conn: Connection, splits: dict[str, SplitInfo], sql: str):
    for split_info in splits.values():
        conn.execute(sql.format(split_info.table_name))
    conn.commit()


def insert(conn: Connection, dataset: Dataset, sql: str, build_row: RowBuilder):
    rows: list[dict[str, Any]] = []
    for i in tqdm(range(len(dataset))):
        if i % 1000 == 0:
            conn.executemany(sql, rows)
            conn.commit()
            rows = []
        rows.append(build_row(i, dataset))
    if rows:
        conn.executemany(sql, rows)
        conn.commit()


class HuggingFaceDataset(Protocol):
    def dbname(self) -> str:
        ...

    def row_builder(self) -> RowBuilder:
        ...

    def splits(self) -> dict[str, SplitInfo]:
        ...

    def insert_sql(self, table_name: str) -> str:
        ...

    def create_sql_template(self) -> str:
        ...

    def dataset(self) -> Dataset:
        ...
