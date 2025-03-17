import pickle
import sys
from pathlib import Path
from sqlite3 import Row, connect

import click
from cprint import info_print
from dataprep.dialogsum import DialogSummary
from dataprep.glue_mrpc import GlueMRPC
from dataprep.quora import Quora
from dataprep.tosql import create, insert, validate
from dataprep.yelp_review_full import YelpReviewFull


@click.command()
@click.option(
    "--outdir", default="~/mldata/", help="Directory where the db file will be created."
)
@click.option("--dataset", help="Dataset to download and convert to sqlite3.")
def main(outdir: str, dataset: str):
    if dataset == "dialogsum":
        ds = DialogSummary()
    elif dataset == "yelp-review-full":
        ds = YelpReviewFull()
    elif dataset == "glue-mrpc":
        ds = GlueMRPC()
    elif dataset == "quora":
        ds = Quora()

    outdir = Path.expanduser(Path(outdir))
    if not validate(outdir, ds.dbname()):
        sys.exit(1)

    dataset = ds.dataset()
    dbfile = outdir / ds.dbname()
    conn = connect(dbfile)
    conn.row_factory = Row
    create(conn, ds.splits(), ds.create_sql_template())
    for split, split_info in ds.splits().items():
        info_print(f"\nPopulating {split_info.table_name}...")
        insert(
            conn, dataset[split], ds.insert_sql(split_info.table_name), ds.row_builder()
        )
        with open(outdir / split_info.feature_file, "wb") as f:
            pickle.dump(dataset[split].features, f)


if __name__ == "__main__":
    main()
