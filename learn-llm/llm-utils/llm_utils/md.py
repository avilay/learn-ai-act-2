import glob
import os
from collections import namedtuple
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents.base import Document
from markdown import markdown

MdFile = namedtuple("MdFile", ["filepath", "contents"])


class BadDirError(Exception):
    def __init__(self, msg):
        self.msg = msg


def _validate(in_dir: Path) -> tuple[bool, str]:
    if not in_dir.exists():
        # raise BadDirError(f"{in_dir} does not exist!")
        return (False, f"{in_dir} does not exist!")
    if not in_dir.is_dir():
        # raise BadDirError(f"{in_dir} is not a directory!")
        return (False, f"{in_dir} is not a directory!")
    if not any(os.scandir(in_dir)):
        # raise BadDirError(f"{in_dir} is empty!")
        return (False, f"{in_dir} is empty!")
    return (True, "")


def md_to_text(in_dir: Path) -> Iterable[MdFile]:
    is_valid, err = _validate(in_dir)
    if not is_valid:
        raise BadDirError(err)

    for md_file in glob.glob("**/*.md", root_dir=in_dir, recursive=True):
        filepath = in_dir / md_file
        with open(filepath, "rt") as fin:
            html = markdown(fin.read())
            soup = BeautifulSoup(html, features="html.parser")
            contents = soup.get_text()
        yield MdFile(md_file, contents)


def chunk_md(
    in_dir: Path,
    headers_to_split_on: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[Document]:
    is_valid, err = _validate(in_dir)
    if not is_valid:
        raise BadDirError(err)

    for md_file in glob.glob("**/*.md", root_dir=in_dir, recursive=True):
        filepath = in_dir / md_file
        with open(filepath, "rt") as fin:
            md_doc = fin.read()

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True
        )
        md_splits = md_splitter.split_text(md_doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(md_splits)

        for split in splits:
            metadata = split.metadata
            headers = "\n".join(metadata.values())
            headers += "\n"
            split.page_content = headers + split.page_content

            split.metadata["file"] = str(filepath)

            yield split
