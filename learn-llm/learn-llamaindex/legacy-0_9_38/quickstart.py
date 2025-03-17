from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARN)


def main():
    load_dotenv()
    documents = SimpleDirectoryReader("./data/", recursive=True).load_data(
        show_progress=True
    )
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    query = input("\nType your query or 'QUIT' and press [ENTER]: ")
    query = query.strip()
    while query.lower() != "quit":
        query_result = query_engine.query(query)
        print(query_result)
        query = input("\nType your query or 'QUIT' and press [ENTER]: ")
        query = query.strip()


if __name__ == "__main__":
    main()
