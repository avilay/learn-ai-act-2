from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex, set_global_handler


def main():
    load_dotenv()
    set_global_handler("simple")
    documents = SimpleDirectoryReader("./data/", recursive=True).load_data(
        show_progress=True
    )
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    query_result = query_engine.query("What did the author do growing up?")
    print(query_result)
    query_result = query_engine.query("How to activate conda environment?")
    print(query_result)


if __name__ == "__main__":
    main()
