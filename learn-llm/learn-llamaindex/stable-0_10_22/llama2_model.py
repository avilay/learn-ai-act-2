import os

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    set_global_handler,
)
from llama_index.llms.ollama import Ollama


def main():
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        print("Set the OPENAI_API_KEY environment variable!")
        return

    set_global_handler("simple")
    llm = Ollama(model="llama2", request_timeout=30)
    Settings.llm = llm

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query("What did the author do growing up?")
    print(response)


if __name__ == "__main__":
    main()
