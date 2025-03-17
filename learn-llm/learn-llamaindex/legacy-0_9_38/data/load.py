from dotenv import load_dotenv
from llama_index.storage import StorageContext
from llama_index import load_index_from_storage


load_dotenv()
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
print(f"Loaded index of type {type(index)}")
query_engine = index.as_query_engine()
resp = query_engine.query("How to create a new conda environment?")
print(resp)
