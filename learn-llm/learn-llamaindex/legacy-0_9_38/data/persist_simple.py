from pathlib import Path
from datetime import datetime
from cprint import info_print

start = datetime.now()
from dotenv import load_dotenv  # noqa
from llama_index import VectorStoreIndex, set_global_handler  # noqa
from llama_index.node_parser import SentenceSplitter  # noqa
from llama_index import SimpleDirectoryReader  # noqa

end = datetime.now()

print(f"Took {end - start} time to load the imports.")

WORKINGDIR = Path.home() / "Desktop" / "temp" / "llama-index"


load_dotenv()
set_global_handler("simple")


docs = SimpleDirectoryReader("./data", recursive=True).load_data(show_progress=True)
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
nodes = splitter.get_nodes_from_documents(docs)
info_print(f"\nAdding {len(nodes)} nodes to the index.\n")
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()
resp = query_engine.query("How to create a new conda environment?")
print(resp)
info_print("\nPersisting index\n")
index.storage_context.persist(persist_dir=str(WORKINGDIR / "storage"))
