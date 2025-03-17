import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.node_parser import SentenceSplitter
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone
from llama_index import SimpleDirectoryReader
from datetime import datetime
from llama_index.ingestion import IngestionPipeline
from llama_index import VectorStoreIndex

DATADIR = Path.home() / "mldata" / "toy-llm"
load_dotenv()

FILETYPES = {".md": "Markdown", ".png": "Image", ".pdf": "PDF", ".txt": "Text"}


def file_metadata(filename):
    filepath = Path(filename)
    name = filepath.name
    filetype = FILETYPES.get(filepath.suffix, "Unknown")
    stat = filepath.stat()
    size = stat.st_size
    created_on = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d")
    return {
        "filepath": str(filepath),
        "name": name,
        "filetype": filetype,
        "size_bytes": size,
        "created_on": created_on,
    }


pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# spec = PodSpec(environment="gcp-starter")
# pinecone.create_index(
#     name="quickllamaindex", dimension=1536, metric="euclidean", spec=spec
# )
pinecone_index = pinecone.Index("learn-llamaindex")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

docs = SimpleDirectoryReader(
    str(DATADIR), recursive=True, file_metadata=file_metadata
).load_data(show_progress=True)

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
    vector_store=vector_store,
)
pipeline.run(documents=docs)
index = VectorStoreIndex.from_vector_store(vector_store)

# splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
# nodes = splitter.get_nodes_from_documents(docs)
# info_print(f"\nAdding {len(nodes)} nodes to the index.\n")
# index = VectorStoreIndex(nodes, storage_context=storage_context)
query_engine = index.as_query_engine()
resp = query_engine.query("How to create a new conda environment?")
print(resp)
# info_print("\nPersisting index\n")
# storage_context.persist(persist_dir=str(WORKINGDIR / "pc-storage"))
