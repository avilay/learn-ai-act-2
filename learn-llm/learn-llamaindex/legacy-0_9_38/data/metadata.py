from pathlib import Path
import pickle

# from dotenv import load_dotenv

from llama_index import ServiceContext, set_global_service_context

from llama_index.embeddings import OllamaEmbedding
from llama_index.llms import Ollama

from llama_index.readers.file.flat_reader import FlatReader
from llama_index.node_parser.file import SimpleFileNodeParser
from llama_index.node_parser import SentenceSplitter
from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    EntityExtractor,
)
from llama_index.ingestion import IngestionPipeline

DATAROOT = Path.home() / "mldata" / "sherlock"
TMPROOT = Path.home() / "Desktop" / "temp"
# load_dotenv()

llm = Ollama(model="llama2", request_timeout=60.0)
embed_model = OllamaEmbedding("llama2")
sc = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(sc)

reader = FlatReader()
parser = SimpleFileNodeParser()
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
title_extractor = TitleExtractor(llm=llm)
qna_extractor = QuestionsAnsweredExtractor(llm=llm)
entity_extractor = EntityExtractor()

docs = reader.load_data(DATAROOT / "md" / "advs.md")
nodes = parser.get_nodes_from_documents(docs)
transformations = [splitter, title_extractor, qna_extractor, entity_extractor]
pipeline = IngestionPipeline(transformations=transformations)
chunks = pipeline.run(documents=nodes)
print(f"Got {len(chunks)} number of chunks.")

# filename = TMPROOT / "chunks" / "chunk_70.pkl"
# with open(filename, "wb") as fout:
#     pickle.dump(chunks[70], fout)

print("Pickling the chunks...")
for i, chunk in enumerate(chunks):
    if chunk is not None:
        filename = TMPROOT / "chunks" / f"chunk_{i}.pkl"
        with open(filename, "wb") as fout:
            pickle.dump(chunk, fout)
    else:
        print(f"Skipping chunk[{i}] because it is None.")
