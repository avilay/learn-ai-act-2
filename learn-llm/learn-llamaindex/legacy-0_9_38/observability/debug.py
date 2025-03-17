import logging
import sys

from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager, llm=llm
)
docs = SimpleDirectoryReader("./data/", recursive=True).load_data(show_progress=True)
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
query_result = query_engine.query("What did the author do growing up?")
print(query_result)
