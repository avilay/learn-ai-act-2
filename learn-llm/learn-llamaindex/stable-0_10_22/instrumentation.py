# import nest_asyncio

# nest_asyncio.apply()
from dotenv import load_dotenv

load_dotenv()

from llama_index.core.instrumentation.event_handlers import BaseEventHandler  # noqa
from llama_index.core.instrumentation.span import BaseSpan  # noqa
from typing import Any, Optional  # noqa
from llama_index.core.bridge.pydantic import Field  # noqa
from llama_index.core.instrumentation.span_handlers import (  # noqa
    BaseSpanHandler,
    SimpleSpanHandler,
)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex  # noqa
import llama_index.core.instrumentation as instrument  # noqa
import numpy as np  # noqa


class MyEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "MyEventHandler"

    def handle(self, event) -> None:
        event = event.dict()

        # def print_keys(parent, obj):
        #     if isinstance(obj, dict):
        #         for key, value in obj.items():
        #             full_key = f"{parent}.{key}"
        #             print(full_key)
        #             print_keys(full_key, value)
        # print_keys("", event)
        # print("")
        if_print = lambda hsh, key, value=lambda val: val: (
            print(key, ": ", value(hsh[key])) if key in hsh else print("", end="")
        )

        if_print(event, "timestamp")
        if "model_dict" in event:
            model = event["model_dict"]
            if_print(model, "model_name")
            if_print(model, "embed_batch_size")
            if_print(model, "dimensions")
            if_print(model, "class_name")
            if_print(model, "system_prompt")
            if_print(model, "model")
            if_print(model, "temperature")
            if_print(model, "max_tokens")
            if_print(model, "logprobs")
            if_print(model, "top_logprobs")
        if_print(event, "class_name")
        if "chunks" in event:
            print("chunks : ")
            for i, chunk in enumerate(event["chunks"]):
                print(f"[{i}]: {chunk[:100]}...")
        if_print(event, "embeddings", value=lambda val: np.array(val).shape)
        if_print(event, "query")
        if_print(event, "messages")
        if "response" in event:
            resp = event["response"]
            if_print(resp, "message")
            if_print(resp, "raw")
            if_print(resp, "delta")
            if_print(resp, "logprobs")

        print("")


# class MyCustomSpan(BaseSpan):
#     custom_field_1: Any = Field(...)
#     custom_field_2: Any = Field(...)


# class MyCustomSpanHandler(BaseSpanHandler[MyCustomSpan]):
#     @classmethod
#     def class_name(cls) -> str:
#         return "MyCustomSpanHandler"

#     def new_span(
#         self, id: str, parent_span_id: Optional[str], **kwargs
#     ) -> Optional[MyCustomSpan]:
#         pass

#     def prepare_to_exit_span(self, id: str, result: Any | None = None, **kwargs) -> Any:
#         pass

#     def prepare_to_drop_span(self, id: str, err: Exception | None, **kwargs) -> Any:
#         pass


def main():
    dispatcher = instrument.get_dispatcher()
    handler = MyEventHandler()
    dispatcher.add_event_handler(handler)
    dispatcher.add_span_handler(SimpleSpanHandler())

    docs = SimpleDirectoryReader(input_dir="./data").load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    query_result = query_engine.query("Who is Paul?")
    print(query_result)
    print("")


if __name__ == "__main__":
    main()
