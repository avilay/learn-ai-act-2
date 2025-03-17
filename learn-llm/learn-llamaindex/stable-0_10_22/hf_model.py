from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
    set_global_handler,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from dotenv import load_dotenv
import os

# import torch


system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha Version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


def main():
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        print("Set the OPENAI_API_KEY environment variable!")
        return

    set_global_handler("simple")
    documents = SimpleDirectoryReader("data").load_data()
    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
        model_name="StabilityAI/stablelm-tuned-alpha-3b",
        device_map="auto",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        model_kwargs={"offload_folder": "offload"},
    )
    Settings.llm = llm
    Settings.chunk_size = 1024

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)


if __name__ == "__main__":
    main()
