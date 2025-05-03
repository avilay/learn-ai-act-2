from enum import StrEnum

from openai.types.chat.chat_completion_message import (  # noqa: F401; type: ignore
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import Function  # noqa: F401
from openai.types.responses.response_function_tool_call import (  # noqa: F401
    ResponseFunctionToolCall,
)


class LLM(StrEnum):
    # General purpose models
    PRE = "gpt-4.1"
    PRE_MINI = "gpt-4.1-mini"
    PRE_NANO = "gpt-4.1-nano"
    PRE_FAST = "gpt-4o"
    PRE_FAST_MINI = "gpt-4o-mini"

    # Tools models
    WEB_SEARCH = "gpt-4o-search-preview"
    COMPUTER_USE = "computer-use-preview"

    # Reasoning models
    RES = "o3"
    RES_MINI = "o4-mini"

    # Audio model
    AUDIO = "gpt-4o-audio"
