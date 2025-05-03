from enum import StrEnum

from openai.types.chat.chat_completion_message import (  # noqa: F401; type: ignore
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import Function  # noqa: F401


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


# Don't use this function, it does not handle apostrophes well.
def inverse(char):
    if char == ")":
        return "("
    elif char == "]":
        return "["
    else:
        raise ValueError(f"{char} is not a paren!")


def pretty_print_repr(reprstr: str) -> None:
    parens = []
    buf = []
    strings = []

    for char in reprstr:
        if char in ['"', "'"]:
            if strings and strings[-1] == char:
                # End of a string
                strings.pop()
            else:
                # Beginning of a string
                strings.append(char)

        if len(strings) > 0:
            # Inside a string
            buf.append(char)
            continue

        char = char.strip()
        if not char:
            continue

        if char in [")", "]"]:
            if buf:
                print("  " * len(parens), "".join(buf))
                buf.clear()
            assert inverse(char) == parens.pop()
            buf.append(char)
            continue

        buf.append(char)
        if char in ["(", "[", ","]:
            if buf:
                print("  " * len(parens), "".join(buf))
                buf.clear()
            if char in ["(", "["]:
                parens.append(char)

    if buf:
        print("  " * len(parens), "".join(buf))


if __name__ == "__main__":
    #     text = """
    # ChatCompletion(id='chatcmpl-BN62ezNYmOfNaNEbNEu58pvi1wlaO', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_TFRZlgvrguGShDwqCJeOzOcj', function=Function(arguments='{"location":"Paris, France"}', name='get_weather'), type='function')]))], created=1744844468, model='gpt-4.1-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_beec22d258', usage=CompletionUsage(completion_tokens=17, prompt_tokens=65, total_tokens=82, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # """
    text = """
ChatCompletion(id='chatcmpl-BN7UB00QRNy2faSvu56aNoMeDbwIO', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="The temperature in Paris today is around 15°C. If you would like more details about the weather (such as if it's sunny or rainy), just let me know!", refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1744850019, model='gpt-4.1-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_beec22d258', usage=CompletionUsage(completion_tokens=36, prompt_tokens=90, total_tokens=126, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
"""
    pretty_print_repr(text)
