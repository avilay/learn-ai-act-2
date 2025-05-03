import asyncio
import logging

from agents import (  # noqa
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    Runner,
    input_guardrail,
)
from dotenv import load_dotenv
from pydantic import BaseModel
from snippets import configure_logger  # type: ignore

load_dotenv()

configure_logger(level=logging.DEBUG, chatty_modules=["httpcore"])

"""
Calling LLM gpt-4o -
Instructions:
You determine which agent to use based on the user's homework.
Input:
[
  {
    "content": "Who was the first president of the United States?",
    "role": "user"
  }
]
Tools:
[
  {
    "name": "transfer_to_history_tutor",
    "parameters": {
      "additionalProperties": false,
      "type": "object",
      "properties": {},
      "required": []
    },
    "strict": true,
    "type": "function",
    "description": "Handoff to the History Tutor agent to handle the request. Specialist agent for historical questions."
  },
  {
    "name": "transfer_to_math_tutor",
    "parameters": {
      "additionalProperties": false,
      "type": "object",
      "properties": {},
      "required": []
    },
    "strict": true,
    "type": "function",
    "description": "Handoff to the Math Tutor agent to handle the request. Specialist agent for math questions."
  }
]

LLM Resp:
[
  {
    "arguments": "{}",
    "call_id": "call_VwXeuEG2viYUayUIfXiboUHg",
    "name": "transfer_to_history_tutor",
    "type": "function_call",
    "id": "fc_680437476da48191966a10f765b13f9f0ca3674611fc6a09",
    "status": "completed"
  }
]

Calling LLM gpt-4o -
Instructions:
You provide assistance queries. Explain important events and context clearly.
Input:
[
  {
    "content": "Who was the first president of the United States?",
    "role": "user"
  },
  {
    "arguments": "{}",
    "call_id": "call_VwXeuEG2viYUayUIfXiboUHg",
    "name": "transfer_to_history_tutor",
    "type": "function_call",
    "id": "fc_680437476da48191966a10f765b13f9f0ca3674611fc6a09",
    "status": "completed"
  },
  {
    "call_id": "call_VwXeuEG2viYUayUIfXiboUHg",
    "output": "{'assistant': 'History Tutor'}",
    "type": "function_call_output"
  }
]

LLM Resp -
[
  {
    "id": "msg_68043748658c8191bd6ec362bd15af6c0ca3674611fc6a09",
    "content": [
      {
        "annotations": [],
        "text": "The first president of the United States was George Washington. He served from 1789 to 1797 and is often referred to asthe \"Father of His Country.\" Washington was a key figure during the American Revolutionary War and presided over the Constitutional Convention in 1787. His leadership helped establish the foundations of the new nation.",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]

In parallel the guardrail agent is also run -
Calling LLM gpt-4o -
Instructions:
Check if the user is asking about homework.
Input:
[
  {
    "content": "Who was the first president of the United States?",
    "role": "user"
  }
]
Response format: {
  'format': {
    'type': 'json_schema',
    'name': 'final_output',
    'schema': {
      'properties': {
        'is_homework': {'title': 'Is Homework', 'type': 'boolean'},
        'reasoning': {'title': 'Reasoning', 'type': 'string'}
      },
      'required': ['is_homework', 'reasoning'],
      'title': 'HomeworkOutput',
      'type': 'object',
      'additionalProperties': False
    },
    'strict': True
  }
}

LLM Resp -
[
  {
    "id": "msg_680437472194819191af60743ccbd0fb0f8bd324fbbfb0cc",
    "content": [
      {
        "annotations": [],
        "text": "{\"is_homework\":true,\"reasoning\":\"The question asks for factual historical information that is often related to academic studies and assignments, particularly in history classes.\"}",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]
"""


class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str


@input_guardrail
async def homework_guardrail(ctx, agent, input_data):
    guardrail_agent = Agent(
        name="Guardrail Check",
        instructions="Check if the user is asking about homework.",
        output_type=HomeworkOutput,
    )
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output, tripwire_triggered=not final_output.is_homework
    )


async def main():
    math_tutor = Agent(
        name="Math Tutor",
        handoff_description="Specialist agent for math questions.",
        instructions="You provide help with math problems. Explain your reasoning at each step and include examples.",
    )

    history_tutor = Agent(
        name="History Tutor",
        handoff_description="Specialist agent for historical questions.",
        instructions="You provide assistance queries. Explain important events and context clearly.",
    )

    triager = Agent(
        name="Triage Agent",
        instructions="You determine which agent to use based on the user's homework.",
        handoffs=[history_tutor, math_tutor],
        input_guardrails=[homework_guardrail],  # type: ignore
    )

    try:
        result = await Runner.run(
            triager, "Write a blog post on software as a service."
        )
        print(result.final_output)
    except InputGuardrailTripwireTriggered as err:
        print(err)


if __name__ == "__main__":
    asyncio.run(main())
