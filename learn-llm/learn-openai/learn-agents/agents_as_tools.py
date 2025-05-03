import asyncio
import json
import logging

import requests
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel
from snippets import configure_logger  # type: ignore

load_dotenv()

configure_logger(logging.ERROR, chatty_modules=["httpcore"])

"""
First input: "Order me a double shot cappuccino."
--------------------------------------------------
Running agent Orchestrator Agent (turn 1)

Calling LLM gpt-4o -
Instructions:
You order coffee for the user or get the weather of a location they ask for.If asked for multiple tasks, you call the relevant tools in order.
Input:
[
  {
    "content": "Order me a double shot cappuccino.",
    "role": "user"
  }
]
Tools:
[
  {
    "name": "get_weather",
    "parameters": {
      "properties": {"input": {"title": "Input", "type": "string"}},
      "required": ["input"],
      "title": "get_weather_args",
      "type": "object",
      "additionalProperties": false
    },
    "strict": true,
    "type": "function",
    "description": "You get the weather of a given location."
  },
  {
    "name": "order_coffee",
    "parameters": {
      "properties": {"input": {"title": "Input", "type": "string"}},
      "required": ["input"],
      "title": "order_coffee_args",
      "type": "object",
      "additionalProperties": false
    },
    "strict": true,
    "type": "function",
    "description": "You order coffee from a local online coffee shop."
  }
]

LLM resp:
[
  {
    "arguments": "{\"input\":\"double shot cappuccino\"}",
    "call_id": "call_rAG3Hhwlp0kW2SGrtH9YRJTs",
    "name": "order_coffee",
    "type": "function_call",
    "id": "fc_68074aa3274481918488cb02f02f658700acb907174cbe73",
    "status": "completed"
  }
]

Invoking tool order_coffee with input {"input":"double shot cappuccino"}

Running agent Coffee Agent (turn 1)

Calling LLM gpt-4o -
Instructions:
You order coffee from a local online coffee shop.
Input:
[
  {
    "content": "double shot cappuccino",
    "role": "user"
  }
]
Tools:
[
  {
    "name": "order_coffee",
    "parameters": {
      "properties": {
        "coffee_name": {
          "description": "The name of the coffee drink to order.",
          "title": "Coffee Name",
          "type": "string"
        }
      },
      "required": ["coffee_name"],
      "title": "order_coffee_args",
      "type": "object",
      "additionalProperties": false
    },
    "strict": true,
    "type": "function",
    "description": "Order coffee from an online coffee shop"
  }
]

LLM resp:
[
  {
    "arguments": "{\"coffee_name\":\"double shot cappuccino\"}",
    "call_id": "call_FIS0b4T0fKQcHQzGew2TDh4q",
    "name": "order_coffee",
    "type": "function_call",
    "id": "fc_68074aa44c44819182183d37e235af0d0ff1c5780d96b2be",
    "status": "completed"
  }
]

Invoking tool order_coffee with input {"coffee_name":"double shot cappuccino"}

Running agent Coffee Agent (turn 2)

Calling LLM gpt-4o -
Instructions:
You order coffee from a local online coffee shop.
Input:
[
  {
    "content": "double shot cappuccino",
    "role": "user"
  },
  {
    "arguments": "{\"coffee_name\":\"double shot cappuccino\"}",
    "call_id": "call_FIS0b4T0fKQcHQzGew2TDh4q",
    "name": "order_coffee",
    "type": "function_call",
    "id": "fc_68074aa44c44819182183d37e235af0d0ff1c5780d96b2be",
    "status": "completed"
  },
  {
    "call_id": "call_FIS0b4T0fKQcHQzGew2TDh4q",
    "output": "code=200 status='OK' order_id=111",
    "type": "function_call_output"
  }
]
Tools:
[...]

LLM resp:
[
  {
    "id": "msg_68074aa542a48191bb11e008829598ce0ff1c5780d96b2be",
    "content": [
      {
        "annotations": [],
        "text": "Your order for a double shot cappuccino has been placed successfully! If you need anything else, just let me know.",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]

Running agent Orchestrator Agent (turn 2)

Calling LLM gpt-4o -
Instructions:
You order coffee for the user or get the weather of a location they ask for.If asked for multiple tasks, you call the relevant tools in order.
Input:
[
  {
    "content": "Order me a double shot cappuccino.",
    "role": "user"
  },
  {
    "arguments": "{\"input\":\"double shot cappuccino\"}",
    "call_id": "call_rAG3Hhwlp0kW2SGrtH9YRJTs",
    "name": "order_coffee",
    "type": "function_call",
    "id": "fc_68074aa3274481918488cb02f02f658700acb907174cbe73",
    "status": "completed"
  },
  {
    "call_id": "call_rAG3Hhwlp0kW2SGrtH9YRJTs",
    "output": "Your order for a double shot cappuccino has been placed successfully! If you need anything else, just let me know.",
    "type": "function_call_output"
  }
]
Tools:
[...]

LLM resp:
[
  {
    "id": "msg_68074aa644f48191a523122dd910a7cc00acb907174cbe73",
    "content": [
      {
        "annotations": [],
        "text": "Your order for a double shot cappuccino has been placed successfully! If you need anything else, just let me know.",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]


Second input
------------
{
  "content": "I am in Seattle right now. If it is less than 15 degrees then order me a hot double shot cappuccino, otherwise order me a nitro cold brew.",
  "role": "user"
}

{
  "arguments": "{\"input\":\"Seattle\"}",
  "call_id": "call_EsREGS0mGWSfcSQ3GiUOgNsu",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_68075395128c8191b5709a664747662106724a36e5f49893",
  "status": "completed"
}

{
  "call_id": "call_EsREGS0mGWSfcSQ3GiUOgNsu",
  "output": "The current temperature in Seattle is 5.7\u00b0C.",
  "type": "function_call_output"
}

{
  "arguments": "{\"input\":\"hot double shot cappuccino\"}",
  "call_id": "call_Ian3b2UWMN32CSlqvGZc4PZL",
  "name": "order_coffee",
  "type": "function_call",
  "id": "fc_6807539966b08191824490a67a0ef13806724a36e5f49893",
  "status": "completed"
}

{
  "call_id": "call_Ian3b2UWMN32CSlqvGZc4PZL",
  "output": "Your order for a hot double shot cappuccino has been placed successfully! Enjoy your coffee! \u2615",
  "type": "function_call_output"
}

{
  "id": "msg_6807539cd0288191a517335f49081d5306724a36e5f49893",
  "content": [
    {
      "annotations": [],
      "text": "The temperature in Seattle is 5.7\u00b0C, so I ordered you a hot double shot cappuccino. Enjoy! \u2615",
      "type": "output_text"
    }
  ],
  "role": "assistant",
  "status": "completed",
  "type": "message"
}
"""


@function_tool
def get_weather(latitude: float, longitude: float) -> int:
    """
    Get the current temperature for provided coordinates in celsius.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.

    Returns:
        The temperature of the given location. The temperature is in Celsius units.
    """
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


class OrderStatus(BaseModel):
    code: int
    status: str
    order_id: int


@function_tool
def order_coffee(coffee_name: str) -> OrderStatus:
    """Order coffee from an online coffee shop

    Args:
        coffee_name: The name of the coffee drink to order.

    Returns:
        The status of the order. If the order was successful a status code of 200
        and status message of "OK" along with the order id is returned. Otherwise,
        the status code will be something else.
    """
    print(f"Ordering {coffee_name}")
    return OrderStatus(code=200, status="OK", order_id=111)


async def main():
    weather_agent = Agent(
        name="Weather Agent",
        instructions="You get the weather of any given location.",
        tools=[get_weather],
    )

    coffee_agent = Agent(
        name="Coffee Agent",
        instructions="You order coffee from a local online coffee shop.",
        tools=[order_coffee],
    )

    weather_agent_tool = weather_agent.as_tool(
        tool_name="get_weather",
        tool_description="You get the weather of a given location.",
    )

    coffee_agent_tool = coffee_agent.as_tool(
        tool_name="order_coffee",
        tool_description="You order coffee from a local online coffee shop.",
    )

    orchestrator_agent = Agent(
        name="Orchestrator Agent",
        instructions=(
            "You order coffee for the user or get the weather of a location they ask for."
            "If asked for multiple tasks, you call the relevant tools in order."
        ),
        tools=[weather_agent_tool, coffee_agent_tool],
    )

    # resp = await Runner.run(orchestrator_agent, "Order me a double shot cappuccino.")
    resp = await Runner.run(
        orchestrator_agent,
        "I am in Seattle right now. If it is less than 15 degrees then order me a hot double shot cappuccino, otherwise order me a nitro cold brew.",
    )
    for input_ in resp.to_input_list():
        print(json.dumps(input_, indent=2))
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
