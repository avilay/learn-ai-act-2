import asyncio
import logging

import requests
from agents import Agent, Runner, function_tool
from snippets import configure_logger  # type: ignore

configure_logger(logging.DEBUG, chatty_modules=["httpcore"])

"""
LLM gpt-4o -
Instructions:
You are a helpful agent.
Input:
[
  {
    "content": "What's the weather in Kolkata?",
    "role": "user"
  }
]
Tools:
[
  {
    "name": "get_weather",
    "parameters": {
      "properties": {
        "latitude": {
          "description": "Latitude of the location.",
          "title": "Latitude",
          "type": "number"
        },
        "longitude": {
          "description": "Longitude of the location.",
          "title": "Longitude",
          "type": "number"
        }
      },
      "required": [
        "latitude",
        "longitude"
      ],
      "title": "get_weather_args",
      "type": "object",
      "additionalProperties": false
    },
    "strict": true,
    "type": "function",
    "description": "Get the current temperature for provided coordinates in celsius."
  }
]

LLM resp:
[
  {
    "arguments": "{\"latitude\":22.5726,\"longitude\":88.3639}",
    "call_id": "call_TI6ySGlcD9BU1GfAHaPBmJUc",
    "name": "get_weather",
    "type": "function_call",
    "id": "fc_6805c2b3a890819191e084b7a2552a260b996e38ea107d62",
    "status": "completed"
  }
]


Calling LLM gpt-4o -
Instructions:
You are a helpful agent.
Input:
[
  {
    "content": "What's the weather in Kolkata?",
    "role": "user"
  },
  {
    "arguments": "{\"latitude\":22.5726,\"longitude\":88.3639}",
    "call_id": "call_TI6ySGlcD9BU1GfAHaPBmJUc",
    "name": "get_weather",
    "type": "function_call",
    "id": "fc_6805c2b3a890819191e084b7a2552a260b996e38ea107d62",
    "status": "completed"
  },
  {
    "call_id": "call_TI6ySGlcD9BU1GfAHaPBmJUc",
    "output": "32.0",
    "type": "function_call_output"
  }
]
Tools:
[...same as before...]

LLM resp:
[
  {
    "id": "msg_6805c2b5459c81918c7cf48a2c54e6b50b996e38ea107d62",
    "content": [
      {
        "annotations": [],
        "text": "The current temperature in Kolkata is 32\u00b0C.",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]
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


async def main():
    agent = Agent(
        name="Weather Agent",
        instructions="You are a helpful agent.",
        tools=[get_weather],
    )
    result = await Runner.run(agent, input="What's the weather in Kolkata?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
