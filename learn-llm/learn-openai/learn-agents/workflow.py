import asyncio
import json
import logging

from agents import Agent, Runner
from dotenv import load_dotenv
from pydantic import BaseModel
from snippets import configure_logger  # type: ignore

load_dotenv()

configure_logger(logging.DEBUG, chatty_modules=["httpcore"])

"""
I took the workflow example and converted it into a agents-as-tools example. For a simple enough
workflow that does not require a high degree of precision, this is good enough.
"""


story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
)

story_outline_agent_tool = story_outline_agent.as_tool(
    tool_name="story_outline_agent_tool",
    tool_description="Generates a short story outline.",
)


class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool


outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions="Read the given story outline, and judge the quality. Also, determine if it is a scifi story.",
    output_type=OutlineCheckerOutput,
)

outline_checker_agent_tool = outline_checker_agent.as_tool(
    tool_name="outline_checker_agent_tool",
    tool_description="Judges the quality of a given story outline. Also determines whether the outline is for a scifi story.",
)

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    output_type=str,
)

story_agent_tool = story_agent.as_tool(
    tool_name="story_agent_tool",
    tool_description="Outputs a short story based on the given outline.",
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=(
        "Write a short story based on user's input. First generate an outline."
        "If the outline is good and for a scifi story, then write the full story."
        "Otherwise, tell the user that outline was not good quality and stop."
    ),
    tools=[story_outline_agent_tool, outline_checker_agent_tool, story_agent_tool],
)


async def main():
    input_prompt = input("What kind of story do you want?")
    resp = await Runner.run(starting_agent=writer_agent, input=input_prompt)
    for input_ in resp.to_input_list():
        try:
            print(json.dumps(input_, indenty=2))
        except TypeError:
            print(input_)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
