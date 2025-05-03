import asyncio
import logging

from agents import Agent, RunContextWrapper, Runner, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel
from snippets import configure_logger  # type: ignore

load_dotenv()
configure_logger(logging.DEBUG, chatty_modules=["httpcore"])

"""
Calling LLM gpt-4o -
Instructions:
You send user notifications upon request
Input:
[
  {
    "content": "Notify user about the latest sale with 50% off on new phones.",
    "role": "user"
  }
]
Tools:
[
  {
    "name": "notify",
    "parameters": {
      "properties": {
        "title": {
          "description": "The title of the notification. In an email, this will be the subject. In a text",
          "title": "Title",
          "type": "string"
        },
        "message": {
          "description": "The full message of the notification. In an email, this will be the body. In a text",
          "title": "Message",
          "type": "string"
        }
      },
      "required": [
        "title",
        "message"
      ],
      "title": "notify_args",
      "type": "object",
      "additionalProperties": false
    },
    "strict": true,
    "type": "function",
    "description": "Notify a user via email or text\n\nIf the user provided in the context has an email address, then they are notified via email.\nOtherwise, their phone number is used to notify them via text."
  }
]

LLM resp:
[
  {
    "arguments": "{\"title\":\"Exclusive 50% Off Sale!\",\"message\":\"Great news! Enjoy an exclusive 50% off on our latest range of new phones. Don't miss out on this limited-time offer to upgrade your tech. Shop now and save big!\"}",
    "call_id": "call_ysxDxb1bZZmniBb4GHkOW6Rj",
    "name": "notify",
    "type": "function_call",
    "id": "fc_6805cf0eb74481919f89cb87f88db1c4009a49e041735c05",
    "status": "completed"
  }
]

Calling LLM gpt-4o -
Instructions:
You send user notifications upon request
Input:
[
  {
    "content": "Notify user about the latest sale with 50% off on new phones.",
    "role": "user"
  },
  {
    "arguments": "{\"title\":\"Exclusive 50% Off Sale!\",\"message\":\"Great news! Enjoy an exclusive 50% off on our latest range of new phones. Don't miss out on this limited-time offer to upgrade your tech. Shop now and save big!\"}",
    "call_id": "call_ysxDxb1bZZmniBb4GHkOW6Rj",
    "name": "notify",
    "type": "function_call",
    "id": "fc_6805cf0eb74481919f89cb87f88db1c4009a49e041735c05",
    "status": "completed"
  },
  {
    "call_id": "call_ysxDxb1bZZmniBb4GHkOW6Rj",
    "output": "code=200 status='OK'",
    "type": "function_call_output"
  }
]
Tools:
[...same as before...]

LLM resp:
[
  {
    "id": "msg_6805cf1004748191ba3be3a49a1f8fa0009a49e041735c05",
    "content": [
      {
        "annotations": [],
        "text": "The user has been successfully notified about the 50% off sale on new phones.",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]
"""


class User(BaseModel):
    name: str
    email: str | None
    uid: int
    phone: str | None


class NotifyStatus(BaseModel):
    code: int
    status: str


@function_tool(docstring_style="google")
def notify(ctx: RunContextWrapper[User], title: str, message: str) -> NotifyStatus:
    """Notify a user via email or text

    If the user provided in the context has an email address, then they are notified via email.
    Otherwise, their phone number is used to notify them via text.

    Args:
        title: The title of the notification. In an email, this will be the subject. In a text
        message, this will just be the first line.
        message: The full message of the notification. In an email, this will be the body. In a text
        message, this will just be the rest of the message.

    Retunrs:
        The notification status. If the notification was sent successfully it will be
        {code: 200, status: "OK"}. Upon failure the return will have a different code and status.
    """
    user: User = ctx.context
    if user.email:
        print(f"Sending email to {user.email} with subject: {title}")
        return NotifyStatus(code=200, status="OK")
    elif user.phone:
        print(f"Sending text message to {user.phone} with title {title}")
        return NotifyStatus(code=200, status="OK")
    else:
        return NotifyStatus(code=400, status="NOT OK")


async def main():
    notification_agent = Agent(
        name="Notification Agent",
        instructions="You send user notifications upon request",
        tools=[notify],
    )
    user = User(name="APTG", email="avilay@gmail.com", uid=1, phone=None)
    result = await Runner.run(
        notification_agent,
        input="Notify user about the latest sale with 50% off on new phones.",
        context=user,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
