import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

response = completion(
    model="gemini/gemini-3.8-flash",
    messages=[
        {
            "role": "user",
            "content": "Are you working?"
        }
    ],
    api_key=os.getenv("AQ.Ab8RN6K0f81J1WHofdUhbgQ4k5xhm7GN3fkRpcPxQfyM1mk-Pw")
)

print(response.choices[0].message.content)