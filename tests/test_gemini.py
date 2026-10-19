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
  #  api_key=os.getenv("GEMINI_API_KEY")
)

print(response.choices[0].message.content)