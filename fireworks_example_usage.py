from fireworks import Fireworks
from dotenv import load_dotenv


load_dotenv()

client = Fireworks()

response = client.chat.completions.create(
  model="accounts/fireworks/models/llama-v3p3-70b-instruct",
  messages=[{
    "role": "user",
    "content": "Say hello in Spanish",
  }],
)

print(response.choices[0].message.content)