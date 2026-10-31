
from ollama import chat

response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": 'Return only this JSON: {"status": "connected"}'
        }
    ],
    format="json"
)

print(response.message.content)
