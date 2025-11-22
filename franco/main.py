from langchain_featherless_ai import ChatFeatherlessAi
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("FEATHERLESS_API_KEY")
api_url = "https://api.featherless.ai/v1"
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

llm = ChatFeatherlessAi(
    api_key=api_key,
    base_url=api_url,
)

system_prompt = "You are a helpful assistant that translates English to French. Translate the user sentence."
user_prompt = "I love programming."
messages = [
    ("system", system_prompt),
    ("human", user_prompt),
]

response = llm.invoke(
    messages,
    model=model,
    temperature=0.7,
    seed=42,
).content

print(response)
