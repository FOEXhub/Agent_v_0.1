import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="qwen/qwq-32b:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)