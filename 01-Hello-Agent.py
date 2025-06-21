from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner
from agents import set_default_openai_client, set_tracing_disabled

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

# global level configuration
set_default_openai_client(external_client)
set_tracing_disabled(True)


agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model)
result = Runner.run_sync(agent, "Hello How are you?")
print(result.final_output)
