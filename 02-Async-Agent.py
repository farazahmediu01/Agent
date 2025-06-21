from agents import OpenAIChatCompletionsModel, AsyncOpenAI, Agent, Runner
from agents import set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
import asyncio
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

# Global level configuration
set_default_openai_client(external_client)
set_tracing_disabled(True)


async def main():
    agent = Agent(
        name="Assistant", instructions="You are a helpful assistant", model=model
    )
    # Runner.run runs asynchronously.
    result = await Runner.run(agent, "Hello How are you?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
