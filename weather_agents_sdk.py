from agents import OpenAIChatCompletionsModel, AsyncOpenAI, Agent, Runner, function_tool
from agents import set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
import requests
import asyncio
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not Found...!")

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


@function_tool
def get_weather(city: str):
    # TODO!: Do an actual API Call
    print("🔨 Tool Called: get_weather", city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


async def main(prompt: str):
    agent = Agent(
        name="weather agent",
        instructions="You are a weather agent. You are given with the name of city and you have to tell the current tempreature in degree celcious.",
        model=model,
        tools=[get_weather],
    )

    # Runner.run runs asynchronously.
    result = await Runner.run(agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    prompt = input("Wether Agent Activated: Ask anything about weather> ")
    asyncio.run(main(prompt))
