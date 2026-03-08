import sys
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper

# IGNORE IF YOU ARE NOT RUNNING LOCALLY
add
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=os.getenv("OPENAI_API_KEY"),
)

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def new_subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information about a topic.
    
    Parameters:
    - query (str): The topic or question to search for on Wikipedia
    
    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

# Update your tools list to include the Wikipedia tool
tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers, search_wikipedia]

llm_ai = ChatOpenAI(model="gpt-4.1-nano")


# Create the agent with all tools including Wikipedia
math_agent_updated = create_react_agent(
    model=llm_ai,
    tools=tools_updated,
    prompt="You are a helpful assistant that can perform various mathematical operations and look up information. Use the tools precisely and explain your reasoning clearly. Only call a tool when it is strictly necessary for the answer. Do not make exploratory or speculative tool calls."
)


query = "What is five multiplied by seven?"

response = math_agent_updated.invoke({"messages": [("human", query)]})

print("\nMessage sequence:")
for i, msg in enumerate(response["messages"]):
    print(f"\n--- Message {i+1} ---")
    print(f"Type: {type(msg).__name__}")
    if hasattr(msg, 'content'):
        print(f"Content: {msg.content}")
    if hasattr(msg, 'name'):
        print(f"Name: {msg.name}")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"Tool calls: {msg.tool_calls}")



# Add a bit of extra information about which tools used
for message in response["messages"]:
    if message.type == "tool":
        print(f"\n[AI tool response: {message.name}]: {message.content}")
    elif message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            print(f"\n[AI calling tool: {tc['name']}] with args: {tc['args']}")
    else:
        print(f"\n[{message.type.upper()}]: {message.content}")



