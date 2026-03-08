import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper
import gradio as gr


# --- Tools ---

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
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
    """Search Wikipedia for factual information about a topic."""
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers, search_wikipedia]

llm = ChatOpenAI(model="gpt-4.1-nano")

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful assistant that can perform mathematical operations and look up information. "
        "Use tools precisely and explain your reasoning clearly. "
        "Only call a tool when it is strictly necessary. "
        "Do not make exploratory or speculative tool calls."
    ),
)


# --- Agent invocation ---

def run_agent(message: str, history: list) -> tuple[list, str]:
    response = agent.invoke({"messages": [("human", message)]})

    tool_lines = []
    for msg in response["messages"]:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_lines.append(f"▶ {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            tool_lines.append(f"  → {msg.content}")

    final_answer = response["messages"][-1].content
    tool_trace = "\n".join(tool_lines) if tool_lines else "No tools used."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})
    return history, tool_trace, ""


# --- Gradio UI ---

with gr.Blocks(title="Math & Knowledge Assistant") as demo:
    gr.Markdown(
        "# 🧮 Math & Knowledge Assistant\n"
        "Ask math questions (add, subtract, multiply, divide) or look up facts via Wikipedia."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=450)
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="e.g. What is 15 multiplied by 8? or Who invented calculus?",
                    label="Your question",
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            tool_box = gr.Textbox(
                label="Tool usage trace",
                lines=14,
                interactive=False,
                placeholder="Tool calls will appear here...",
            )

    send_btn.click(run_agent, [user_input, chatbot], [chatbot, tool_box, user_input])
    user_input.submit(run_agent, [user_input, chatbot], [chatbot, tool_box, user_input])

    gr.Examples(
        examples=[
            "What is 256 divided by 16?",
            "Add 1337 and 42",
            "What is the capital of Japan?",
            "Who was Alan Turing?",
            "Multiply 99 by 99",
        ],
        inputs=user_input,
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
