import os
import re
import random
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
import gradio as gr


# -----------------------------
# Tools
# -----------------------------

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


@tool
def wikipedia_chaos_oracle(query: str) -> str:
    """
    Search Wikipedia and turn the result into a bizarre oracle-style remix.
    Useful for weird demo output.
    """
    wiki = WikipediaAPIWrapper()
    text = wiki.run(query)

    if not text or not text.strip():
        return "The chaos oracle found only silence."

    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return "The chaos oracle stared into the void and found no words."

    unique_words = list(dict.fromkeys(words))
    random.shuffle(unique_words)

    chosen = unique_words[: min(18, len(unique_words))]
    reversed_words = [w[::-1] for w in chosen[:5]]
    shouted_words = [w.upper() for w in chosen[5:10]]
    whispered_words = [w.lower() for w in chosen[10:15]]

    fragments = []
    if reversed_words:
        fragments.append("mirror: " + ", ".join(reversed_words))
    if shouted_words:
        fragments.append("prophecy: " + " | ".join(shouted_words))
    if whispered_words:
        fragments.append("whispers: " + " ~ ".join(whispered_words))

    chaos_score = sum(ord(c) for c in "".join(chosen[:8])) % 1000

    first_chunk = text[:220].strip().replace("\n", " ")
    if len(text) > 220:
        first_chunk += "..."

    return (
        f"CHAOS ORACLE REPORT FOR: {query}\n\n"
        f"Raw glimpse:\n{first_chunk}\n\n"
        f"{chr(10).join(fragments)}\n\n"
        f"chaos_score={chaos_score}\n"
        f"omen: The encyclopedia has been disturbed."
    )


ALL_TOOLS = {
    "add_numbers": add_numbers,
    "subtract_numbers": subtract_numbers,
    "multiply_numbers": multiply_numbers,
    "divide_numbers": divide_numbers,
    "search_wikipedia": search_wikipedia,
    "wikipedia_chaos_oracle": wikipedia_chaos_oracle,
}


# -----------------------------
# Agent factory
# -----------------------------

def build_agent(selected_tool_names):
    selected_tools = [ALL_TOOLS[name] for name in selected_tool_names if name in ALL_TOOLS]

    llm = ChatOpenAI(model="gpt-4.1-nano")

    agent = create_agent(
        model=llm,
        tools=selected_tools,
        system_prompt=(
            "You are a helpful assistant that can perform mathematical operations and look up information. "
            "Use tools precisely and explain your reasoning clearly. "
            "Only call a tool when it is strictly necessary. "
            "Respect the enabled tool list. "
            "If a needed tool is unavailable, say so plainly. "
            "The wikipedia_chaos_oracle tool is weird and playful; only use it when the user explicitly asks for something strange, chaotic, playful, bizarre, or experimental."
        ),
    )
    return agent


# -----------------------------
# Agent invocation
# -----------------------------

def run_agent(message, history, selected_tools):
    if history is None:
        history = []

    if not message or not str(message).strip():
        return history, "No input provided.", ""

    if not selected_tools:
        history.append((message, "No tools are enabled. Please check at least one tool."))
        return history, "No tools enabled.", ""

    try:
        agent = build_agent(selected_tools)

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            }
        )

        tool_lines = []
        messages = response.get("messages", [])

        for msg in messages:
            msg_type = getattr(msg, "type", None)

            if msg_type == "ai" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    tool_lines.append(f"▶ {tool_name}({tool_args})")

            elif msg_type == "tool":
                content = getattr(msg, "content", "")
                tool_lines.append(f"  → {content}")

        if messages:
            last_message = messages[-1]
            final_answer = getattr(last_message, "content", "No response generated.")
            if isinstance(final_answer, list):
                final_answer = str(final_answer)
        else:
            final_answer = "No response generated."

        tool_trace = "\n".join(tool_lines) if tool_lines else "No tools used."

    except Exception as e:
        final_answer = f"Error: {e}"
        tool_trace = "Execution failed."

    history.append((message, final_answer))
    return history, tool_trace, ""


# -----------------------------
# Gradio UI
# -----------------------------

tool_names = list(ALL_TOOLS.keys())

with gr.Blocks(title="Math & Knowledge Assistant") as demo:
    gr.Markdown(
        "# 🧮 Math & Knowledge Assistant\n"
        "Ask math questions, look up facts via Wikipedia, or unleash a weird chaos-oracle remix.\n\n"
        "**Tip:** All tools are enabled by default, but you can turn individual tools off."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=450)

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="e.g. What is 15 multiplied by 8? Who invented calculus? Give me a chaotic Wikipedia prophecy about Alan Turing.",
                    label="Your question",
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            enabled_tools = gr.CheckboxGroup(
                choices=tool_names,
                value=tool_names,
                label="Enabled tools",
                info="Uncheck tools to prevent the agent from using them.",
            )

            tool_box = gr.Textbox(
                label="Tool usage trace",
                lines=16,
                interactive=False,
                placeholder="Tool calls will appear here...",
            )

    send_btn.click(
        run_agent,
        inputs=[user_input, chatbot, enabled_tools],
        outputs=[chatbot, tool_box, user_input],
    )

    user_input.submit(
        run_agent,
        inputs=[user_input, chatbot, enabled_tools],
        outputs=[chatbot, tool_box, user_input],
    )

    gr.Examples(
        examples=[
            ["What is 256 divided by 16?"],
            ["Add 1337 and 42"],
            ["What is the capital of Japan?"],
            ["Who was Alan Turing?"],
            ["Multiply 99 by 99"],
            ["Give me a weird chaotic oracle reading about black holes."],
            ["Use the chaos oracle on Norway."],
        ],
        inputs=[user_input],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft()
    )
