import os
import re
import json
import uuid
import random
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
load_dotenv()

import gradio as gr
import yfinance as yf
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun


# ----------------------------
# Shared wrappers
# ----------------------------

ddg_search = DuckDuckGoSearchRun()

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=1200,
    )
)

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ----------------------------
# Helper functions
# ----------------------------

def save_line_chart(title, x_values, y_values, x_label="X", y_label="Y"):
    """
    Save a line chart and return the file path.
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must be non-empty and the same length.")

    filename = os.path.join(CHART_DIR, f"{uuid.uuid4().hex}.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x_values, y_values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    return filename


def try_extract_chart_payload(text):
    """
    Extract chart JSON from a tool output if present.
    We use markers to make parsing robust.
    """
    if not text:
        return None

    start_marker = "CHART_DATA_START"
    end_marker = "CHART_DATA_END"

    if start_marker not in text or end_marker not in text:
        return None

    try:
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker)
        payload_text = text[start:end].strip()
        payload = json.loads(payload_text)

        required = {"title", "x", "y", "x_label", "y_label"}
        if not required.issubset(payload.keys()):
            return None

        return payload
    except Exception:
        return None


def detect_chartworthy_request(message: str) -> bool:
    text = message.lower()
    keywords = [
        "trend", "over time", "performance", "history", "historical", "chart",
        "graph", "plot", "results", "compare", "comparison", "this month",
        "this year", "last month", "last year", "6 months", "1 year"
    ]
    return any(k in text for k in keywords)


# ----------------------------
# Math tools
# ----------------------------

@tool
def add_numbers(a: float, b: float) -> float:
    """Use this tool for addition."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Use this tool for subtraction."""
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Use this tool for multiplication."""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Use this tool for division."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


# ----------------------------
# Knowledge tools
# ----------------------------

@tool
def search_wikipedia(query: str) -> str:
    """Use this for stable factual information about people, places, history, science, and concepts."""
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


@tool
def web_search(query: str) -> str:
    """Use this for recent or up-to-date information, such as current events, latest developments, and changing facts."""
    return ddg_search.run(query)


@tool
def search_arxiv(query: str) -> str:
    """Use this for scientific papers, research topics, and academic literature."""
    return arxiv_tool.run(query)


@tool
def get_current_utc_time(_: str = "") -> str:
    """Use this when the user asks about the current time, today's date, or other time-sensitive context."""
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.isoformat()}"


# ----------------------------
# Market tools
# ----------------------------

@tool
def get_stock_price(ticker: str) -> str:
    """
    Use this for a recent quote snapshot for a stock, ETF, index, or crypto ticker.
    Examples: AAPL, MSFT, TSLA, SPY, BTC-USD, ^GSPC
    """
    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", interval="1d", auto_adjust=False)

    if hist is None or hist.empty:
        return f"No recent market data found for ticker '{ticker}'."

    last_row = hist.iloc[-1]
    last_close = float(last_row["Close"])

    previous_close = None
    if len(hist) >= 2:
        previous_close = float(hist.iloc[-2]["Close"])

    lines = [
        f"Ticker: {ticker}",
        f"Latest close: {last_close:.4f}",
    ]

    if previous_close is not None and previous_close != 0:
        pct = ((last_close - previous_close) / previous_close) * 100
        lines.append(f"Change vs previous close: {pct:+.2f}%")

    return "\n".join(lines)


@tool
def get_stock_history(ticker: str, period: str = "6mo", interval: str = "1d") -> str:
    """
    Use this for historical market performance and trends.
    It also returns chart data for the app to render automatically.
    Valid examples:
    - ticker='AAPL', period='1mo', interval='1d'
    - ticker='^GSPC', period='6mo', interval='1d'
    - ticker='BTC-USD', period='1y', interval='1d'
    """
    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=False)

    if hist is None or hist.empty:
        return f"No historical market data found for ticker '{ticker}' with period='{period}' and interval='{interval}'."

    hist = hist.reset_index()

    date_col = hist.columns[0]
    x_vals = []
    for d in hist[date_col]:
        try:
            x_vals.append(str(d.date()))
        except Exception:
            x_vals.append(str(d))

    y_vals = [float(v) for v in hist["Close"]]

    start_close = y_vals[0]
    end_close = y_vals[-1]
    pct_change = ((end_close - start_close) / start_close) * 100 if start_close != 0 else 0.0

    payload = {
        "title": f"{ticker} closing price ({period})",
        "x": x_vals,
        "y": y_vals,
        "x_label": "Date",
        "y_label": "Close price",
    }

    summary = (
        f"Ticker: {ticker}\n"
        f"Period: {period}\n"
        f"Interval: {interval}\n"
        f"Start close: {start_close:.4f}\n"
        f"End close: {end_close:.4f}\n"
        f"Performance over selected period: {pct_change:+.2f}%\n\n"
        f"CHART_DATA_START\n"
        f"{json.dumps(payload)}\n"
        f"CHART_DATA_END"
    )

    return summary


# ----------------------------
# Chart tool
# ----------------------------

@tool
def generate_line_chart(title: str, x_values: list, y_values: list, x_label: str = "X", y_label: str = "Y") -> str:
    """Use this to generate a line chart when the user asks for a chart, graph, plot, trend, or time-based numeric visualization."""
    path = save_line_chart(
        title=title,
        x_values=x_values,
        y_values=y_values,
        x_label=x_label,
        y_label=y_label,
    )
    return f"Chart saved to: {path}"


# ----------------------------
# Chaos tool
# ----------------------------

@tool
def wikipedia_chaos_oracle(query: str) -> str:
    """Use this only when the user explicitly wants a weird, chaotic, prophetic, or surreal Wikipedia-style answer."""
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
    "web_search": web_search,
    "search_arxiv": search_arxiv,
    "get_current_utc_time": get_current_utc_time,
    "get_stock_price": get_stock_price,
    "get_stock_history": get_stock_history,
    "generate_line_chart": generate_line_chart,
    "wikipedia_chaos_oracle": wikipedia_chaos_oracle,
}


def build_agent(selected_tool_names, user_message: str = ""):
    selected_tools = [ALL_TOOLS[name] for name in selected_tool_names if name in ALL_TOOLS]
    llm = ChatOpenAI(model="gpt-4.1-nano")

    lower_msg = user_message.lower()

    dynamic_hint = ""
    if any(word in lower_msg for word in ["chart", "graph", "plot"]):
        dynamic_hint = (
            "The user explicitly wants a visual. If chart data is available, generate a line chart."
        )
    elif any(word in lower_msg for word in ["stock", "stocks", "market", "ticker", "etf", "index", "crypto", "bitcoin"]):
        dynamic_hint = (
            "This looks like market data. Prefer get_stock_price or get_stock_history. "
            "If the user asks about performance, trends, or history, use get_stock_history."
        )
    elif any(word in lower_msg for word in ["today", "latest", "recent", "currently", "now", "this week", "this year"]):
        dynamic_hint = (
            "This looks time-sensitive. Use web_search and get_current_utc_time when relevant."
        )
    elif any(word in lower_msg for word in ["paper", "research", "study", "arxiv", "scientific"]):
        dynamic_hint = "This looks research-oriented. Prefer search_arxiv."
    elif any(word in lower_msg for word in ["chaos", "oracle", "weird", "prophecy", "chaotic", "surreal"]):
        dynamic_hint = "The user wants a chaotic answer. Prefer wikipedia_chaos_oracle."
    else:
        dynamic_hint = (
            "For stable facts use search_wikipedia. For changing facts use web_search."
        )

    agent = create_agent(
        model=llm,
        tools=selected_tools,
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use tools according to these rules: "
            "For arithmetic, use the math tools. "
            "For stable factual questions, use search_wikipedia. "
            "For current or updated information, use web_search and get_current_utc_time when helpful. "
            "For stock, ETF, index, and crypto price/history questions, use get_stock_price or get_stock_history. "
            "For research papers and academic literature, use search_arxiv. "
            "For weird oracle-style answers, use wikipedia_chaos_oracle. "
            "When the user asks about performance, results over time, trends, comparisons, or a chart/graph/plot, generate a chart if relevant chart data is available and the chart tool is enabled. "
            "Do not answer time-sensitive questions from memory when a relevant enabled tool exists. "
            "If a needed tool is unavailable, say so plainly. "
            "Use only the tools needed. "
            "Briefly mention which tools were used in the final answer. "
            f"{dynamic_hint}"
        ),
    )
    return agent


def run_agent(message, history, selected_tools):
    if history is None:
        history = []

    if not message or not str(message).strip():
        return history, "No input provided.", "", None

    if not selected_tools:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "No tools are enabled. Please check at least one tool."})
        return history, "No tools enabled.", "", None

    chart_path = None

    try:
        agent = build_agent(selected_tools, message)

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
                content_str = str(content)

                payload = try_extract_chart_payload(content_str)
                if payload:
                    try:
                        chart_path = save_line_chart(
                            title=payload["title"],
                            x_values=payload["x"],
                            y_values=payload["y"],
                            x_label=payload["x_label"],
                            y_label=payload["y_label"],
                        )
                    except Exception as chart_error:
                        tool_lines.append(f"  → Chart generation failed: {chart_error}")

                shortened = content_str
                if len(shortened) > 1200:
                    shortened = shortened[:1200] + "..."
                tool_lines.append(f"  → {shortened}")

        if messages:
            last_message = messages[-1]
            final_answer = getattr(last_message, "content", "No response generated.")
            if isinstance(final_answer, list):
                final_answer = str(final_answer)
        else:
            final_answer = "No response generated."

        # Optional fallback: if user clearly asked for a chart and chart tool exists,
        # but agent only used get_stock_history (which returned payload),
        # the app-side auto-chart already handled it.
        # So no extra fallback is needed here.

        tool_trace = "\n".join(tool_lines) if tool_lines else "No tools used."

    except Exception as e:
        final_answer = f"Error: {e}"
        tool_trace = "Execution failed."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})

    return history, tool_trace, "", chart_path


tool_names = list(ALL_TOOLS.keys())

with gr.Blocks(title="Math, Knowledge & Charts Assistant") as demo:
    gr.Markdown(
        "# 📈 Math, Knowledge & Charts Assistant\n"
        "Ask math questions, look up facts, search the web, check market data, and auto-generate charts for trends and results.\n\n"
        "**Tip:** All tools are enabled by default, but you can turn individual tools off."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=450)

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder=(
                        "e.g. What is 15 multiplied by 8? "
                        "How has AAPL performed over 6 months? "
                        "Show me a chart of BTC-USD over 1 year. "
                        "Who invented calculus?"
                    ),
                    label="Your question",
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            chart_output = gr.Image(
                label="Generated chart",
                type="filepath",
                height=320
            )

        with gr.Column(scale=1):
            enabled_tools = gr.CheckboxGroup(
                choices=tool_names,
                value=tool_names,
                label="Enabled tools",
                info="Uncheck tools to prevent the agent from using them.",
            )

            tool_box = gr.Textbox(
                label="Tool usage trace",
                lines=18,
                interactive=False,
                placeholder="Tool calls will appear here...",
            )

    send_btn.click(
        run_agent,
        inputs=[user_input, chatbot, enabled_tools],
        outputs=[chatbot, tool_box, user_input, chart_output],
    )

    user_input.submit(
        run_agent,
        inputs=[user_input, chatbot, enabled_tools],
        outputs=[chatbot, tool_box, user_input, chart_output],
    )

    gr.Examples(
        examples=[
            ["What is 256 divided by 16?"],
            ["Who was Alan Turing?"],
            ["What is the latest news on Nvidia?"],
            ["How has AAPL performed over 6 months?"],
            ["Show me a chart of TSLA over 1 year."],
            ["Plot BTC-USD performance over 6 months."],
            ["Find recent research papers about retrieval-augmented generation."],
            ["Give me a weird chaotic oracle reading about black holes."],
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
