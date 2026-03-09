import os
import re
import random
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

load_dotenv()

import gradio as gr
import yfinance as yf

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
# Knowledge / live info tools
# ----------------------------

@tool
def search_wikipedia(query: str) -> str:
    """Use this for timeless factual information about people, places, history, science, and general topics."""
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


@tool
def web_search(query: str) -> str:
    """Use this for up-to-date information, current events, recent developments, live facts, or anything that may have changed recently."""
    return ddg_search.run(query)


@tool
def search_arxiv(query: str) -> str:
    """Use this for scientific papers, AI research, technical topics, and academic literature."""
    return arxiv_tool.run(query)


@tool
def get_current_utc_time(_: str = "") -> str:
    """Use this when the user asks about the current time, today's date, or anything time-sensitive."""
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.isoformat()}"


# ----------------------------
# Market data tools
# ----------------------------

@tool
def get_stock_price(ticker: str) -> str:
    """
    Use this for a current or recent market snapshot for a stock, ETF, index, or crypto ticker.
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

    info = {}
    try:
        info = t.fast_info or {}
    except Exception:
        info = {}

    currency = info.get("currency", "N/A")
    exchange = info.get("exchange", "N/A")
    day_high = info.get("dayHigh", None)
    day_low = info.get("dayLow", None)
    market_cap = info.get("marketCap", None)

    lines = [
        f"Ticker: {ticker}",
        f"Latest close: {last_close:.4f} {currency}" if currency != "N/A" else f"Latest close: {last_close:.4f}",
        f"Exchange: {exchange}",
    ]

    if previous_close is not None and previous_close != 0:
        pct = ((last_close - previous_close) / previous_close) * 100
        lines.append(f"Change vs previous close: {pct:+.2f}%")

    if day_high is not None:
        lines.append(f"Day high: {day_high}")
    if day_low is not None:
        lines.append(f"Day low: {day_low}")
    if market_cap is not None:
        lines.append(f"Market cap: {market_cap}")

    return "\n".join(lines)


@tool
def get_stock_history(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Use this for recent or historical market performance.
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

    start_close = float(hist.iloc[0]["Close"])
    end_close = float(hist.iloc[-1]["Close"])
    pct_change = ((end_close - start_close) / start_close) * 100 if start_close != 0 else 0.0

    tail = hist.tail(5)[["Open", "High", "Low", "Close", "Volume"]].round(4)

    return (
        f"Ticker: {ticker}\n"
        f"Period: {period}\n"
        f"Interval: {interval}\n"
        f"Start close: {start_close:.4f}\n"
        f"End close: {end_close:.4f}\n"
        f"Performance over selected period: {pct_change:+.2f}%\n\n"
        f"Last 5 rows:\n{tail.to_string()}"
    )


# ----------------------------
# Chaos tool
# ----------------------------

@tool
def wikipedia_chaos_oracle(query: str) -> str:
    """Use this only when the user explicitly wants a weird, chaotic, prophetic, or surreal Wikipedia-style response."""
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
    "wikipedia_chaos_oracle": wikipedia_chaos_oracle,
}


def build_agent(selected_tool_names, user_message: str = ""):
    selected_tools = [ALL_TOOLS[name] for name in selected_tool_names if name in ALL_TOOLS]

    llm = ChatOpenAI(model="gpt-4.1-nano")

    lower_msg = user_message.lower()

    dynamic_hint = ""
    if any(word in lower_msg for word in ["stock", "stocks", "market", "share price", "ticker", "etf", "index", "crypto", "bitcoin"]):
        dynamic_hint = (
            "This looks like a market-data question. Prefer get_stock_price or get_stock_history when possible. "
            "If the user asks about recent market news or developments, also use web_search."
        )
    elif any(word in lower_msg for word in ["today", "latest", "recent", "currently", "now", "this week", "this year"]):
        dynamic_hint = (
            "This looks time-sensitive. Use get_current_utc_time and web_search instead of relying on memory."
        )
    elif any(word in lower_msg for word in ["paper", "research", "study", "arxiv", "scientific"]):
        dynamic_hint = (
            "This looks research-oriented. Prefer search_arxiv."
        )
    elif any(word in lower_msg for word in ["chaos", "oracle", "weird", "prophecy", "chaotic", "surreal"]):
        dynamic_hint = (
            "The user explicitly wants a chaotic answer. Prefer wikipedia_chaos_oracle."
        )
    else:
        dynamic_hint = (
            "For timeless facts, prefer search_wikipedia. For changing or recent facts, prefer web_search."
        )

    agent = create_agent(
        model=llm,
        tools=selected_tools,
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use tools according to these rules: "
            "For arithmetic, use the math tools. "
            "For timeless factual questions, use search_wikipedia. "
            "For current, recent, live, or updated information, use web_search and get_current_utc_time when relevant. "
            "For stocks, ETFs, indexes, and crypto tickers, use get_stock_price or get_stock_history. "
            "For scientific papers and technical research, use search_arxiv. "
            "For weird oracle-style answers, use wikipedia_chaos_oracle. "
            "Do not answer time-sensitive questions from memory when a relevant enabled tool exists. "
            "If a needed tool is unavailable, say so plainly. "
            "Use only the tools needed. "
            "In the final answer, briefly mention which tools were used. "
            f"{dynamic_hint}"
        ),
    )
    return agent


def run_agent(message, history, selected_tools):
    if history is None:
        history = []

    if not message or not str(message).strip():
        return history, "No input provided.", ""

    if not selected_tools:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "No tools are enabled. Please check at least one tool."})
        return history, "No tools enabled.", ""

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
                if len(content_str) > 1200:
                    content_str = content_str[:1200] + "..."
                tool_lines.append(f"  → {content_str}")

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

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})

    return history, tool_trace, ""


tool_names = list(ALL_TOOLS.keys())

with gr.Blocks(title="Math & Knowledge Assistant") as demo:
    gr.Markdown(
        "# 🧮 Math & Knowledge Assistant\n"
        "Ask math questions, look up facts, search the web, check market data, or unleash a weird chaos-oracle remix.\n\n"
        "**Tip:** All tools are enabled by default, but you can turn individual tools off."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=450)

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder=(
                        "e.g. What is 15 multiplied by 8? "
                        "What is the latest news on OpenAI? "
                        "How has AAPL performed over 1 month? "
                        "Who invented calculus?"
                    ),
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
                lines=18,
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
            ["Who was Alan Turing?"],
            ["What is the latest news on Nvidia?"],
            ["What is happening in AI this week?"],
            ["What is the current price of AAPL?"],
            ["How has TSLA performed over 6 months?"],
            ["Give me recent information about the S&P 500."],
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
