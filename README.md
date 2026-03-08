---
title: Math and Knowledge Assistant
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Math & Knowledge Assistant

A conversational AI agent that can perform mathematical operations and look up factual information via Wikipedia.

## Features
- Addition, subtraction, multiplication, division
- Wikipedia search for factual questions
- Powered by GPT-4.1-nano via LangGraph ReAct agent
- Shows tool usage trace for each response

## Setup

### Running locally
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Run:
   ```bash
   python app.py
   ```

### Deploying to HuggingFace Spaces
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Gradio** SDK.
2. Push this repository to the Space.
3. In the Space **Settings → Variables and secrets**, add:
   - `OPENAI_API_KEY` → your OpenAI API key (as a **Secret**)
