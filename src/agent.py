import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import search_files

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are a file assistant for a local project folder. Keep every answer short: 1-3 sentences, plain text, no headings or long explanations.

You have one tool:
search_files(name_pattern: str, subpath: str = ".") -> list of file paths whose names match a glob pattern
- name_pattern matches file names only, for example "*.py", "README.md", "*retry*"
- subpath is a folder inside the project, for example "docs". Use "." for the whole project.

Rules:
- Use the tool only for questions about files in this project.
- Do not use the tool for greetings or general knowledge questions. Answer those directly and briefly.
- If an earlier tool result in this conversation already answers the question, reuse it instead of searching again.
- When counting files, count the items in the tool result exactly.
- When you use the tool, respond with ONLY this JSON, no other text:
{"tool": "search_files", "arguments": {"name_pattern": "<pattern>", "subpath": "<folder>"}}"""

def parse_tool_call(raw):
    """Return the tool call as a dict, or None if the reply is a normal answer."""
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        call = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(call, dict) or call.get("tool") != "search_files":
        return None
    return call


def run_tool(call):
    """Run search_files with the model's arguments. Return an error message instead of crashing."""
    arguments = call.get("arguments", {})
    try:
        return search_files(**arguments)
    except (ValueError, TypeError) as error:
        return f"Error: {error}"


history = [{"role": "system", "content": SYSTEM_PROMPT}]


def ask(user_text):
    """Send one user message, run the tool if needed, and return the final answer."""
    history.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(model=MODEL, messages=history, temperature=0)
    raw = resp.choices[0].message.content or ""
    call = parse_tool_call(raw)

    if call is None:
        answer = (raw or "").strip()
        history.append({"role": "assistant", "content": answer})
        return answer

    result = run_tool(call)
    print(f"  [tool] search_files({call.get('arguments')}) -> {result}")

    history.append({"role": "assistant", "content": raw})
    history.append({"role": "user", "content": f"Tool result: {json.dumps(result)}. Answer the user's question concisely."})

    final = client.chat.completions.create(model=MODEL, messages=history)
    answer = (final.choices[0].message.content or "").strip()
    history.append({"role": "assistant", "content": answer})
    return answer


if __name__ == "__main__":
    print("File search agent. Type 'exit' to quit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in ("exit", "quit"):
            break
        if not user_text:
            continue
        print("Agent:", ask(user_text))