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

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about files in a local project folder.

You can use one tool:
search_files(name_pattern: str, subpath: str = ".") -> a list of file paths whose names match a glob pattern
- name_pattern matches file names only, for example "*.py", "README.md", "*retry*"
- subpath is a folder inside the project, for example "docs". Use "." for the whole project.

If the question is about files in the project, respond with ONLY this JSON, no other text:
{"tool": "search_files", "arguments": {"name_pattern": "<pattern>", "subpath": "<folder>"}}
Otherwise, answer normally."""


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


question = "Which files are in the secrets folder?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]

# 1. The model decides whether to use the tool
resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
raw = resp.choices[0].message.content
print("Model said:", raw)

# 2. Is it a tool call or a normal answer?
call = parse_tool_call(raw)

if call is None:
    print("Answer:", raw.strip())
else:
    result = run_tool(call)
    print("Tool result:", result)

    # 3. Send the result back and get the final answer
    messages.append({"role": "assistant", "content": raw})
    messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}. Answer the user's question concisely."})

    final = client.chat.completions.create(model=MODEL, messages=messages)
    print("Answer:", final.choices[0].message.content)