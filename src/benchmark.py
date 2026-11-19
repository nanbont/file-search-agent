import csv
import json
import time

from agent import SYSTEM_PROMPT, client, parse_tool_call, run_tool

# Model ID -> (input price, output price) in $ per 1M tokens
MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
}

# (question, should the tool be used?)
QUESTIONS = [
    ("Hi, how are you?", False),
    ("What is quantization?", False),
    ("How many Python files are there in the current directory?", True),
    ("Which Markdown files do we have in the docs folder?", True),
    ("Where is README.md?", True),
    ("Are there any files with retry in their name?", True),
]

RUNS = 3


def call_model(model, messages, temperature=None):
    """Stream one model call. Return the text, the time to first token, and the token usage."""
    start = time.perf_counter()
    ttft = None
    parts = []
    usage = None

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = delta.content
        thinking = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if ttft is None and (text or thinking):
            ttft = time.perf_counter() - start
        if text:
            parts.append(text)

    return "".join(parts), ttft, usage


def run_question(model, history, question):
    """Run one question through the agent loop and measure it."""
    start = time.perf_counter()
    history.append({"role": "user", "content": question})

    text, ttft, usage = call_model(model, history, temperature=0)
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0

    call = parse_tool_call(text)
    if call is None:
        history.append({"role": "assistant", "content": text.strip()})
    else:
        result = run_tool(call)
        history.append({"role": "assistant", "content": text})
        history.append({"role": "user", "content": f"Tool result: {json.dumps(result)}. Answer the user's question concisely."})

        final_text, _, final_usage = call_model(model, history)
        if final_usage:
            tokens_in += final_usage.prompt_tokens
            tokens_out += final_usage.completion_tokens
        history.append({"role": "assistant", "content": final_text.strip()})

    return {
        "e2el": time.perf_counter() - start,
        "ttft": ttft or 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_used": call is not None,
    }


def average(rows, key):
    return sum(row[key] for row in rows) / len(rows)


if __name__ == "__main__":
    rows = []
    repeated_calls = {model: 0 for model in MODELS}

    for model, (price_in, price_out) in MODELS.items():
        print(f"\nBenchmarking {model} ...")
        for run in range(1, RUNS + 1):
            for question, should_use_tool in QUESTIONS:
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                result = run_question(model, history, question)
                cost = (result["tokens_in"] * price_in + result["tokens_out"] * price_out) / 1e6
                rows.append({
                    "model": model,
                    "run": run,
                    "question": question,
                    "e2el": round(result["e2el"], 2),
                    "ttft": round(result["ttft"], 2),
                    "tokens_in": result["tokens_in"],
                    "tokens_out": result["tokens_out"],
                    "cost": round(cost, 6),
                    "tool_used": result["tool_used"],
                    "tool_correct": result["tool_used"] == should_use_tool,
                })
                print(f"  run {run}: {question[:40]:40} {result['e2el']:.2f}s")

            # Follow-up test: the second question should be answered from memory
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            run_question(model, history, "How many Python files are there?")
            follow_up = run_question(model, history, "Which of those are tests?")
            if follow_up["tool_used"]:
                repeated_calls[model] += 1

    with open("results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Model':38} {'E2EL':>6} {'TTFT':>6} {'In':>6} {'Out':>6} {'Cost':>10} {'Tool ok':>8} {'Repeats':>8}")
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        correct = sum(row["tool_correct"] for row in model_rows)
        print(
            f"{model.split('/')[-1]:38} "
            f"{average(model_rows, 'e2el'):6.2f} "
            f"{average(model_rows, 'ttft'):6.2f} "
            f"{average(model_rows, 'tokens_in'):6.0f} "
            f"{average(model_rows, 'tokens_out'):6.0f} "
            f"{average(model_rows, 'cost'):10.6f} "
            f"{correct:>3}/{len(model_rows):<4} "
            f"{repeated_calls[model]:>3}/{RUNS}"
        )