"""Tool calling with ChatTrtLlm via `bind_tools` (non-streaming).

Works on tool-capable models (Llama-3.1, Qwen2.5-Coder, Mistral). Tool calling is non-streaming;
keep tool turns non-streaming (the backend doesn't parse tool calls while streaming).

Prereq: a backend running with a tool-capable model loaded (e.g. llama-3_1-8b-fp16).
Run: uv run python examples/tool_calling.py
"""

from langchain_core.tools import tool

from trt_llm_langchain import ChatTrtLlm


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def main() -> None:
    chat = ChatTrtLlm(max_tokens=128, temperature=0).bind_tools([add])
    msg = chat.invoke("What is 2 + 3? Use the add tool.")

    for call in msg.tool_calls:
        print(f"{call['name']}({call['args']})")  # add({'a': 2, 'b': 3})
        if call["name"] == "add":
            print("result:", add.invoke(call["args"]))


if __name__ == "__main__":
    main()
