"""Minimal ChatTrtLlm usage — the drop-in swap for ChatAnthropic.

Prereq: a trt-llm-explore-style backend running (OpenAI proxy on :8003, KServe on :8000).
Run: uv run python examples/basic_chat.py
"""

from trt_llm_langchain import ChatTrtLlm


def main() -> None:
    # chat = ChatAnthropic(model="claude-sonnet-4-6")
    chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16", temperature=0.2, max_tokens=128)

    response = chat.invoke("Write a one-line Python function to compute factorial.")
    print(response.content)


if __name__ == "__main__":
    main()
