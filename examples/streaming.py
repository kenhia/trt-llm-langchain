"""Token streaming with ChatTrtLlm via `.stream()`.

Prereq: a trt-llm-explore-style backend running (OpenAI proxy on :8003, KServe on :8000).
Run: uv run python examples/streaming.py
"""

from trt_llm_langchain import ChatTrtLlm


def main() -> None:
    chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16", max_tokens=128)

    for chunk in chat.stream("Count from 1 to 5, one number per line."):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
